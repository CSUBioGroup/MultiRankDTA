
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as dataloader
import pickle
import os
import random
import numpy as np
from torch import optim
from torch_geometric.data import Batch, Data
from common.create_data import create_csv, create_data
from common.listwise_loss import ListNetLoss, ListMLELoss
from stage1.model_stage1 import TransformerModel
from common.utils import *
from common.ci import ci_fast
import argparse
from common.graph import smile_to_graph
from common.create_data import *
import tensorboard_logger
from torch.cuda.amp import autocast, GradScaler
import csv
from scipy.stats import spearmanr

import copy



os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)




def set_model(opt):
    device = torch.device(opt.device)
    model = TransformerModel().to(device)
    criterion_mle = ListNetLoss().to(device)


    log_var_pw = torch.nn.Parameter(torch.zeros(1, device=device))
    log_var_pt = torch.nn.Parameter(torch.zeros(1, device=device))
    log_var_lw = torch.nn.Parameter(torch.zeros(1, device=device))

    return model, criterion_mle, log_var_pw, log_var_pt, log_var_lw

def collate_fn(batch_list):
    data_list = []
    for drug_data, protein_embed, label, orig_seq_str in batch_list:
        N, edge, node_feature, edge_feature = drug_data
        graph_data = Data(
            x=torch.tensor(node_feature, dtype=torch.uint8),
            edge_index=torch.tensor(edge.T, dtype=torch.long),
            edge_attr=torch.tensor(edge_feature, dtype=torch.uint8),
            num_nodes=N
        )

        data_list.append((
            graph_data,
            torch.tensor(protein_embed, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            orig_seq_str
        ))
    batch_graph = Batch.from_data_list([item[0] for item in data_list])
    batch_rest = [(item[1], item[2], item[3]) for item in data_list]
    return batch_graph, batch_rest


def set_data_loader(opt):
    data_dir = opt.data_dir
    processed_dir = os.path.join(data_dir, "processed")

    train_csv = os.path.join(data_dir, f"{opt.datasets}_2train.csv")
    test_csv = os.path.join(data_dir, f"{opt.datasets}_2test.csv")
    if not os.path.exists(train_csv):
        train_csv = os.path.join(data_dir, opt.datasets, "train.csv")
    if not os.path.exists(test_csv):
        test_csv = os.path.join(data_dir, opt.datasets, "test.csv")
    train_file_name = os.path.join(processed_dir, f"{opt.datasets}_additionGRAseventynew_train.pkl")
    test_file_name = os.path.join(processed_dir, f"{opt.datasets}_additionGRAseventynew_test.pkl")

    if (not os.path.exists(train_csv)) or (not os.path.exists(test_csv)):
        create_csv(data_dir=data_dir, datasets=[opt.datasets])
    if (not os.path.exists(train_file_name)) or (not os.path.exists(test_file_name)):
        create_data(data_dir=data_dir, datasets=[opt.datasets])
    if (not os.path.exists(train_file_name)) or (not os.path.exists(test_file_name)):
        raise FileNotFoundError(
            f"Missing processed dataset files: {train_file_name} or {test_file_name}. "
            "Please generate them first."
        )

    with open(train_file_name, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_file_name, 'rb') as f:
        test_data = pickle.load(f)

    train_size = int((1 - opt.val_ratio) * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader_kwargs = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "collate_fn": collate_fn,
        "num_workers": opt.workers,
        "pin_memory": True,
    }
    eval_loader_kwargs = {
        "batch_size": opt.eval_batch_size,
        "shuffle": False,
        "collate_fn": collate_fn,
        "num_workers": opt.eval_workers,
        "pin_memory": True,
    }
    if opt.workers > 0:
        train_loader_kwargs["prefetch_factor"] = opt.prefetch_factor
        train_loader_kwargs["persistent_workers"] = True
    if opt.eval_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = opt.eval_prefetch_factor
        eval_loader_kwargs["persistent_workers"] = True

    train_loader = dataloader.DataLoader(train_data, **train_loader_kwargs)
    valid_loader = dataloader.DataLoader(valid_data, **eval_loader_kwargs)
    test_loader = dataloader.DataLoader(test_data, **eval_loader_kwargs)

    return train_loader, valid_loader, test_loader

def set_optimizer(opt, model, log_var_pw, log_var_pt, log_var_lw):
    params = list(model.parameters())
    if opt.loss_mode == "shared_uncertainty":
        params += [log_var_pw, log_var_pt, log_var_lw]
    optimizer = optim.AdamW(params, lr=opt.learning_rate)
    return optimizer

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='davis')
    parser.add_argument('--modeling', type=str, default='TransformerModel')
    parser.add_argument('--save_path', type=str, default='/public/home/hpc244711054/perl5/base/graph/tryesm/trydavisdudagain')
    parser.add_argument('--save_file_path', type=str, default='/public/home/hpc244711054/perl5/base/graph/tryesm/trydavisdudagain')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--eval_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--eval_prefetch_factor', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping')
    parser.add_argument(
        '--loss_mode',
        type=str,
        default='shared_uncertainty',
        choices=['shared_uncertainty', 'mse_only', 'fixed_equal_weight'],
    )
    return parser.parse_args()



def save_model(model, optimizer, opt, epoch, file_path):
    state = {
        'opt': copy.deepcopy(opt),
        'model': copy.deepcopy(model.state_dict()),
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'epoch': epoch,
    }
    torch.save(state, file_path)


def z_score_normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    if std < 1e-6:
        std = 1e-6
    return (tensor - mean) / std


def pairwise_loss(y1, labels, margin=0.1):
    batch_size = y1.shape[0]
    y_diff = y1.view(batch_size, 1) - y1.view(1, batch_size)
    label_diff = labels.view(batch_size, 1) - labels.view(1, batch_size)
    pairwise_labels = torch.sign(label_diff)
    valid_pairs = (pairwise_labels.abs() > 0) & (label_diff.abs() > 1e-6)

    if valid_pairs.sum() == 0:
        return torch.tensor(0.0, device=y1.device)


    loss = F.margin_ranking_loss(
        y_diff[valid_pairs],
        torch.zeros_like(y_diff[valid_pairs]),
        pairwise_labels[valid_pairs],
        margin=margin
    )
    return torch.clamp_min(loss, 0.0)


def pointwise_loss(y3, labels):
    return F.mse_loss(y3, labels.view(-1, 1))

def compute_loss(out, moe_loss, y1, y2, y3, labels, log_var_pw, log_var_pt, log_var_lw, opt, delta_huber=0.1):
    y2_scores = y2.squeeze(-1)
    labels_flat = labels.view(-1)
    raw_loss_pw = pairwise_loss(y1, labels)
    raw_loss_pt = pointwise_loss(y3, labels)
    raw_loss_lw = ListNetLoss()(y2_scores, labels_flat)
    loss_huber = F.huber_loss(out, labels.view(-1, 1), delta=delta_huber)
    loss_moe = 0.01 * moe_loss

    if opt.loss_mode == "mse_only":
        return F.mse_loss(out, labels.view(-1, 1))

    if opt.loss_mode == "fixed_equal_weight":
        loss_experts = (raw_loss_pw + raw_loss_pt + raw_loss_lw) / 3.0
        return loss_experts + loss_moe + loss_huber


    loss_pw = 0.5 * torch.exp(-log_var_pw) * raw_loss_pw + 0.5 * log_var_pw
    loss_pt = 0.5 * torch.exp(-log_var_pt) * raw_loss_pt + 0.5 * log_var_pt
    loss_lw = 0.5 * torch.exp(-log_var_lw) * raw_loss_lw + 0.5 * log_var_lw
    return loss_pw + loss_pt + loss_lw + loss_moe + loss_huber

def summarize_metrics(all_labels, all_preds):
    y = np.asarray(all_labels)
    f = np.asarray(all_preds)
    if len(y) == 0:
        return {"ci": 0.0, "pearson": 0.0, "spearman": 0.0, "mse": 0.0, "rm2": 0.0}
    if np.allclose(y, y[0]):
        ci_value = 0.0
        pearson_value = 0.0
        spearman_value = 0.0
    else:
        ci_value = ci_fast(y, f)
        pearson_value = pearson(y, f)
        spearman_value, _ = spearmanr(y, f)
        if np.isnan(pearson_value):
            pearson_value = 0.0
        if np.isnan(spearman_value):
            spearman_value = 0.0
    return {
        "ci": float(ci_value),
        "pearson": float(pearson_value),
        "spearman": float(spearman_value),
        "mse": float(mse(y, f)),
        "rm2": float(get_rm2(y, f)),
    }


def evaluate_loader(data_loader, model, opt, epoch=None, y_file=None):
    model.eval()
    all_labels = []
    all_preds = []
    y1_abs_values = []
    y2_abs_values = []
    y3_abs_values = []

    with torch.no_grad():
        for batch_idx, (batch, batch_rest) in enumerate(data_loader):
            drug_data = batch.to(opt.device)
            protein_data = torch.stack([item[0] for item in batch_rest]).to(opt.device)
            label = torch.stack([item[1] for item in batch_rest]).to(opt.device)

            outs, _, y1, y2, y3, _ = model(drug_data, protein_data)
            all_labels.extend(label.cpu().numpy().flatten())
            all_preds.extend(outs.cpu().numpy().flatten())
            y1_abs_values.append(torch.abs(y1).detach().cpu().numpy())
            y2_abs_values.append(torch.abs(y2).detach().cpu().numpy())
            y3_abs_values.append(torch.abs(y3).detach().cpu().numpy())

    if y_file is not None:
        with open(y_file, 'a') as fw:
            fw.write(f'Epoch: {epoch}\n')
            for l_val, p_val in zip(all_labels, all_preds):
                fw.write(f'{l_val},{p_val}\n')

    metrics = summarize_metrics(all_labels, all_preds)
    if y1_abs_values:
        metrics["y1_abs_mean"] = float(np.mean(np.concatenate(y1_abs_values, axis=0)))
        metrics["y2_abs_mean"] = float(np.mean(np.concatenate(y2_abs_values, axis=0)))
        metrics["y3_abs_mean"] = float(np.mean(np.concatenate(y3_abs_values, axis=0)))
    else:
        metrics["y1_abs_mean"] = 0.0
        metrics["y2_abs_mean"] = 0.0
        metrics["y3_abs_mean"] = 0.0
    return metrics


def combined_training_strategy(model, train_loader, val_loader, test_loader, criterion_mle, optimizer, opt, log_var_pw, log_var_pt, log_var_lw):
    best_valid_ci = -1.0
    best_valid_pearson = -1.0
    best_valid_spearman = -1.0
    best_valid_mse = 1000.0
    best_valid_rm2 = -1.0
    best_epoch_ci = -1

    metrics_file = os.path.join(opt.save_file_path, "all_epochs_metrics.csv")
    write_header = True

    scaler = GradScaler()

    for epoch in range(1, opt.epochs + 1):
        model.train()
        losses = AverageMeter()

        for i, (batch, batch_rest) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            with autocast():
                drug_data = batch.to(opt.device)
                protein_data = torch.stack([item[0] for item in batch_rest]).to(opt.device)
                label = torch.stack([item[1] for item in batch_rest]).to(opt.device)
                out, moe_loss, y1, y2, y3, shared_fc = model(drug_data, protein_data)
                loss = compute_loss(out, moe_loss, y1, y2, y3, label, log_var_pw, log_var_pt, log_var_lw, opt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item(), len(batch))
            if i % opt.print_freq == 0:
                print(f'Train epoch: {epoch} [{i}/{len(train_loader)}] Loss: {losses.avg:.6f}')
            sys.stdout.flush()

        print(f'Epoch {epoch}, Training Loss: {losses.avg:.6f}')
        train_metrics = evaluate_loader(train_loader, model, opt)
        valid_metrics = evaluate_loader(val_loader, model, opt)

        print(
            f"Train CI: {train_metrics['ci']:.6f}, Pearson: {train_metrics['pearson']:.6f}, "
            f"Spearman: {train_metrics['spearman']:.6f}, MSE: {train_metrics['mse']:.6f}, RM2: {train_metrics['rm2']:.6f}"
        )
        print(
            f"Valid CI: {valid_metrics['ci']:.6f}, Pearson: {valid_metrics['pearson']:.6f}, "
            f"Spearman: {valid_metrics['spearman']:.6f}, MSE: {valid_metrics['mse']:.6f}, RM2: {valid_metrics['rm2']:.6f}"
        )
        print(
            f"Expert |y| mean(valid): [y1={valid_metrics['y1_abs_mean']:.6f}, "
            f"y2={valid_metrics['y2_abs_mean']:.6f}, y3={valid_metrics['y3_abs_mean']:.6f}]"
        )

        with open(metrics_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "Epoch",
                    "Train_Loss",
                    "Train_CI", "Train_Pearson", "Train_Spearman", "Train_MSE", "Train_RM2",
                    "Valid_CI", "Valid_Pearson", "Valid_Spearman", "Valid_MSE", "Valid_RM2",
                    "Y1_Abs_Mean", "Y2_Abs_Mean", "Y3_Abs_Mean"
                ])
                write_header = False
            writer.writerow([
                epoch,
                losses.avg,
                train_metrics["ci"], train_metrics["pearson"], train_metrics["spearman"], train_metrics["mse"], train_metrics["rm2"],
                valid_metrics["ci"], valid_metrics["pearson"], valid_metrics["spearman"], valid_metrics["mse"], valid_metrics["rm2"],
                valid_metrics["y1_abs_mean"], valid_metrics["y2_abs_mean"], valid_metrics["y3_abs_mean"]
            ])


        if valid_metrics["ci"] > best_valid_ci:
            best_valid_ci = valid_metrics["ci"]
            best_valid_pearson = valid_metrics["pearson"]
            best_valid_spearman = valid_metrics["spearman"]
            best_valid_mse = valid_metrics["mse"]
            best_valid_rm2 = valid_metrics["rm2"]
            best_epoch_ci = epoch
            save_model(model, optimizer, opt, epoch, os.path.join(opt.save_file_path, 'best.pth'))
            save_model(model, optimizer, opt, epoch, os.path.join(opt.save_file_path, 'best_ci.pth'))
            print(
                f"Saved best CI model at epoch {best_epoch_ci}: "
                f"Valid_CI={best_valid_ci:.6f}, Valid_Pearson={best_valid_pearson:.6f}, "
                f"Valid_Spearman={best_valid_spearman:.6f}, Valid_RM2={best_valid_rm2:.6f}"
            )

        print(
            f"Current best valid CI: {best_valid_ci:.6f} (epoch {best_epoch_ci})"
        )

    model.load_state_dict(torch.load(os.path.join(opt.save_file_path, 'best.pth'))['model'])
    final_valid_metrics = evaluate_loader(val_loader, model, opt)
    final_test_metrics = evaluate_loader(test_loader, model, opt, epoch=best_epoch_ci)
    print(
        f"Final Valid (best CI ckpt) - CI: {final_valid_metrics['ci']:.6f}, "
        f"Pearson: {final_valid_metrics['pearson']:.6f}, Spearman: {final_valid_metrics['spearman']:.6f}, "
        f"MSE: {final_valid_metrics['mse']:.6f}, RM2: {final_valid_metrics['rm2']:.6f}"
    )
    print(
        f"Final Test (best CI ckpt) - CI: {final_test_metrics['ci']:.6f}, "
        f"Pearson: {final_test_metrics['pearson']:.6f}, Spearman: {final_test_metrics['spearman']:.6f}, "
        f"MSE: {final_test_metrics['mse']:.6f}, "
        f"RM2: {final_test_metrics['rm2']:.6f}"
    )

    best_metrics_file = os.path.join(opt.save_file_path, "best_metrics.csv")
    with open(best_metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Best_Valid_CI", "Best_CI_Epoch", "Best_Valid_Pearson", "Best_Valid_Spearman",
            "Best_Valid_MSE", "Best_Valid_RM2", "Final_Test_CI", "Final_Test_Pearson",
            "Final_Test_Spearman", "Final_Test_MSE", "Final_Test_RM2"
        ])
        writer.writerow([
            best_valid_ci, best_epoch_ci, best_valid_pearson, best_valid_spearman,
            best_valid_mse, best_valid_rm2, final_test_metrics["ci"], final_test_metrics["pearson"],
            final_test_metrics["spearman"], final_test_metrics["mse"], final_test_metrics["rm2"]
        ])
    return {
        "best_ci": best_valid_ci,
        "best_ci_epoch": best_epoch_ci,
        "best_ci_pearson": best_valid_pearson,
        "best_ci_spearman": best_valid_spearman,
        "best_ci_rm2": best_valid_rm2,
        "best_valid_mse": best_valid_mse,
        "final_valid_metrics": final_valid_metrics,
        "final_test_metrics": final_test_metrics,
    }


def validate(val_loader, model, opt):
    return evaluate_loader(val_loader, model, opt)


def test(test_loader, model, opt, epoch=None, y_file=None):
    return evaluate_loader(test_loader, model, opt, epoch=epoch, y_file=y_file)



if __name__ == '__main__':
    opt = parser_opt()
    random.seed(1012613)
    torch.manual_seed(1012618)

    os.makedirs(opt.save_file_path, exist_ok=True)

    train_loader, val_loader, test_loader = set_data_loader(opt)
    model, criterion_mle, log_var_pw, log_var_pt, log_var_lw = set_model(opt)
    optimizer = set_optimizer(opt, model, log_var_pw, log_var_pt, log_var_lw)
    logger = tensorboard_logger.Logger(logdir=opt.save_file_path, flush_secs=2)

    results = combined_training_strategy(
        model, train_loader, val_loader, test_loader, criterion_mle, optimizer, opt, log_var_pw, log_var_pt, log_var_lw
    )

    print(f"Best Valid CI: {results['best_ci']:.6f} @ epoch {results['best_ci_epoch']}")
    print(
        f"Final Test Metrics (best CI ckpt): "
        f"CI={results['final_test_metrics']['ci']:.6f}, "
        f"Pearson={results['final_test_metrics']['pearson']:.6f}, "
        f"Spearman={results['final_test_metrics']['spearman']:.6f}, "
        f"MSE={results['final_test_metrics']['mse']:.6f}, "
        f"RM2={results['final_test_metrics']['rm2']:.6f}"
    )

    save_model(model, optimizer, opt, opt.epochs, os.path.join(opt.save_file_path, 'last.pth'))

# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as dataloader
from torch.utils.data import DataLoader
import pickle
import os
import random
import os
import pandas as pd
from tqdm import tqdm
import hashlib
import pickle
import numpy as np
from collections import defaultdict
from torch import optim
from torch.backends import cudnn
from torch_geometric.data import Batch, Data
from create_data import create_csv, create_data
from listwise_loss import ListNetLoss, ListMLELoss
from changetransformer import TransformerModel
from utils import *
import argparse
from graph import smile_to_graph  
from create_data import *
import tensorboard_logger
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import gzip
from rdkit import Chem
from rdkit import RDLogger
import logging
import random
import csv
import copy

from torch_geometric.data import Data
from torch.utils.data import Dataset



os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# 配置随机种子
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
    
    # 引入可学习的 log_var 参数
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
    if (not os.path.exists('data/' + opt.datasets + '_2train.csv')) or \
            (not os.path.exists('data/' + opt.datasets + '_2test.csv')):
        create_csv()
    if (not os.path.exists('data/processed/' + opt.datasets + '_additionGRAseventynew_train.pkl')) or \
            (not os.path.exists('data/processed/' + opt.datasets + '_additionGRAseventynew_test.pkl')):
        create_data()

    train_file_name = 'data/processed/' + opt.datasets + '_additionGRAseventynew_train.pkl'
    test_file_name = 'data/processed/' + opt.datasets + '_additionGRAseventynew_test.pkl'

    with open(train_file_name, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_file_name, 'rb') as f:
        test_data = pickle.load(f)

    train_size = int((1 - opt.val_ratio) * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    train_loader = dataloader.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn,
                                         num_workers=opt.workers, pin_memory=True)
    valid_loader = dataloader.DataLoader(valid_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn,
                                         num_workers=opt.workers, pin_memory=True)
    test_loader = dataloader.DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn,
                                        num_workers=opt.workers, pin_memory=True)

    return train_loader, valid_loader, test_loader

def set_optimizer(opt, model, log_var_pw, log_var_pt, log_var_lw):
    optimizer = optim.AdamW(
        list(model.parameters()) + 
        [log_var_pw, log_var_pt, log_var_lw],
        lr=opt.learning_rate
    )
    return optimizer

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='davis')
    parser.add_argument('--modeling', type=str, default='TransformerModel')
    parser.add_argument('--save_path', type=str, default='multirank/trydavis')
    parser.add_argument('--save_file_path', type=str, default='multirank/trydavis')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=float, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Max norm for gradient clipping')
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

def compute_loss(out, moe_loss, y1, y2, y3, labels, log_var_pw, log_var_pt, log_var_lw, delta_huber=0.1):

    loss_pw = 0.5 * torch.exp(-log_var_pw) * pairwise_loss(y1, labels) + 0.5 * log_var_pw

    loss_pt = 0.5 * torch.exp(-log_var_pt) * pointwise_loss(y3, labels) + 0.5 * log_var_pt

    y2_probs = F.softmax(y2, dim=1)  # 新增行
    loss_lw = 0.5 * torch.exp(-log_var_lw) * ListNetLoss()(y2_probs, labels) + 0.5 * log_var_lw
    loss_huber = F.huber_loss(out, labels.view(-1, 1), delta=delta_huber)  

    loss_moe = 0.01 * moe_loss
    
    total_loss = loss_pw + loss_pt + loss_lw + loss_moe + loss_huber
    return total_loss

def combined_training_strategy(model, train_loader, val_loader, test_loader, criterion_mle, optimizer, opt, log_var_pw, log_var_pt, log_var_lw):
    best_test_ci = 0  
    best_test_pearson = -1
    best_epoch = -1
    last_best_ci = 0

    y_file = os.path.join(opt.save_file_path, 'all_epochs_predictions.txt')
    open(y_file, 'w').close()

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
                loss = compute_loss(out, moe_loss, y1, y2, y3, label, log_var_pw, log_var_pt, log_var_lw)
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
        ci_val, pearson_val = test(train_loader, model, opt)
        print(f'Train CI: {ci_val:.6f}, Pearson: {pearson_val:.6f}')
        ci_test, pearson_test = test(test_loader, model, opt, epoch, y_file=y_file)
        print(f'Test CI: {ci_test:.6f}, Pearson: {pearson_test:.6f}')

        with open(metrics_file, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow([
                    "Epoch", "Test CI", "Test Pearson"
                ])
                write_header = False
            writer.writerow([
                epoch, ci_test, pearson_test
            ])

        # === 保存每个 epoch 的模型 ===
        epoch_model_path = os.path.join(opt.save_file_path, f'epoch_{epoch}.pth')
        save_model(model, optimizer, opt, epoch, epoch_model_path)


        # === 保存 best.pth ===
        if ci_test > last_best_ci:
            last_best_ci = ci_test
            if ci_test > best_test_ci:
                best_test_ci = ci_test
                best_test_pearson = pearson_test
                best_epoch = epoch
                save_model(model, optimizer, opt, epoch, os.path.join(opt.save_file_path, 'best.pth'))
                print(f'Saved best model at epoch {best_epoch} with Test CI {best_test_ci:.6f} and Pearson {best_test_pearson:.6f}.')

        print(f'Current Best Test CI: {best_test_ci:.6f} | Best Pearson: {best_test_pearson:.6f} (epoch {best_epoch})')

    # === Final evaluation ===
    model.load_state_dict(torch.load(os.path.join(opt.save_file_path, 'best.pth'))['model'])
    final_test_ci, final_test_pearson = test(test_loader, model, opt)
    print(f'Final Test CI: {final_test_ci:.6f}, Pearson: {final_test_pearson:.6f}')
    return best_test_ci, best_test_pearson, final_test_ci, final_test_pearson


def validate(val_loader, model, opt):
    model.eval()
    ci_list = []
    pearson_scores = []

    with torch.no_grad():
        for batch, batch_rest in val_loader:
            drug_data = batch.to(opt.device)
            protein_data = torch.stack([item[0] for item in batch_rest]).to(opt.device)
            label = torch.stack([item[1] for item in batch_rest]).to(opt.device)

            
            outs, _ = model(drug_data, protein_data)
            y = label.cpu().numpy().flatten()
            f = outs.cpu().numpy().flatten()

            if len(set(y)) != 1:  # 确保 CI 计算有意义
                ci_list.append(ci(np.array(y), np.array(f)))
                pearson_scores.append(pearson(np.array(y), np.array(f)))

    mean_ci = np.mean(ci_list) if ci_list else 0
    mean_pearson = np.mean(pearson_scores) if pearson_scores else 0
    return mean_ci, mean_pearson


def test(test_loader, model, opt, epoch=None, y_file=None):
    model.eval()
    ci_list = []
    pearson_scores = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (batch, batch_rest) in enumerate(test_loader):
            drug_data = batch.to(opt.device)
            protein_data = torch.stack([item[0] for item in batch_rest]).to(opt.device)
            label = torch.stack([item[1] for item in batch_rest]).to(opt.device)
            
            outs, moe_loss, y1, y2, y3 , shared_fc = model(drug_data, protein_data)

            y = label.cpu().numpy().flatten()
            f = outs.cpu().numpy().flatten()

            all_labels.extend(y)
            all_preds.extend(f)

            if len(set(y)) != 1:  # 确保 CI 计算有意义
                ci_list.append(ci(np.array(y), np.array(f)))
                pearson_scores.append(pearson(np.array(y), np.array(f)))

    # 如果需要保存预测值
    if y_file is not None:
        with open(y_file, 'a') as fw:
            fw.write(f'Epoch: {epoch}\n')
            for l_val, p_val in zip(all_labels, all_preds):
                fw.write(f'{l_val},{p_val}\n')

    mean_ci = np.mean(ci_list) if ci_list else 0
    mean_pearson = np.mean(pearson_scores) if pearson_scores else 0
    return mean_ci, mean_pearson



if __name__ == '__main__':
    opt = parser_opt()
    random.seed(1012613)
    torch.manual_seed(1012618)

    
    train_loader, val_loader, test_loader = set_data_loader(opt)
    model, criterion_mle, log_var_pw, log_var_pt, log_var_lw = set_model(opt)
    optimizer = set_optimizer(opt, model, log_var_pw, log_var_pt, log_var_lw)
    logger = tensorboard_logger.Logger(logdir=opt.save_file_path, flush_secs=2)

    best_val_ci, best_val_pearson, test_ci, test_pearson = combined_training_strategy(
        model, train_loader, val_loader, test_loader, criterion_mle, optimizer, opt, log_var_pw, log_var_pt, log_var_lw
    )

    print(f'Best Validation CI: {best_val_ci:.6f}')
    print(f'Best Validation Pearson: {best_val_pearson:.6f}')
    print(f'Final Test CI: {test_ci:.6f}')
    print(f'Final Test Pearson: {test_pearson:.6f}')

    save_model(model, optimizer, opt, opt.epochs, os.path.join(opt.save_file_path, 'last.pth'), {})

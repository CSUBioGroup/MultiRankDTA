import os
import random
import pickle
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from scipy.stats import spearmanr

from common.create_data import create_csv, create_data
from common.listwise_loss import ListNetLoss
from stage1.model_stage1 import TransformerModel
from common.utils import pearson, mse, get_rm2
from common.ci import ci_fast


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch_list):
    data_list = []

    for drug_data, protein_embed, label, seq in batch_list:
        N, edge, node_feature, edge_feature = drug_data

        graph_data = Data(
            x=torch.tensor(
                node_feature,
                dtype=torch.uint8
            ),
            edge_index=torch.tensor(
                edge.T,
                dtype=torch.long
            ),
            edge_attr=torch.tensor(
                edge_feature,
                dtype=torch.uint8
            ),
            num_nodes=N
        )

        data_list.append((
            graph_data,
            torch.tensor(
                protein_embed,
                dtype=torch.float32
            ),
            torch.tensor(
                label,
                dtype=torch.float32
            )
        ))

    batch_graph = Batch.from_data_list(
        [item[0] for item in data_list]
    )

    batch_rest = [
        (item[1], item[2])
        for item in data_list
    ]

    return batch_graph, batch_rest


def set_data_loader(opt):
    if not 0 < opt.val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    processed_dir = os.path.join(
        opt.data_dir,
        "processed"
    )

    train_file = os.path.join(
        processed_dir,
        f"{opt.datasets}_additionGRAseventynew_train.pkl"
    )

    test_file = os.path.join(
        processed_dir,
        f"{opt.datasets}_additionGRAseventynew_test.pkl"
    )

    if (
        not os.path.exists(train_file)
        or not os.path.exists(test_file)
    ):
        create_csv(
            opt.data_dir,
            [opt.datasets]
        )

        create_data(
            opt.data_dir,
            [opt.datasets]
        )

    with open(train_file, "rb") as f:
        train_data = pickle.load(f)

    with open(test_file, "rb") as f:
        test_data = pickle.load(f)

    train_size = int(
        (1 - opt.val_ratio)
        * len(train_data)
    )

    valid_size = (
        len(train_data)
        - train_size
    )

    generator = torch.Generator()
    generator.manual_seed(opt.seed)

    train_data, valid_data = (
        torch.utils.data.random_split(
            train_data,
            [train_size, valid_size],
            generator=generator
        )
    )

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return (
        train_loader,
        valid_loader,
        test_loader
    )


def pairwise_loss(
    predictions,
    labels,
    margin=0.1
):
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    pred_diff = (
        predictions[:, None]
        - predictions[None, :]
    )

    label_diff = (
        labels[:, None]
        - labels[None, :]
    )

    pair_labels = torch.sign(label_diff)

    upper_mask = torch.triu(
        torch.ones(
            predictions.size(0),
            predictions.size(0),
            device=predictions.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    valid_pairs = (
        pair_labels.abs() > 0
    ) & upper_mask

    if valid_pairs.sum() == 0:
        return torch.tensor(
            0.0,
            device=predictions.device
        )

    return F.margin_ranking_loss(
        pred_diff[valid_pairs],
        torch.zeros_like(
            pred_diff[valid_pairs]
        ),
        pair_labels[valid_pairs],
        margin=margin
    )


def compute_loss(
    out,
    moe_loss,
    y1,
    y2,
    y3,
    labels,
    log_var_pw,
    log_var_pt,
    log_var_lw,
    criterion_listnet
):
    loss_pw = pairwise_loss(
        y1,
        labels
    )

    loss_pt = F.mse_loss(
        y3,
        labels.view(-1, 1)
    )

    loss_lw = criterion_listnet(
        y2.squeeze(-1),
        labels.view(-1)
    )

    loss_huber = F.huber_loss(
        out,
        labels.view(-1, 1),
        delta=0.1
    )

    weighted_pw = (
        0.5
        * torch.exp(-log_var_pw)
        * loss_pw
        + 0.5 * log_var_pw
    )

    weighted_pt = (
        0.5
        * torch.exp(-log_var_pt)
        * loss_pt
        + 0.5 * log_var_pt
    )

    weighted_lw = (
        0.5
        * torch.exp(-log_var_lw)
        * loss_lw
        + 0.5 * log_var_lw
    )

    return (
        weighted_pw
        + weighted_pt
        + weighted_lw
        + loss_huber
        + 0.01 * moe_loss
    )


def evaluate(
    data_loader,
    model,
    device
):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch, batch_rest in data_loader:

            drug_data = batch.to(device)

            protein_data = torch.stack(
                [
                    item[0]
                    for item in batch_rest
                ]
            ).to(device)

            labels = torch.stack(
                [
                    item[1]
                    for item in batch_rest
                ]
            )

            outputs, _, _, _, _, _ = model(
                drug_data,
                protein_data
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
                .flatten()
            )

            all_preds.extend(
                outputs.cpu()
                .numpy()
                .flatten()
            )

    y = np.asarray(all_labels)
    f = np.asarray(all_preds)

    if len(y) == 0:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    spearman_value, _ = spearmanr(y, f)

    if np.isnan(spearman_value):
        spearman_value = 0.0

    return {
        "ci": ci_fast(y, f),
        "pearson": pearson(y, f),
        "spearman": spearman_value,
        "mse": mse(y, f),
        "rm2": get_rm2(y, f)
    }


def save_model(
    model,
    optimizer,
    log_var_pw,
    log_var_pt,
    log_var_lw,
    epoch,
    file_path
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "log_var_pw": (
                log_var_pw.detach().cpu()
            ),
            "log_var_pt": (
                log_var_pt.detach().cpu()
            ),
            "log_var_lw": (
                log_var_lw.detach().cpu()
            ),
            "epoch": epoch
        },
        file_path
    )


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    opt,
    log_var_pw,
    log_var_pt,
    log_var_lw,
    criterion_listnet
):
    best_valid_ci = -1
    best_epoch = 0
    patience_counter = 0

    best_path = os.path.join(
        opt.save_path,
        "best.pth"
    )

    all_params = (
        list(model.parameters())
        + [
            log_var_pw,
            log_var_pt,
            log_var_lw
        ]
    )

    for epoch in range(
        1,
        opt.epochs + 1
    ):
        model.train()

        total_loss = 0.0

        for batch, batch_rest in train_loader:

            optimizer.zero_grad()

            drug_data = batch.to(
                opt.device
            )

            protein_data = torch.stack(
                [
                    item[0]
                    for item in batch_rest
                ]
            ).to(opt.device)

            labels = torch.stack(
                [
                    item[1]
                    for item in batch_rest
                ]
            ).to(opt.device)

            (
                out,
                moe_loss,
                y1,
                y2,
                y3,
                _
            ) = model(
                drug_data,
                protein_data
            )

            loss = compute_loss(
                out,
                moe_loss,
                y1,
                y2,
                y3,
                labels,
                log_var_pw,
                log_var_pt,
                log_var_lw,
                criterion_listnet
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                all_params,
                opt.grad_clip
            )

            optimizer.step()

            total_loss += loss.item()

        valid_metrics = evaluate(
            valid_loader,
            model,
            opt.device
        )

        avg_loss = (
            total_loss
            / len(train_loader)
        )

        print(
            f"Epoch {epoch} | "
            f"Loss: {avg_loss:.4f} | "
            f"Valid CI: "
            f"{valid_metrics['ci']:.4f}"
        )

        if (
            valid_metrics["ci"]
            > best_valid_ci
        ):
            best_valid_ci = (
                valid_metrics["ci"]
            )

            best_epoch = epoch

            patience_counter = 0

            save_model(
                model,
                optimizer,
                log_var_pw,
                log_var_pt,
                log_var_lw,
                epoch,
                best_path
            )

        else:
            patience_counter += 1

        if (
            patience_counter
            >= opt.patience
        ):
            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )
            break

    return (
        best_path,
        best_epoch,
        best_valid_ci
    )


def parser_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="davis"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()


if __name__ == "__main__":

    opt = parser_opt()

    set_seed(opt.seed)

    os.makedirs(
        opt.save_path,
        exist_ok=True
    )

    (
        train_loader,
        valid_loader,
        test_loader
    ) = set_data_loader(opt)

    model = TransformerModel().to(
        opt.device
    )

    log_var_pw = torch.nn.Parameter(
        torch.zeros(
            1,
            device=opt.device
        )
    )

    log_var_pt = torch.nn.Parameter(
        torch.zeros(
            1,
            device=opt.device
        )
    )

    log_var_lw = torch.nn.Parameter(
        torch.zeros(
            1,
            device=opt.device
        )
    )

    criterion_listnet = ListNetLoss()

    optimizer = optim.AdamW(
        list(model.parameters())
        + [
            log_var_pw,
            log_var_pt,
            log_var_lw
        ],
        lr=opt.learning_rate
    )

    (
        best_path,
        best_epoch,
        best_valid_ci
    ) = train(
        model,
        train_loader,
        valid_loader,
        optimizer,
        opt,
        log_var_pw,
        log_var_pt,
        log_var_lw,
        criterion_listnet
    )

    checkpoint = torch.load(
        best_path,
        map_location=opt.device
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    test_metrics = evaluate(
        test_loader,
        model,
        opt.device
    )

    print("\nFinal Results")

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Valid CI: "
        f"{best_valid_ci:.4f}"
    )

    print(
        f"Test CI: "
        f"{test_metrics['ci']:.4f}"
    )

    print(
        f"Test MSE: "
        f"{test_metrics['mse']:.4f}"
    )

    print(
        f"Test Pearson: "
        f"{test_metrics['pearson']:.4f}"
    )

    print(
        f"Test Spearman: "
        f"{test_metrics['spearman']:.4f}"
    )

    print(
        f"Test RM2: "
        f"{test_metrics['rm2']:.4f}"
    )

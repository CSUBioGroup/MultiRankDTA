import os
import torch
import numpy as np
import joblib
import argparse
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils import pearson
from create_data import *
from changetransformer import TransformerModel
from tqdm import tqdm
import warnings  
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_absolute_error
import scipy.stats as stats
import pandas as pd
from torch.utils.data import ConcatDataset
import torch.nn.functional as F 
from collections import deque 
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import spearmanr
import csv
from sklearn.metrics import r2_score, mean_squared_error  # <-- 导入 MSE 和 R²
import copy


BEST_PTH_FILES = {
    "davismodel": "multirank/trydavis/epoch_891.pth",
    "bindingdbICmodel": "multirank/trybindingdb/epoch_669.pth",
    "kibamodel": "multirank/trykiba/epoch_859.pth"
}

TARGET_EPOCHS = {
    "davismodel": 891,
    "bindingdbICmodel": 669,
    "kibamodel": 859
}

SAVE_DIR = "multirank/trysix"
os.makedirs(SAVE_DIR, exist_ok=True)


TASK_MAP = {"davis": 0, "bindingdbIC": 1, "kiba": 2}

def ci(yp,yt):
    ind = np.argsort(yt)
    yp,yt = yp[ind],yt[ind]
    yp_diff = yp.reshape(1,-1) - yp.reshape(-1,1)
    yt_diff = yt.reshape(1,-1) - yt.reshape(-1,1)
    yp_diff,yt_diff = np.triu(yp_diff,1),np.triu(yt_diff,1)
    tmp = yp_diff[yt_diff>0].reshape(-1)
    return (1.0*(tmp>0).sum() + 0.5*(tmp==0).sum()) / len(tmp)


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


def set_data_loader(opt, dataset_name):    
    base_dir = "data/processed"
    
    train_pkl = os.path.join(base_dir, f"{dataset_name}_additionGRAseventynew_train.pkl")
    test_pkl = os.path.join(base_dir, f"{dataset_name}_additionGRAseventynew_test.pkl")

    with open(train_pkl, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_pkl, 'rb') as f:
        test_data = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1800)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    return parser.parse_args()


# Loss Functions (Huber Loss, Pairwise Loss, ListNet Loss, Uncertainty Weighted Loss)
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        is_small_error = torch.abs(error) < self.delta
        loss = torch.where(is_small_error, 0.5 * error ** 2, self.delta * (torch.abs(error) - 0.5 * self.delta))
        return torch.mean(loss)

class PairwiseLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseLoss, self).__init__()
        self.margin = margin
    
    def forward(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        y_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        label_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
        pairwise_labels = torch.sign(label_diff)
        valid_pairs = (pairwise_labels.abs() > 0) & (label_diff.abs() > 1e-6)
        
        if valid_pairs.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        
        loss = nn.functional.margin_ranking_loss(
            y_diff[valid_pairs],
            torch.zeros_like(y_diff[valid_pairs]),
            pairwise_labels[valid_pairs],
            margin=self.margin
        )
        return torch.clamp(loss, min=0.0)

class ListNetLoss(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super(ListNetLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, features, labels):
        # Top-1 损失，添加标签平滑和数值稳定性
        top1_features = F.softmax(features, dim=0)
        top1_labels = F.softmax(labels, dim=0)
        
        # 标签平滑处理
        top1_labels = top1_labels * (1 - self.label_smoothing) + self.label_smoothing / top1_labels.size(0)
        
        log_top1_features = torch.log(top1_features + 1e-10)  # 避免log(0)
        loss = torch.mean(-1 * torch.sum(top1_labels * log_top1_features))
        
        return loss


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        self.log_var_huber = nn.Parameter(torch.zeros(1))
        self.log_var_pairwise = nn.Parameter(torch.zeros(1))
        self.log_var_listwise = nn.Parameter(torch.zeros(1))

    def forward(self, loss_huber, loss_pairwise, loss_listwise):
        precision_huber = torch.exp(-self.log_var_huber)
        precision_pairwise = torch.exp(-self.log_var_pairwise)
        precision_listwise = torch.exp(-self.log_var_listwise)

        loss = (
            precision_huber * loss_huber + self.log_var_huber +
            precision_pairwise * loss_pairwise + self.log_var_pairwise +
            precision_listwise * loss_listwise + self.log_var_listwise
        )
        return loss


class MetaMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.3):
        super(MetaMLP, self).__init__()


        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  
            nn.Dropout(dropout)
        )


        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(output_dim)
        ])

        self._initialize_weights()

    def forward(self, x):
        shared_feat = self.shared(x)  # shape: (batch_size, hidden_dim)
        outputs = torch.cat([
            head(shared_feat) for head in self.task_heads  # 每个任务的预测结果
        ], dim=1)  # shape: (batch_size, output_dim)
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def extract_shared_fc():
    opt = parser_opt()
    device = torch.device(opt.device)

    for dataset_name in TASK_MAP.keys():
        print("TASK:", dataset_name)
        train_loader, test_loader = set_data_loader(opt, dataset_name)
        train_features_all_models, train_pseudo_labels_all_models = [], []
        test_features_all_models, test_pseudo_labels_all_models = [], []
        pseudo_sources = []

        def extract_features(loader):
            feature_list, pseudo_list, real_list = [], [], []
            with torch.no_grad():
                for batch in tqdm(loader):
                    batch_graph, batch_rest = batch
                    drug_data = batch_graph.to(device)
                    protein_data = torch.stack([item[0] for item in batch_rest]).to(device)
                    true_label = torch.stack([item[1] for item in batch_rest]).cpu().numpy().flatten()
                    pred_output, _, _, _, _, shared_fc = model(drug_data, protein_data)
                    feature_list.append(shared_fc.cpu().numpy())
                    pseudo_list.append(pred_output.cpu().numpy())
                    real_list.append(true_label)
            return np.vstack(feature_list), np.vstack(pseudo_list), np.concatenate(real_list)

        current_model_name = f"{dataset_name}model"
        model_path = BEST_PTH_FILES[current_model_name]
        checkpoint = torch.load(model_path, map_location=device)

        if "model" in checkpoint:
            model = TransformerModel().to(device)
            model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and TARGET_EPOCHS[current_model_name] in checkpoint:
            selected_epoch = TARGET_EPOCHS[current_model_name]
            model = TransformerModel().to(device)
            model.load_state_dict(checkpoint[selected_epoch]["model"])
        else:
            model = TransformerModel().to(device)
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        train_features, train_pseudos, train_reals = extract_features(train_loader)
        test_features, _, test_reals = extract_features(test_loader)
        train_real_labels = train_reals
        test_real_labels = test_reals

        model_names = ["davismodel", "bindingdbICmodel", "kibamodel"]
        for model_name in model_names:
            print(model_name)
            model_path = BEST_PTH_FILES[model_name]
            checkpoint = torch.load(model_path, map_location=device)

            if "model" in checkpoint:
                model = TransformerModel().to(device)
                model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict) and TARGET_EPOCHS[model_name] in checkpoint:
                selected_epoch = TARGET_EPOCHS[model_name]
                model = TransformerModel().to(device)
                model.load_state_dict(checkpoint[selected_epoch]["model"])
            else:
                model = TransformerModel().to(device)
                model.load_state_dict(checkpoint)

            model = model.to(device)
            model.eval()

            other_train_features, other_train_pseudos, _ = extract_features(train_loader)
            other_test_features, other_test_pseudos, _ = extract_features(test_loader)

            train_features_all_models.append(other_train_features)
            test_features_all_models.append(other_test_features)

            task_idx = TASK_MAP[model_name.replace("model", "")]
            train_pseudo_labels_all_models.append(other_train_pseudos)
            test_pseudo_labels_all_models.append(other_test_pseudos)
            if model_name != current_model_name:
                pseudo_sources.append(task_idx)

        train_features_concat = np.hstack(train_features_all_models)
        train_pseudo_concat = np.hstack(train_pseudo_labels_all_models)
        test_features_concat = np.hstack(test_features_all_models)
        test_pseudo_concat = np.hstack(test_pseudo_labels_all_models)

        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_train_fc.npy"), train_features_concat)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_train_pseudo.npy"), train_pseudo_concat)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_test_fc.npy"), test_features_concat)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_test_pseudo.npy"), test_pseudo_concat)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_train_real.npy"), train_real_labels)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_test_real.npy"), test_real_labels)
        np.save(os.path.join(SAVE_DIR, f"{dataset_name}_pseudo_sources.npy"), np.array(pseudo_sources))



def orthogonal_regularization(model, lambda_reg=1e-3):
    """
    对每个任务 head 的权重施加正交约束，鼓励不同任务的权重方向尽量正交。
    """
    reg_loss = 0.0
    heads = model.task_heads
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            w_i = heads[i].weight  # shape: (1, hidden_dim)
            w_j = heads[j].weight
            cosine = F.cosine_similarity(w_i, w_j, dim=1)
            reg_loss += torch.sum(cosine ** 2)  # cos²越小越正交
    return lambda_reg * reg_loss


def train_mlp():
    opt = parser_opt()
    device = torch.device(opt.device)

    # === 加载 scaler 和构造训练数据 ===
    task_real_data = {0: [], 1: [], 2: []}
    for dataset in TASK_MAP:
        real = np.load(os.path.join(SAVE_DIR, f"{dataset}_train_real.npy"))
        task_idx = TASK_MAP[dataset]
        task_real_data[task_idx].append(real)

    scalers = []
    for task_idx in range(3):
        scaler = StandardScaler()
        all_real = np.concatenate(task_real_data[task_idx])
        scaler.fit(all_real.reshape(-1, 1))
        scalers.append(scaler)
    joblib.dump(scalers, os.path.join(SAVE_DIR, "label_scalers.pkl"))

    X_all, Y_all = [], []
    for dataset in TASK_MAP:
        fc = np.load(os.path.join(SAVE_DIR, f"{dataset}_train_fc.npy"))
        pseudo = np.load(os.path.join(SAVE_DIR, f"{dataset}_train_pseudo.npy"))
        real = np.load(os.path.join(SAVE_DIR, f"{dataset}_train_real.npy"))
        pseudo_sources = np.load(os.path.join(SAVE_DIR, f"{dataset}_pseudo_sources.npy"))

        y = np.zeros((real.shape[0], 3))
        task_idx = TASK_MAP[dataset]
        y[:, task_idx] = scalers[task_idx].transform(real.reshape(-1, 1)).flatten()

        assert task_idx not in pseudo_sources
        for i, src_task_idx in enumerate(pseudo_sources):
            scaled_pseudo = scalers[src_task_idx].transform(pseudo[:, src_task_idx].reshape(-1, 1))
            y[:, src_task_idx] = scaled_pseudo.flatten()
        for i in range(3):
            pseudo[:, i] = scalers[i].transform(pseudo[:, i].reshape(-1, 1)).reshape(-1)

        X_all.append(np.hstack([fc, pseudo]))
        Y_all.append(y)

    X_train = np.vstack(X_all)
    y_train = np.vstack(Y_all)

    model = MetaMLP(X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate)
    scaler_amp = GradScaler()

    mse_loss_fn = nn.MSELoss()
    huber_loss_fn = HuberLoss()
    pairwise_loss_fn = PairwiseLoss()
    listnet_loss_fn = ListNetLoss()
    uncertainty_loss_fn = UncertaintyWeightedLoss().to(device)

    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, 
                        num_workers=0, prefetch_factor=None, pin_memory=True)

    best_ci_per_task = [0, 0, 0]
    best_avg_ci = 0

    log_path = os.path.join(SAVE_DIR, "train_eval_log.csv")
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "epoch", "total_loss", "huber_loss", "pairwise_loss", "listnet_loss", "uncertainty_weighted_loss",
            "train_spearman_0", "train_spearman_1", "train_spearman_2", "train_spearman_avg",
            "test_ci_0", "test_ci_1", "test_ci_2", "test_ci_avg",
            "test_pearson_0", "test_rmse_0", "test_spearman_0",
            "test_pearson_1", "test_rmse_1", "test_spearman_1",
            "test_pearson_2", "test_rmse_2", "test_spearman_2"
        ])

        for epoch in range(opt.epochs):
            model.train()
            total_loss = 0
            total_huber_loss = 0
            total_pairwise_loss = 0
            total_listnet_loss = 0
            total_uncertainty_loss = 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                with autocast():
                    out = model(x)
                    loss_huber = huber_loss_fn(out, y)
                    loss_pairwise = pairwise_loss_fn(out, y)
                    loss_listnet = listnet_loss_fn(out, y)
                    main_loss = uncertainty_loss_fn(loss_huber, loss_pairwise, loss_listnet)

                scaler_amp.scale(main_loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

                total_loss += main_loss.item()
                total_huber_loss += loss_huber.item()
                total_pairwise_loss += loss_pairwise.item()
                total_listnet_loss += loss_listnet.item()
                total_uncertainty_loss += main_loss.item()

            avg_loss = total_loss / len(loader)
            avg_huber_loss = total_huber_loss / len(loader)
            avg_pairwise_loss = total_pairwise_loss / len(loader)
            avg_listnet_loss = total_listnet_loss / len(loader)
            avg_uncertainty_loss = total_uncertainty_loss / len(loader)

            # === 训练集评估 ===
            model.eval()
            with torch.no_grad():
                pred_train = model(torch.tensor(X_train).float().to(device)).cpu().numpy()

            spearman_list = []
            for i in range(3):
                real = scalers[i].inverse_transform(y_train[:, i].reshape(-1, 1)).flatten()
                pred = scalers[i].inverse_transform(pred_train[:, i].reshape(-1, 1)).flatten()
                rho, _ = spearmanr(real, pred)
                spearman_list.append(rho)
            avg_spearman = np.mean(spearman_list)

            # === 测试集评估 ===
            ci_list, pearson_list, rmse_list, spearman_test_list = [], [], [], []
            for dataset, task_idx in TASK_MAP.items():
                X_test = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_fc.npy"))
                X_pseudo = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_pseudo.npy"))
                for i in range(3):
                    X_pseudo[:, i] = scalers[i].transform(X_pseudo[:, i].reshape(-1, 1)).reshape(-1)
                X_test = np.hstack([X_test, X_pseudo])
                y_test = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_real.npy"))
                scaler = scalers[task_idx]

                with torch.no_grad():
                    pred = model(torch.tensor(X_test).float().to(device)).cpu().numpy()[:, task_idx]

                pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                ci_score = ci(pred, y_test)
                pearson_score = pearson(y_test, pred)
                rmse_score = np.sqrt(mean_squared_error(y_test, pred))
                spearman_score, _ = spearmanr(y_test, pred)

                ci_list.append(ci_score)
                pearson_list.append(pearson_score)
                rmse_list.append(rmse_score)
                spearman_test_list.append(spearman_score)


                # ✅ 每个 head 保存自己最优的完整模型
                if ci_score > best_ci_per_task[task_idx]:
                    best_ci_per_task[task_idx] = ci_score
                    best_model_state = copy.deepcopy(model.state_dict())  # ✅ 强制拷贝
                    torch.save({
                        'epoch': epoch,
                        'model_state': best_model_state
                    }, os.path.join(SAVE_DIR, f"best_head{task_idx}_add.pth"))
                    print(f"[Saved] Best model for task {task_idx} (dataset: {dataset}) at epoch {epoch} with CI={ci_score:.4f}")



            avg_ci = np.mean(ci_list)
            if avg_ci > best_avg_ci:
                best_avg_ci = avg_ci
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict()
                }, os.path.join(SAVE_DIR, "best_avg_add.pth"))

            print(f"Epoch {epoch+1:04d} | Total Loss={avg_loss:.2f} | Huber Loss={avg_huber_loss:.2f} | "
                  f"Pairwise Loss={avg_pairwise_loss:.2f} | ListNet Loss={avg_listnet_loss:.2f} | "
                  f"Uncertainty Weighted Loss={avg_uncertainty_loss:.2f} | "
                  f"Train Spearman: {spearman_list} | Test CI: {ci_list} | Avg CI: {avg_ci:.4f}")

            writer.writerow([
                epoch + 1, avg_loss, avg_huber_loss, avg_pairwise_loss, avg_listnet_loss, avg_uncertainty_loss,
                *spearman_list, avg_spearman,
                *ci_list, avg_ci,
                pearson_list[0], rmse_list[0], spearman_test_list[0],
                pearson_list[1], rmse_list[1], spearman_test_list[1],
                pearson_list[2], rmse_list[2], spearman_test_list[2]
            ])


def get_input_dim():
    """
    获取 test_fc 和 test_pseudo 的拼接维度，确保 test 阶段 MLP 输入维度一致
    """
    fc_dim, pseudo_dim = None, None
    for dataset in TASK_MAP:
        fc_path = os.path.join(SAVE_DIR, f"{dataset}_test_fc.npy")
        pseudo_path = os.path.join(SAVE_DIR, f"{dataset}_test_pseudo.npy")
        if os.path.exists(fc_path) and os.path.exists(pseudo_path):
            fc = np.load(fc_path)
            pseudo = np.load(pseudo_path)
            if fc_dim is None:
                fc_dim = fc.shape[1]
                pseudo_dim = pseudo.shape[1]
            else:
                assert fc_dim == fc.shape[1], f"Inconsistent fc dim: {fc.shape[1]}"
                assert pseudo_dim == pseudo.shape[1], f"Inconsistent pseudo dim: {pseudo.shape[1]}"
    total_input_dim = fc_dim + pseudo_dim
    print(f"Detected input_dim = {fc_dim} + {pseudo_dim} = {total_input_dim}")
    return total_input_dim


def evaluate_mlp():
    opt = parser_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = joblib.load(os.path.join(SAVE_DIR, "label_scalers.pkl"))

    result_rows = []

    for pth_name in ["best_head0_add.pth", "best_head1_add.pth", "best_head2_add.pth", "best_avg_add.pth"]:
        print(f"\n========== Evaluate {pth_name} ==========")
        ckpt = torch.load(os.path.join(SAVE_DIR, pth_name), map_location=device)
        model = MetaMLP(get_input_dim()).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        for dataset in TASK_MAP:
            task_idx = TASK_MAP[dataset]

            X_fc = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_fc.npy"))
            X_pseudo = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_pseudo.npy"))
            for i in range(3):
                X_pseudo[:, i] = scalers[i].transform(X_pseudo[:, i].reshape(-1, 1)).reshape(-1)
            X = np.hstack([X_fc, X_pseudo])

            y_real = np.load(os.path.join(SAVE_DIR, f"{dataset}_test_real.npy"))
            scaler = scalers[task_idx]

            with torch.no_grad():
                out = model(torch.FloatTensor(X).to(device))

            mode = "head"
            preds = out[:, task_idx].cpu().numpy().flatten()
            preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

            ci_score = ci(preds, y_real)
            pearson_score = pearson(y_real, preds)
            mse_score = mean_squared_error(y_real, preds)
            r2_score_val = r2_score(y_real, preds)

            result_rows.append({                    
                "model_path": pth_name,
                "dataset": dataset,
                "predict_mode": mode,
                "CI": ci_score,
                "Pearson": pearson_score,
                "MSE": mse_score,
                    "R2": r2_score_val
            })

            print(f"[{dataset.upper()}] {mode.upper()} | CI={ci_score:.4f}, Pearson={pearson_score:.4f}, MSE={mse_score:.4f}, R²={r2_score_val:.4f}")

    df = pd.DataFrame(result_rows)
    df.to_csv(os.path.join(SAVE_DIR, "all_eval_results.csv"), index=False)
    print("\n Saved all_eval_results.csv")


if __name__ == '__main__':
    opt = parser_opt()
    print("extract_shared_fc() begin")
    extract_shared_fc()
    print("train_mlp() begin")
    train_mlp()
    print("train_mlp() last")
    evaluate_mlp()

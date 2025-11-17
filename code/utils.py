import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from sklearn.metrics import mean_squared_error
from torch_geometric import data as DATA
import torch

class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Pearson Correlation
def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def ci(y, f):
    # 检查 y 和 f 的长度是否一致
    if len(y) != len(f):
        raise ValueError("The lengths of 'y' and 'f' must be the same.")
    
    ind = np.argsort(y)
    
    # 检查 ind 是否超出范围
    if max(ind) >= len(f):
        print("调试信息：f 长度为", len(f), "，ind 最大值为", max(ind))
        raise IndexError("Index in 'ind' is out of bounds for 'f'")

    y = y[ind]
    f = f[ind]
    
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1

    if z == 0:
        return 0
    return S / z

# Mean Squared Error (MSE)
def mse(y, f):
    return mean_squared_error(y, f)

def top_overlap(y, f, top_k=10):
    top_k = min(top_k, len(y), len(f))
    # 根据预测分数f对索引进行排序，获取前top_k个预测值的索引
    top_pred_indices = np.argsort(f)[::-1][:top_k]
    
    # 根据真实值y对索引进行排序，获取前top_k个真实值的索引
    top_true_indices = np.argsort(y)[::-1][:top_k]

    # print(f"Top-{top_k} Predicted Indices: {top_pred_indices}")
    # print(f"Top-{top_k} True Indices: {top_true_indices}")
    
    # 计算交集的大小，即预测值和真实值在前 top_k 的重叠情况
    overlap = np.intersect1d(top_pred_indices, top_true_indices).size
    # print(f"Top-{top_k} Overlap: {overlap / top_k}")

    # 计算重叠度
    return overlap / top_k

def calculate_top_overlap(y, f):

    #计算不同K值下的Top-K重叠度，确保最大K值为数据长度

    # 动态计算数据的最大K值
    max_k = len(y)

    top1_overlap = top_overlap(y, f, top_k=1)
    top10_overlap = top_overlap(y, f, top_k=10)
    top_15_overlap = top_overlap(y, f, top_k=15)

    return {
        "top1_overlap": top_overlap(y, f, top_k=1),
        "top10_overlap": top_overlap(y, f, top_k=10),
        "top_15_overlap": top_overlap(y, f, top_k=15)
    }

# Normalized Discounted Cumulative Gain (NDCG)
def dcg_score(y, f, k=10):
    # 确保不超过y和f的长度
    k = min(len(y), k)
    
    order = np.argsort(f)[::-1]
    gains = 2 ** y[order[:k]] - 1
    discounts = np.log2(np.arange(2, k + 2))
    
    # 防止除以0
    discounts = np.maximum(discounts, 1e-10)
    return np.sum(gains / discounts)

def ndcg_score(y, f, k=10):
    actual_dcg = dcg_score(y, f, k)
    ideal_dcg = dcg_score(y, y, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

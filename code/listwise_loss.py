import torch
import torch.nn as nn
import torch.nn.functional as F

class ListNetLoss(nn.Module):
    """ListNet 排序损失函数，带数值稳定性和标签平滑"""
    
    def __init__(self, label_smoothing=0.1):
        super(ListNetLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, features, labels):
        top1_features = F.softmax(features, dim=0)
        top1_labels = F.softmax(labels, dim=0)
        
        top1_labels = top1_labels * (1 - self.label_smoothing) + self.label_smoothing / top1_labels.size(0)
        
        log_top1_features = torch.log(top1_features + 1e-10)  
        loss = torch.mean(-1 * torch.sum(top1_labels * log_top1_features))
        
        return loss


class ListMLELoss(nn.Module):
    """ListMLE 排序损失函数，改进了冗余计算和数值稳定性"""
    
    def __init__(self):
        super(ListMLELoss, self).__init__()

    def forward(self, features, labels):
        _, index = labels.sort(descending=True, dim=0)
        features_sorted_by_label = features.gather(dim=0, index=index)
        
        exp_features_sorted = features_sorted_by_label.exp()
        cum_sums = exp_features_sorted.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]) + 1e-10  # 避免除0
        
        loss = torch.mean(-1 * torch.sum(torch.log(exp_features_sorted / cum_sums)))
        
        return loss


class ListKLLoss(nn.Module):
    
    def __init__(self):
        super(ListKLLoss, self).__init__()

    def forward(self, features, labels):
        pro_labels = F.softmax(labels, dim=0)
        pro_features = F.softmax(features, dim=0)
        
        loss = torch.mean(torch.sum(pro_labels * torch.log((pro_labels / (pro_features + 1e-10)) + 1e-10)))
        
        return loss
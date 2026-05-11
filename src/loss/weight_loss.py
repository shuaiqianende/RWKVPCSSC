import torch
import torch.nn as nn
import torch.nn.functional
from torch.autograd import Variable
from typing import Iterable

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values with shape {tuple(tensor.shape)}.")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values with shape {tuple(tensor.shape)}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} is not contiguous.")

class WeightLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(WeightLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        self.alpha = torch.sqrt(torch.sqrt(self.alpha))
        self.size_average = size_average

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1, pred.size(2))  # N, H*W, C => N*H*W, C
        target = target.view(-1, 1).long()

        logpt = nn.functional.log_softmax(pred, dim=1)

        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1  * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class _FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, beta=1., size_average=True):
        super(_FocalLoss, self).__init__()
        self.gamma = gamma

        self.alpha = torch.tensor(alpha)
        self.alpha = torch.sqrt(torch.sqrt(self.alpha))
        self.alpha = self.alpha * beta

        self.size_average = size_average

    def forward(self, pred, target):
        self.alpha = self.alpha.to(pred.device)
        B, N, labels_num = pred.shape
        pred = pred.view(-1, labels_num).contiguous()  # (B, N, labels_num) -> (B * N, labels_num)
        target = target.reshape(-1, labels_num).contiguous()  # (B, N, labels_num) -> (B * N, labels_num)
        pred_sftmx = nn.functional.softmax(pred, dim=1)  # (B * N, labels_num)
        differences = torch.abs(target - pred_sftmx)  # (B * N, labels_num)
        differences = torch.clamp(differences, 0, 1.0-1e-6)
        logpt = torch.log(1 - differences)  # (B * N, labels_num)
        pt = Variable(logpt.data.exp())
        logpt = logpt * Variable(self.alpha)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = torch.sum(loss, dim=-1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

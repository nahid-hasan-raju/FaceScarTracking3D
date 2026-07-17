"""Dice + Focal combined loss — handles class imbalance well."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1.0):
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, w_dice=0.5, w_focal=0.5):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss()
        self.w_dice  = w_dice
        self.w_focal = w_focal

    def forward(self, pred, target):
        return self.w_dice * self.dice(pred, target) + self.w_focal * self.focal(pred, target)

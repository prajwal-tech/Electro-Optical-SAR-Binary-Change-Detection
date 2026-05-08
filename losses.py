"""
losses.py — Combined Focal + Dice loss for imbalanced binary segmentation.

Design rationale:
  - Class imbalance: ~87% no-change vs ~13% change pixels
  - Focal loss: down-weights easy negatives, focuses learning on hard change pixels
  - Dice loss: directly optimises the IoU-like metric, robust to imbalance
  - Combined loss gives stable training and strong F1/IoU scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    alpha: weight for positive (change) class. Higher -> penalise missed changes more.
    gamma: focusing parameter. Higher -> more focus on hard examples.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits:  (B, 1, H, W) raw (pre-sigmoid)
        targets: (B, H, W) int64 with values {0, 1}
        """
        targets_f = targets.float().unsqueeze(1)   # (B, 1, H, W)
        bce       = F.binary_cross_entropy_with_logits(logits, targets_f, reduction="none")
        probs     = torch.sigmoid(logits)
        pt        = torch.where(targets_f == 1, probs, 1 - probs)
        alpha_t   = torch.where(targets_f == 1,
                                torch.tensor(self.alpha, device=logits.device),
                                torch.tensor(1 - self.alpha, device=logits.device))
        focal     = alpha_t * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    smooth: prevents division by zero and improves gradient flow.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)                      # (B, 1, H, W)
        targets = targets.float().unsqueeze(1)               # (B, 1, H, W)
        inter   = (probs * targets).sum(dim=(2, 3))          # (B, 1)
        union   = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice    = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of Focal Loss and Dice Loss.
    focal_weight + dice_weight should sum to 1.0.
    """
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0,
                 focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal        = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice         = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight

    def forward(self, logits, targets):
        fl = self.focal(logits, targets)
        dl = self.dice(logits, targets)
        return self.focal_weight * fl + self.dice_weight * dl


def build_loss(cfg):
    lc = cfg["loss"]
    return CombinedLoss(
        focal_alpha=lc["focal_alpha"],
        focal_gamma=lc["focal_gamma"],
        focal_weight=lc["focal_weight"],
        dice_weight=lc["dice_weight"],
    )

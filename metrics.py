"""
metrics.py — Evaluation metrics for binary change detection.

Reports per the assignment:
  - IoU   (Intersection over Union)  for the Change class (label=1)
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
"""

import numpy as np
import torch


class MetricTracker:
    """Accumulates TP, FP, FN, TN across batches then computes metrics."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits:  (B, 1, H, W) raw or (B, H, W) probabilities
        targets: (B, H, W) int64 {0, 1}
        """
        if logits.dim() == 4:
            probs = torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        else:
            probs = torch.sigmoid(logits)

        preds   = (probs >= self.threshold).long()
        targets = targets.long()

        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()

    def compute(self):
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)

        return {
            "iou":       round(iou,       4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
        }

    def confusion_matrix(self):
        """Returns 2x2 numpy array [[TN, FP], [FN, TP]]."""
        return np.array([[self.tn, self.fp],
                         [self.fn, self.tp]])


def find_best_threshold(logits_list, targets_list, thresholds=None):
    """
    Sweep thresholds on the validation set and return the one
    maximising F1 for the change class.
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

    best_f1, best_thr = -1, 0.5
    for thr in thresholds:
        tracker = MetricTracker(threshold=thr)
        for logits, targets in zip(logits_list, targets_list):
            tracker.update(logits, targets)
        metrics = tracker.compute()
        if metrics["f1"] > best_f1:
            best_f1  = metrics["f1"]
            best_thr = thr

    print(f"[Threshold sweep] best threshold = {best_thr:.2f}  (F1 = {best_f1:.4f})")
    return best_thr

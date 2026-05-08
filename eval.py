#!/usr/bin/env python3
"""
eval.py — Evaluate a trained change detection model on any split.

Usage:
    # Evaluate on test set (default)
    python eval.py --config configs/config.yaml --weights checkpoints/best.pth

    # Evaluate on a custom data path
    python eval.py --config configs/config.yaml --weights checkpoints/best.pth \
                   --data_path /path/to/test --split test

    # Tune threshold on val, then evaluate on test
    python eval.py --config configs/config.yaml --weights checkpoints/best.pth \
                   --tune_threshold
"""

import argparse
import os
import sys
import json

import torch
import yaml
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset    import ChangeDetectionDataset, build_val_transforms
from models.model    import build_model
from models.losses   import build_loss
from utils.metrics   import MetricTracker, find_best_threshold
from utils.visualize import save_prediction_grid, save_confusion_matrix


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(model, loader, criterion, cfg, device, threshold, split_name,
             save_vis=True, out_dir="./eval_outputs"):
    model.eval()
    tracker    = MetricTracker(threshold=threshold)
    total_loss = 0.0
    use_amp    = cfg.get("amp", True) and device.type == "cuda"

    vis_pre, vis_post, vis_masks, vis_preds = [], [], [], []

    for batch in loader:
        pre    = batch["pre"].to(device)
        post   = batch["post"].to(device)
        mask   = batch["mask"].to(device)

        with autocast(enabled=use_amp):
            logits = model(pre, post)
            loss   = criterion(logits, mask)

        total_loss += loss.item()
        tracker.update(logits, mask)

        if save_vis and len(vis_pre) < 10:
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= threshold).long()
            for b in range(pre.shape[0]):
                if len(vis_pre) < 10:
                    vis_pre.append(pre[b].cpu())
                    vis_post.append(post[b].cpu())
                    vis_masks.append(mask[b].cpu())
                    vis_preds.append(preds[b].cpu())

    metrics = tracker.compute()
    cm      = tracker.confusion_matrix()
    avg_loss = total_loss / len(loader)

    print("\n" + "="*55)
    print(f"  Evaluation — {split_name.upper()} split")
    print("="*55)
    print(f"  Loss:      {avg_loss:.4f}")
    print(f"  IoU:       {metrics['iou']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print("\n  Confusion Matrix  [[TN, FP], [FN, TP]]:")
    print(f"  {cm}")
    print("="*55)

    if save_vis:
        os.makedirs(out_dir, exist_ok=True)
        save_prediction_grid(
            vis_pre, vis_post, vis_masks, vis_preds,
            save_path=os.path.join(out_dir, f"{split_name}_predictions.png"),
            eo_bands=cfg["data"]["eo_bands"],
            epoch=0,
        )
        save_confusion_matrix(
            cm,
            save_path=os.path.join(out_dir, f"{split_name}_confusion_matrix.png"),
            title=f"{split_name.title()} Split Confusion Matrix",
        )
        # Save metrics JSON
        results = {"split": split_name, "threshold": threshold,
                   "loss": avg_loss, **metrics,
                   "confusion_matrix": cm.tolist()}
        jpath = os.path.join(out_dir, f"{split_name}_metrics.json")
        with open(jpath, "w") as jf:
            json.dump(results, jf, indent=2)
        print(f"  Results saved -> {out_dir}/")

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(description="GalaxEye Change Detection Evaluation")
    parser.add_argument("--config",         default="configs/config.yaml")
    parser.add_argument("--weights",        required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_path",      default=None,  help="Override dataset root")
    parser.add_argument("--split",          default="test", help="Split to evaluate: train/val/test")
    parser.add_argument("--threshold",      default=None, type=float,
                        help="Decision threshold (default: from config or tuned)")
    parser.add_argument("--tune_threshold", action="store_true",
                        help="Tune threshold on val set before evaluating test")
    parser.add_argument("--out_dir",        default="./eval_outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_path:
        cfg["data"]["root"] = args.data_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Loaded weights from {args.weights}")

    criterion  = build_loss(cfg)
    val_tf     = build_val_transforms(cfg["data"]["image_size"])

    threshold = args.threshold or cfg["evaluation"]["threshold"]

    # Optionally tune threshold on validation set
    if args.tune_threshold:
        print("[INFO] Tuning threshold on validation set...")
        val_ds = ChangeDetectionDataset(
            cfg["data"]["root"], cfg["data"]["val_split"], cfg, val_tf)
        val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=2)
        logits_list, targets_list = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                pre  = batch["pre"].to(device)
                post = batch["post"].to(device)
                mask = batch["mask"]
                with autocast(enabled=cfg.get("amp", True) and device.type == "cuda"):
                    logits = model(pre, post)
                logits_list.append(logits.cpu())
                targets_list.append(mask)
        threshold = find_best_threshold(logits_list, targets_list,
                                        cfg["evaluation"]["threshold_sweep"])

    # Evaluate on requested split
    eval_ds = ChangeDetectionDataset(
        cfg["data"]["root"], args.split, cfg, val_tf)
    eval_loader = DataLoader(eval_ds, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=2)

    run_eval(model, eval_loader, criterion, cfg, device,
             threshold=threshold, split_name=args.split,
             save_vis=True, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

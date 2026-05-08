#!/usr/bin/env python3
"""
train.py — Training script for GalaxEye EO+SAR Change Detection.

Usage:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --resume checkpoints/last.pth
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.dataset   import build_datasets
from models.model   import build_model
from models.losses  import build_loss
from utils.metrics  import MetricTracker, find_best_threshold
from utils.visualize import save_prediction_grid, save_confusion_matrix


# ── Seed everything ────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Warmup + cosine scheduler ─────────────────────────────────────────────────
def build_scheduler(optimizer, cfg):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = cfg["scheduler"]["warmup_epochs"]
    total_epochs  = cfg["training"]["epochs"]

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=total_epochs - warmup_epochs,
                               eta_min=cfg["scheduler"]["eta_min"])
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# ── One epoch train ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, cfg, device, epoch):
    model.train()
    total_loss  = 0.0
    accum_steps = cfg["training"]["accumulation_steps"]
    grad_clip   = cfg["training"]["grad_clip"]
    use_amp     = cfg.get("amp", True) and device.type == "cuda"

    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        pre    = batch["pre"].to(device)
        post   = batch["post"].to(device)
        mask   = batch["mask"].to(device)

        with autocast(enabled=use_amp):
            logits = model(pre, post)
            loss   = criterion(logits, mask) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps

        if step % cfg["logging"]["log_every_n_steps"] == 0:
            print(f"  [Train] Ep{epoch} step {step}/{len(loader)}  "
                  f"loss={loss.item()*accum_steps:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(loader)


# ── Validation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, cfg, device, threshold=0.5,
             save_vis=False, vis_dir=None, epoch=0):
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

        # Collect for visualisation
        if save_vis and len(vis_pre) < cfg["logging"]["num_vis_samples"]:
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= threshold).long()
            for b in range(pre.shape[0]):
                if len(vis_pre) < cfg["logging"]["num_vis_samples"]:
                    vis_pre.append(pre[b].cpu())
                    vis_post.append(post[b].cpu())
                    vis_masks.append(mask[b].cpu())
                    vis_preds.append(preds[b].cpu())

    metrics = tracker.compute()
    cm      = tracker.confusion_matrix()

    if save_vis and vis_dir:
        save_prediction_grid(
            vis_pre, vis_post, vis_masks, vis_preds,
            save_path=os.path.join(vis_dir, f"epoch_{epoch:03d}.png"),
            eo_bands=cfg["data"]["eo_bands"],
            epoch=epoch,
        )
        save_confusion_matrix(
            cm,
            save_path=os.path.join(vis_dir, f"cm_epoch_{epoch:03d}.png"),
            title=f"Epoch {epoch} Confusion Matrix"
        )

    return total_loss / len(loader), metrics, cm


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[CKPT] Saved -> {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_score  = ckpt.get("best_score", -1)
    if optimizer  and "optimizer"  in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler  and "scheduler"  in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[CKPT] Loaded from {path} (epoch {start_epoch-1}, best={best_score:.4f})")
    return start_epoch, best_score


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GalaxEye Change Detection Training")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, _ = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"], drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"]
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model     = build_model(cfg).to(device)
    criterion = build_loss(cfg)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model params: {n_params/1e6:.2f}M")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    opt_cfg   = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
    )
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=cfg.get("amp", True) and device.type == "cuda")

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_score  = -1.0
    resume_path = args.resume or cfg["checkpoint"].get("resume")
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_score = load_checkpoint(
            resume_path, model, optimizer, scheduler)

    ckpt_dir  = cfg["checkpoint"]["save_dir"]
    vis_dir   = os.path.join(cfg["logging"]["log_dir"], "visualisations")
    best_metric = cfg["checkpoint"]["save_best_metric"]
    log_path  = os.path.join(cfg["logging"]["log_dir"], "train_log.csv")
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)

    # Write log header
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,iou,precision,recall,f1,lr\n")

    print("\n" + "="*60)
    print("  GalaxEye — EO+SAR Binary Change Detection Training")
    print("="*60)

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{cfg['training']['epochs']} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, device, epoch)

        # Validate
        save_vis   = (epoch % cfg["logging"]["save_vis_every_n_epochs"] == 0)
        val_loss, metrics, cm = evaluate(
            model, val_loader, criterion, cfg, device,
            threshold=cfg["evaluation"]["threshold"],
            save_vis=save_vis, vis_dir=vis_dir, epoch=epoch
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"  Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")
        print(f"  IoU={metrics['iou']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
              f"lr={lr:.2e}  [{elapsed:.0f}s]")

        # Log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},"
                    f"{metrics['iou']},{metrics['precision']},"
                    f"{metrics['recall']},{metrics['f1']},{lr:.2e}\n")

        # Save best
        score = metrics[best_metric]
        if score > best_score:
            best_score = score
            save_checkpoint({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score, "metrics": metrics, "cfg": cfg,
            }, path=os.path.join(ckpt_dir, "best.pth"))
            print(f"  *** New best {best_metric.upper()}={best_score:.4f} — checkpoint saved ***")

        # Always save last
        save_checkpoint({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score, "cfg": cfg,
        }, path=os.path.join(ckpt_dir, "last.pth"))

    print("\n" + "="*60)
    print(f"  Training complete. Best val {best_metric.upper()} = {best_score:.4f}")
    print(f"  Best checkpoint: {ckpt_dir}/best.pth")
    print("="*60)


if __name__ == "__main__":
    main()

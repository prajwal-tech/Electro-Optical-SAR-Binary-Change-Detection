#!/usr/bin/env python3
"""
explore_data.py — Dataset exploration script.

Run this FIRST before training to understand your data.
Outputs:
  - Band count and resolution per split
  - Class distribution (before and after remapping)
  - Sample visualisations saved to ./data_exploration/

Usage:
    python scripts/explore_data.py --config configs/config.yaml
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import read_tif, read_mask, LABEL_REMAP


def analyse_split(split_dir: Path, cfg: dict, out_dir: str, max_files: int = 20):
    pre_dir  = split_dir / cfg["data"]["pre_event_dir"]
    post_dir = split_dir / cfg["data"]["post_event_dir"]
    tgt_dir  = split_dir / cfg["data"]["target_dir"]

    pre_files = sorted(pre_dir.glob("*.tif"))[:max_files]
    print(f"\n{'='*55}")
    print(f"  Split: {split_dir.name}   ({len(list(pre_dir.glob('*.tif')))} total files)")
    print(f"{'='*55}")

    all_change_ratios = []
    band_counts       = set()

    for pf in pre_files:
        name = pf.name
        po   = post_dir / name
        tg   = tgt_dir  / name
        if not (po.exists() and tg.exists()):
            continue

        with rasterio.open(str(pf)) as src:
            h, w  = src.height, src.width
            bands = src.count
            band_counts.add(bands)

        mask   = read_mask(str(tg))
        n_ch   = (mask == 1).sum()
        n_tot  = mask.size
        all_change_ratios.append(n_ch / n_tot)

    print(f"  Image bands:        {band_counts}")
    print(f"  Image size (last):  {h} x {w}")
    print(f"  Change pixel ratio: mean={np.mean(all_change_ratios):.3f}  "
          f"min={np.min(all_change_ratios):.3f}  max={np.max(all_change_ratios):.3f}")
    print(f"  No-change ratio:    mean={1-np.mean(all_change_ratios):.3f}")

    # Plot distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_change_ratios, bins=20, color="#e74c3c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Change pixel fraction per image", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Change pixel distribution — {split_dir.name} split", fontsize=12)
    ax.axvline(np.mean(all_change_ratios), color="black", linestyle="--",
               label=f"Mean = {np.mean(all_change_ratios):.3f}")
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{split_dir.name}_class_distribution.png"), dpi=120)
    plt.close()

    # Visualise a few samples
    sample_files = list(pre_dir.glob("*.tif"))[:4]
    fig, axes = plt.subplots(len(sample_files), 4, figsize=(16, 4*len(sample_files)))
    if len(sample_files) == 1:
        axes = axes[np.newaxis, :]

    for i, pf in enumerate(sample_files):
        name = pf.name
        po   = post_dir / name
        tg   = tgt_dir  / name
        if not (po.exists() and tg.exists()):
            continue

        pre_img  = read_tif(str(pf))
        post_img = read_tif(str(po))
        mask     = read_mask(str(tg))

        def to_rgb(arr):
            rgb = arr[:, :, :3] if arr.shape[2] >= 3 else np.repeat(arr[:,:,:1], 3, axis=2)
            for c in range(rgb.shape[2]):
                ch = rgb[:,:,c]
                p2, p98 = np.percentile(ch[ch>0], [2,98]) if (ch>0).any() else (0,1)
                rgb[:,:,c] = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
            return rgb

        axes[i,0].imshow(to_rgb(pre_img));   axes[i,0].set_title("Pre-event (EO)"); axes[i,0].axis("off")
        axes[i,1].imshow(to_rgb(post_img));  axes[i,1].set_title("Post-event (EO)"); axes[i,1].axis("off")
        axes[i,2].imshow(pre_img[:,:,min(3,pre_img.shape[2]-1)], cmap="gray")
        axes[i,2].set_title("SAR channel"); axes[i,2].axis("off")
        axes[i,3].imshow(mask, cmap="hot", vmin=0, vmax=1)
        axes[i,3].set_title(f"Change mask  ({mask.mean():.2%} change)")
        axes[i,3].axis("off")

    plt.suptitle(f"Sample images — {split_dir.name}", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{split_dir.name}_samples.png"),
                bbox_inches="tight", dpi=100)
    plt.close()
    print(f"  Visualisations saved -> {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out_dir", default="./data_exploration")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])
    for split in [cfg["data"]["train_split"],
                  cfg["data"]["val_split"],
                  cfg["data"]["test_split"]]:
        split_dir = root / split
        if split_dir.exists():
            analyse_split(split_dir, cfg, args.out_dir)
        else:
            print(f"[WARN] Split dir not found: {split_dir}")

    print("\n[DONE] Exploration complete.")


if __name__ == "__main__":
    main()

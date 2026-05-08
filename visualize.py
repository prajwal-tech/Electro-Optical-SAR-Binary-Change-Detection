"""
visualize.py — Qualitative visualisations for change detection predictions.

Produces side-by-side panels:
  Pre-event EO | Post-event EO | Ground Truth | Prediction | Overlay
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch


def tensor_to_rgb(t: torch.Tensor, eo_bands: int = 3) -> np.ndarray:
    """
    Convert image tensor (C, H, W) to displayable RGB uint8.
    Uses first 3 EO channels; applies simple per-channel normalisation.
    """
    arr = t.cpu().numpy()                      # (C, H, W)
    n   = min(eo_bands, arr.shape[0], 3)
    rgb = arr[:n].transpose(1, 2, 0)          # (H, W, n)

    # Un-do ImageNet normalisation for display
    MEAN = np.array([0.485, 0.456, 0.406])[:n]
    STD  = np.array([0.229, 0.224, 0.225])[:n]
    rgb  = rgb * STD + MEAN
    rgb  = np.clip(rgb, 0, 1)

    # Pad to 3 channels if grayscale
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 2:
        rgb = np.concatenate([rgb, rgb[:,:,:1]], axis=2)

    return (rgb * 255).astype(np.uint8)


def make_colour_mask(mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Convert binary mask (H,W) -> RGBA overlay."""
    H, W  = mask.shape
    rgba  = np.zeros((H, W, 4), dtype=np.float32)
    rgba[mask == 1] = [1.0, 0.0, 0.0, alpha]   # Red for change
    rgba[mask == 0] = [0.0, 1.0, 0.0, alpha]   # Green for no-change
    return rgba


def save_prediction_grid(
    pre_imgs:   list,
    post_imgs:  list,
    masks:      list,
    preds:      list,
    save_path:  str,
    eo_bands:   int = 3,
    epoch:      int = 0,
):
    """
    Save a grid of N prediction examples (up to 8).
    Each row: Pre EO | Post EO | Ground Truth | Prediction | Error Map
    """
    N = min(len(pre_imgs), 8)
    fig, axes = plt.subplots(N, 5, figsize=(20, 4 * N))
    if N == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Pre-Event (EO)", "Post-Event (EO)", "Ground Truth", "Prediction", "Error Map"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    for i in range(N):
        pre_rgb  = tensor_to_rgb(pre_imgs[i],  eo_bands)
        post_rgb = tensor_to_rgb(post_imgs[i], eo_bands)
        mask     = masks[i].cpu().numpy()  if torch.is_tensor(masks[i])  else masks[i]
        pred     = preds[i].cpu().numpy()  if torch.is_tensor(preds[i])  else preds[i]

        # Error map: TP=white, FP=red, FN=blue, TN=black
        error           = np.zeros((*mask.shape, 3), dtype=np.uint8)
        error[(mask==1) & (pred==1)] = [255, 255, 255]  # TP - white
        error[(mask==0) & (pred==1)] = [255,   0,   0]  # FP - red
        error[(mask==1) & (pred==0)] = [  0,   0, 255]  # FN - blue
        # TN stays black

        axes[i, 0].imshow(pre_rgb);          axes[i, 0].axis("off")
        axes[i, 1].imshow(post_rgb);         axes[i, 1].axis("off")
        axes[i, 2].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[i, 2].axis("off")
        axes[i, 3].imshow(pred, cmap="gray", vmin=0, vmax=1); axes[i, 3].axis("off")
        axes[i, 4].imshow(error);            axes[i, 4].axis("off")

    # Legend for error map
    legend_patches = [
        mpatches.Patch(color="white",  label="TP"),
        mpatches.Patch(color="red",    label="FP"),
        mpatches.Patch(color="blue",   label="FN"),
        mpatches.Patch(color="black",  label="TN"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(f"Epoch {epoch} — Predictions", fontsize=13, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[VIS] Saved -> {save_path}")


def save_confusion_matrix(cm: np.ndarray, save_path: str, title: str = "Confusion Matrix"):
    """Save a 2x2 confusion matrix as an image."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ticks  = [0, 1]
    labels = ["No-Change", "Change"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[VIS] Saved CM -> {save_path}")

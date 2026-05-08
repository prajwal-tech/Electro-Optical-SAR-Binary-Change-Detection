"""
dataset.py — EO+SAR Change Detection Dataset

Directory structure (same for train / val / test):
    {root}/{split}/pre-event/   *.tif   pre-disaster images
    {root}/{split}/post-event/  *.tif   post-disaster images (same filenames)
    {root}/{split}/target/      *.tif   label masks (same filenames)

Label remapping (mandatory):
    0 Background -> 0  No-Change
    1 Intact     -> 0  No-Change
    2 Damaged    -> 1  Change
    3 Destroyed  -> 1  Change
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import albumentations as A
from pathlib import Path

LABEL_REMAP = {0: 0, 1: 0, 2: 1, 3: 1}

def remap_label(mask):
    out = np.zeros_like(mask, dtype=np.uint8)
    for orig, rm in LABEL_REMAP.items():
        out[mask == orig] = rm
    return out

def read_tif(path):
    """Read GeoTIFF -> (H, W, C) float32."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    return np.transpose(data, (1, 2, 0))

def read_mask(path):
    """Read label mask, apply remap -> (H, W) uint8."""
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.int32)
    return remap_label(mask)

def normalize_eo_sar(img):
    """
    Normalize all channels: percentile clip -> [0, 1].
    img shape: (H, W, C)
    """
    C   = img.shape[2]
    out = img.copy()

    for i in range(C):
        ch    = out[:, :, i]
        valid = ch[ch > 0]
        p2, p98 = (np.percentile(valid, [2, 98]) if len(valid) > 0 else (0.0, 1.0))
        ch = np.clip(ch, p2, p98)
        ch = (ch - p2) / (p98 - p2 + 1e-8)
        out[:, :, i] = ch

    return out.astype(np.float32)


def to_3channel(img):
    """
    Ensure image has exactly 3 channels.
    If 1 channel -> repeat 3 times.
    If 2 channels -> add a third by averaging.
    If 3+ channels -> take first 3.
    img shape: (H, W, C)
    """
    C = img.shape[2]
    if C == 1:
        return np.repeat(img, 3, axis=2)
    elif C == 2:
        third = ((img[:, :, 0:1] + img[:, :, 1:2]) / 2.0)
        return np.concatenate([img, third], axis=2)
    else:
        return img[:, :, :3]


def build_train_transforms(image_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.3),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=0.3),
        A.PadIfNeeded(image_size, image_size, border_mode=0),
        A.RandomCrop(image_size, image_size),
    ], additional_targets={"image2": "image"})


def build_val_transforms(image_size):
    return A.Compose([
        A.PadIfNeeded(image_size, image_size, border_mode=0),
        A.CenterCrop(image_size, image_size),
    ], additional_targets={"image2": "image"})


class ChangeDetectionDataset(Dataset):
    """Paired pre/post EO+SAR images with binary change masks."""

    def __init__(self, root, split, cfg, transforms=None):
        self.transforms = transforms

        split_dir = Path(root) / split
        pre_dir   = split_dir / cfg["data"]["pre_event_dir"]
        post_dir  = split_dir / cfg["data"]["post_event_dir"]
        tgt_dir   = split_dir / cfg["data"]["target_dir"]

        pre_files = sorted(pre_dir.glob("*.tif"))
        if not pre_files:
            raise FileNotFoundError(f"No .tif files in {pre_dir}. "
                                    "Check data root and split folder names.")

        self.samples = []
        missing = 0
        for pf in pre_files:
            name = pf.name
            po   = post_dir / name
            tg   = tgt_dir  / name
            if po.exists() and tg.exists():
                self.samples.append((str(pf), str(po), str(tg)))
            else:
                missing += 1

        if missing:
            print(f"[WARN:{split}] {missing} files skipped (no matching pair).")
        if not self.samples:
            raise RuntimeError(f"No valid triplets in {split_dir}.")
        print(f"[Dataset:{split}] {len(self.samples)} samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pre_p, post_p, tgt_p = self.samples[idx]

        pre_img  = to_3channel(normalize_eo_sar(read_tif(pre_p)))
        post_img = to_3channel(normalize_eo_sar(read_tif(post_p)))
        mask     = read_mask(tgt_p)

        H = min(pre_img.shape[0], post_img.shape[0])
        W = min(pre_img.shape[1], post_img.shape[1])
        pre_img  = pre_img[:H, :W, :]
        post_img = post_img[:H, :W, :]
        mask     = mask[:H, :W]

        if self.transforms:
            aug      = self.transforms(image=pre_img, image2=post_img, mask=mask)
            pre_img  = aug["image"]
            post_img = aug["image2"]
            mask     = aug["mask"]

        pre_t  = torch.from_numpy(pre_img.transpose(2, 0, 1))
        post_t = torch.from_numpy(post_img.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask.copy()).long()

        return {"pre": pre_t, "post": post_t, "mask": mask_t,
                "name": os.path.basename(pre_p)}


def build_datasets(cfg):
    sz   = cfg["data"]["image_size"]
    root = cfg["data"]["root"]
    train_ds = ChangeDetectionDataset(root, cfg["data"]["train_split"], cfg, build_train_transforms(sz))
    val_ds   = ChangeDetectionDataset(root, cfg["data"]["val_split"],   cfg, build_val_transforms(sz))
    test_ds  = ChangeDetectionDataset(root, cfg["data"]["test_split"],  cfg, build_val_transforms(sz))
    return train_ds, val_ds, test_ds
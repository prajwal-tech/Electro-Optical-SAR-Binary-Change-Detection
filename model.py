"""
model.py — Siamese UNet with EfficientNet-B2 backbone for binary change detection.

Architecture:
  - Shared EfficientNet-B2 encoder processes pre and post images independently
  - Feature difference at each scale is passed to the decoder
  - UNet-style decoder with skip connections produces a binary change map

Key design choices:
  - Weight-shared encoder enforces consistent feature space for both timesteps
  - Feature difference (|f_post - f_pre|) is modality-agnostic and captures change
  - EfficientNet-B2 chosen for strong ImageNet features + efficiency
  - Pretrained weights used for EO channels; SAR channels initialised randomly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


# ── Decoder blocks ─────────────────────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Upsample + skip connection + two conv layers."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ── Siamese UNet ───────────────────────────────────────────────────────────────
class SiamUNet(nn.Module):
    """
    Siamese UNet for binary change detection.

    Input:  pre  (B, C, H, W)  and  post  (B, C, H, W)
    Output: logits (B, 1, H, W) — raw before sigmoid
    """

    def __init__(self, in_channels=5, pretrained=True, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        # ── Shared encoder ──────────────────────────────────────────────────
        # Load EfficientNet-B2 with 3-channel pretrained weights
        encoder = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Adapt first conv if in_channels != 3
        if in_channels != 3:
            old_conv = encoder.conv_stem
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            # Copy ImageNet weights for first 3 channels; init rest randomly
            with torch.no_grad():
                new_conv.weight[:, :min(3, in_channels)] =                     old_conv.weight[:, :min(3, in_channels)]
                if in_channels > 3:
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:])
            encoder.conv_stem = new_conv

        self.encoder = encoder

        # Get encoder output channel sizes by a dummy forward pass
        with torch.no_grad():
            dummy   = torch.zeros(1, in_channels, 256, 256)
            feats   = self.encoder(dummy)
            enc_chs = [f.shape[1] for f in feats]  # e.g. [24, 48, 120, 208, 352]

        # ── Decoder ─────────────────────────────────────────────────────────
        # After difference: each skip has enc_chs[i] channels
        dec_ch = [256, 128, 64, 32, 16]
        self.decoder = nn.ModuleList()

        # bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_chs[-1], dec_ch[0]),
            ConvBnRelu(dec_ch[0], dec_ch[0]),
        )

        # decoder blocks: input = previous decoder output + skip from encoder diff
        for i in range(4):
            in_ch   = dec_ch[i]
            skip_ch = enc_chs[-(i+2)]
            out_ch  = dec_ch[i+1]
            self.decoder.append(DecoderBlock(in_ch, skip_ch, out_ch))

        self.dropout  = nn.Dropout2d(p=dropout)
        self.head     = nn.Conv2d(dec_ch[-1], 1, kernel_size=1)

        # Final upsample to input resolution if needed
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def encode(self, x):
        """Extract multi-scale features."""
        return self.encoder(x)

    def forward(self, pre, post):
        """
        pre, post: (B, C, H, W)
        Returns logits: (B, 1, H, W) at input resolution
        """
        B, C, H, W = pre.shape

        f_pre  = self.encode(pre)
        f_post = self.encode(post)

        # Compute absolute feature difference at each scale
        diff = [torch.abs(fp - fpr) for fp, fpr in zip(f_post, f_pre)]

        # Bottleneck on deepest difference map
        x = self.bottleneck(diff[-1])

        # Decode with skip connections
        for i, dec_block in enumerate(self.decoder):
            skip = diff[-(i+2)]
            x = dec_block(x, skip)

        x = self.dropout(x)
        x = self.head(x)                         # (B, 1, H//2, W//2) approx
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x


def build_model(cfg):
    in_ch = cfg["data"]["eo_bands"] + cfg["data"]["sar_bands"]
    print(f"[Model] Building SiamUNet with in_channels={in_ch}")
    model = SiamUNet(
        in_channels=in_ch,
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    )
    return model

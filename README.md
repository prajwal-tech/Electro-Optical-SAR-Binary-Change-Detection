# Electro-Optical-SAR-Binary-Change-Detection
Pixel-level binary change detection on paired Electro-Optical and Synthetic Aperture Radar satellite imagery using Siamese UNet with EfficientNet-B2 backbone for disaster response and urban monitoring.


Problem Statement
Given a co-registered pre-event and post-event satellite image pair, the model produces a binary pixel-level change mask:

1 = Change (Damaged or Destroyed)
0 = No-Change (Background or Intact)

The dataset contains extreme class imbalance (~97% no-change pixels), which is addressed through a combined Focal + Dice loss strategy.

Architecture
pre-event image  ──→ ┐
                     ├──  Shared EfficientNet-B2 Encoder
post-event image ──→ ┘
                          ↓
                     |f_post - f_pre|  (absolute feature difference at each scale)
                          ↓
                     UNet Decoder with skip connections
                          ↓
                     Binary Change Mask (H × W × 1)
Key design decisions:

Weight-shared Siamese encoder — forces consistent feature space for both timesteps
Absolute feature difference — modality-agnostic, directly captures change signal at multiple scales
UNet-style decoder — recovers spatial resolution with skip connections
Focal + Dice loss — Focal loss handles hard examples, Dice loss directly optimises IoU-like metric
Threshold tuning — default 0.5 is suboptimal for heavily imbalanced data; tuned on validation set


Requirements

Python 3.10+
PyTorch 2.1.0
CPU or CUDA GPU (GPU strongly recommended)


Environment Setup
bash# Create virtual environment
python -m venv galaxeye_env

# Activate (Windows)
galaxeye_env\Scripts\activate

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install rasterio albumentations==1.3.1 timm==0.9.12 pyyaml matplotlib numpy tqdm Pillow opencv-python-headless

Dataset Structure
dataset/
├── train/
│   ├── pre-event/        *.tif   (pre-disaster images)
│   ├── post-event/       *.tif   (post-disaster images, same filenames)
│   └── target/           *.tif   (label masks)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
Update configs/config.yaml:
yamldata:
  root: "/path/to/your/dataset"
Label remapping applied automatically:
Original ClassOriginal ValueRemapped ValueRemapped ClassBackground00No-ChangeIntact10No-ChangeDamaged21ChangeDestroyed31Change

Step 0 — Explore Data First
bashpython scripts/explore_data.py --config configs/config.yaml
Outputs class distribution plots and sample visualisations to ./data_exploration/.

Training
python train.py --config configs/config.yaml

Resume from checkpoint:

python train.py --config configs/config.yaml --resume checkpoints/last.pth
Key hyperparameters:

ParameterValueRationaleBackboneEfficientNet-B2Strong pretrained features, computationally efficientLossFocal (α=0.85) + DiceHandles ~97% no-change class imbalanceOptimizerAdamW lr=3e-4Stable convergence with weight decaySchedulerCosineAnnealingLR + 3ep warmupSmooth LR decay, avoids early instabilitypos_weight20.0Compensates for extreme no-change dominanceimage_size256Memory efficient patch-based training

Evaluation
python eval.py --config configs/config.yaml --weights checkpoints/best.pth --tune_threshold
Outputs saved to ./eval_outputs/:

test_predictions.png — qualitative grid (pre | post | ground truth | prediction | error map)
test_confusion_matrix.png
test_metrics.json




Repository Structure
├── configs/
│   └── config.yaml           # All hyperparameters
├── data/
│   └── dataset.py            # Dataset, normalisation, augmentation
├── models/
│   ├── model.py              # SiamUNet architecture
│   └── losses.py             # Focal + Dice combined loss
├── utils/
│   ├── metrics.py            # IoU, Precision, Recall, F1, confusion matrix
│   └── visualize.py          # Prediction grid and confusion matrix plots
├── scripts/
│   └── explore_data.py       # Dataset exploration script
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
└── requirements.txt

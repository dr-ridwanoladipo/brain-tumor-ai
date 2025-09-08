#!/usr/bin/env python3
"""
🧠 Clinical-Grade Brain Tumor Segmentation Training - SageMaker Script
by Ridwan Oladipo, MD | AI Specialist

Professional nnU-Net 2025 training pipeline for SageMaker Training Jobs
Targeting WT Dice ≥ 90% and BraTS Avg ≥ 80%
"""

import os
import sys
import argparse
import json
import random
import time
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import nibabel as nib
from tqdm import tqdm
import warnings
from scipy.ndimage import rotate, zoom
import boto3
from datetime import datetime
import shutil

warnings.filterwarnings('ignore')


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)


class BrainTumorDataset(Dataset):
    def __init__(self, npz_files, patch_size, patches_per_epoch, max_volumes=None):
        self.patch_size = patch_size
        self.pH, self.pW, self.pD = patch_size
        self.patches_per_epoch = patches_per_epoch

        # Index all cases at startup
        if max_volumes:
            npz_files = npz_files[:max_volumes]

        self.cases = []
        for npz_file in tqdm(npz_files, desc="Indexing cases"):
            # Memory-mapped loading to avoid full RAM usage
            with np.load(npz_file, mmap_mode='r') as vol:
                # Extract metadata without loading full arrays
                shape = vol['image'].shape[:3]

                # Compute indices and store only the indices
                tumor_indices = np.array(np.where(vol['label'] > 0)).T
                enhancing_indices = np.array(np.where(vol['label'] == 3)).T
                brain_indices = np.array(np.where(vol['brain_mask_full'] > 0)).T

            self.cases.append({
                'file': npz_file,
                'shape': shape,
                'tumor_indices': tumor_indices,
                'enhancing_indices': enhancing_indices,
                'brain_indices': brain_indices
            })

        print(f"Memory-efficient indexing complete: {len(self.cases)} cases")

    def __len__(self):
        return self.patches_per_epoch * 2

    def __getitem__(self, idx):
        # Select random case
        case = random.choice(self.cases)

        # Lazy loading with proper file handle cleanup
        with np.load(case['file']) as vol:
            image_data = vol['image'].astype(np.float32)
            label_data = vol['label'].astype(np.int64)

        H, W, D = case['shape']

        # Smart sampling with oversampling
        sampling_prob = random.random()
        if len(case['enhancing_indices']) > 0 and sampling_prob < 0.4:
            center = case['enhancing_indices'][random.randint(0, len(case['enhancing_indices']) - 1)]
        elif len(case['tumor_indices']) > 0 and sampling_prob < 0.7:
            center = case['tumor_indices'][random.randint(0, len(case['tumor_indices']) - 1)]
        else:
            center = case['brain_indices'][random.randint(0, len(case['brain_indices']) - 1)]

        # Extract patch with bounds checking
        h = np.clip(center[0] - self.pH // 2, 0, H - self.pH)
        w = np.clip(center[1] - self.pW // 2, 0, W - self.pW)
        d = np.clip(center[2] - self.pD // 2, 0, D - self.pD)

        # Extract patch from volume
        img_patch = image_data[h:h + self.pH, w:w + self.pW, d:d + self.pD, :].copy()
        lbl_patch = label_data[h:h + self.pH, w:w + self.pW, d:d + self.pD].copy()

        # Transpose to [C,H,W,D]
        img_patch = np.transpose(img_patch, (3, 0, 1, 2))

        # Augmentations
        img_patch, lbl_patch = self.augment(img_patch, lbl_patch)

        return torch.from_numpy(img_patch), torch.from_numpy(lbl_patch)

    def augment(self, img_patch, lbl_patch):
        # Spatial augmentations
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            # For image (4D: C,H,W,D)
            axes_img = random.choice([(1, 2), (1, 3), (2, 3)])
            img_patch = rotate(img_patch, angle, axes=axes_img, reshape=False, order=1).copy()
            # For label (3D: H,W,D)
            axes_lbl = random.choice([(0, 1), (0, 2), (1, 2)])
            lbl_patch = rotate(lbl_patch, angle, axes=axes_lbl, reshape=False, order=0).copy()

        if random.random() < 0.2:
            scale = random.uniform(0.85, 1.15)
            img_patch = zoom(img_patch, (1, scale, scale, scale), order=1).copy()
            lbl_patch = zoom(lbl_patch, (scale, scale, scale), order=0).copy()

            # Crop/pad to original size
            ph, pw, pd = img_patch.shape[1:]
            if ph != self.pH or pw != self.pW or pd != self.pD:
                img_patch = img_patch[:, :self.pH, :self.pW, :self.pD] if ph >= self.pH else np.pad(img_patch, (
                    (0, 0), (0, max(0, self.pH - ph)), (0, max(0, self.pW - pw)), (0, max(0, self.pD - pd))))
                lbl_patch = lbl_patch[:self.pH, :self.pW, :self.pD] if ph >= self.pH else np.pad(lbl_patch, (
                    (0, max(0, self.pH - ph)), (0, max(0, self.pW - pw)), (0, max(0, self.pD - pd))))

        # Intensity augmentations
        if random.random() < 0.15:
            img_patch *= random.uniform(0.85, 1.25)
        if random.random() < 0.15:
            gamma = random.uniform(0.7, 1.5)
            img_patch = np.sign(img_patch) * np.abs(img_patch) ** gamma
        if random.random() < 0.15:
            for c in range(img_patch.shape[0]):
                if np.any(img_patch[c] != 0):
                    noise_std = 0.1 * np.std(img_patch[c][img_patch[c] != 0])
                    img_patch[c] += np.random.normal(0, noise_std, img_patch[c].shape).astype(np.float32)

        return img_patch, lbl_patch


def conv_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_f, out_f, 3, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )


def down_block(in_f, out_f):
    return nn.Sequential(
        nn.Conv3d(in_f, out_f, 3, stride=2, padding=1, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )


def up_block(in_f, out_f):
    return nn.Sequential(
        nn.ConvTranspose3d(in_f, out_f, 2, stride=2, bias=False),
        nn.InstanceNorm3d(out_f),
        nn.LeakyReLU(0.01, inplace=True)
    )


class nnUNet2025(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters):
        super().__init__()
        # 5-level encoder
        self.enc1 = conv_block(in_channels, base_filters)
        self.down1 = down_block(base_filters, base_filters * 2)
        self.enc2 = conv_block(base_filters * 2, base_filters * 2)
        self.down2 = down_block(base_filters * 2, base_filters * 4)
        self.enc3 = conv_block(base_filters * 4, base_filters * 4)
        self.down3 = down_block(base_filters * 4, base_filters * 8)
        self.enc4 = conv_block(base_filters * 8, base_filters * 8)
        self.down4 = down_block(base_filters * 8, base_filters * 16)

        # Bottleneck
        self.bottleneck = conv_block(base_filters * 16, base_filters * 16)

        # 5-level decoder
        self.up4 = up_block(base_filters * 16, base_filters * 8)
        self.dec4 = conv_block(base_filters * 16, base_filters * 8)
        self.up3 = up_block(base_filters * 8, base_filters * 4)
        self.dec3 = conv_block(base_filters * 8, base_filters * 4)
        self.up2 = up_block(base_filters * 4, base_filters * 2)
        self.dec2 = conv_block(base_filters * 4, base_filters * 2)
        self.up1 = up_block(base_filters * 2, base_filters)
        self.dec1 = conv_block(base_filters * 2, base_filters)

        # Output layers
        self.out_conv = nn.Conv3d(base_filters, out_channels, 1)
        self.ds_out2 = nn.Conv3d(base_filters * 2, out_channels, 1)
        self.ds_out3 = nn.Conv3d(base_filters * 4, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck
        b = self.bottleneck(self.down4(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Outputs
        main_out = self.out_conv(d1)
        ds2 = self.ds_out2(d2)
        ds3 = self.ds_out3(d3)

        return main_out, ds2, ds3


def calculate_wt_dice(model, val_loader, device, amp_enabled, amp_dtype):
    """Calculate validation WT Dice score"""
    if val_loader is None:
        return 0.0

    model.eval()
    dice_scores = []

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            # Forward pass
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(imgs)
                main_out = outputs[0] if isinstance(outputs, tuple) else outputs

            # Convert to predictions
            preds = torch.argmax(main_out, dim=1)

            # Calculate WT Dice (labels > 0)
            wt_pred = (preds > 0).float()
            wt_true = (lbls > 0).float()

            # Dice calculation per batch
            for i in range(wt_pred.shape[0]):
                pred_i = wt_pred[i].flatten()
                true_i = wt_true[i].flatten()

                intersection = (pred_i * true_i).sum()
                union = pred_i.sum() + true_i.sum()

                if union > 0:
                    dice = (2.0 * intersection) / union
                    dice_scores.append(dice.item())

        model.train()
        return np.mean(dice_scores) if dice_scores else 0.0

    def region_weighted_dice_loss(logits, targets, class_weights, out_channels, eps=1e-6):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=out_channels).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = (probs ** 2).sum(dims) + (targets_one_hot ** 2).sum(dims)
        dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)

        weighted_dice = (dice_per_class * class_weights).sum() / class_weights.sum()
        return 1.0 - weighted_dice

    def deep_supervision_loss(outputs, targets, class_weights, ce_weights, dice_weight, ce_weight, out_channels):
        main_out, ds2, ds3 = outputs

        # Main loss
        dice_loss = region_weighted_dice_loss(main_out, targets, class_weights, out_channels)
        ce_loss = F.cross_entropy(main_out, targets, weight=ce_weights)
        main_loss = dice_weight * dice_loss + ce_weight * ce_loss

        # Deep supervision losses
        targets_ds2 = F.interpolate(targets.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).long()
        targets_ds3 = F.interpolate(targets.float().unsqueeze(1), scale_factor=0.25, mode='nearest').squeeze(1).long()

        ds2_loss = F.cross_entropy(ds2, targets_ds2, weight=ce_weights) + region_weighted_dice_loss(ds2, targets_ds2,
                                                                                                    class_weights,
                                                                                                    out_channels)
        ds3_loss = F.cross_entropy(ds3, targets_ds3, weight=ce_weights) + region_weighted_dice_loss(ds3, targets_ds3,
                                                                                                    class_weights,
                                                                                                    out_channels)

        return main_loss + 0.5 * ds2_loss + 0.25 * ds3_loss

    def main():
        # Parse SageMaker arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
        parser.add_argument('--output-dir', type=str, default='/opt/ml/output')
        parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
        parser.add_argument('--checkpoint-dir', type=str, default='/opt/ml/checkpoints')
        parser.add_argument('--use-full-dataset', action='store_true')
        parser.add_argument('--no-use-full-dataset', dest='use_full_dataset', action='store_false')
        parser.set_defaults(use_full_dataset=True)
        parser.add_argument('--max-volumes', type=int, default=None)
        parser.add_argument('--random-seed', type=int, default=42)
        args = parser.parse_args()

        # Set paths for SageMaker
        data_dir = Path(args.data_dir)
        base_dir = Path(args.output_dir)
        model_dir = Path(args.model_dir)
        checkpoint_dir = Path(args.checkpoint_dir)

        # Dataset configuration
        USE_FULL_DATASET = args.use_full_dataset
        MAX_VOLUMES = args.max_volumes
        RANDOM_SEED = args.random_seed

        print(f"Brain Tumor Training Pipeline - SageMaker")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {base_dir}")
        print(f"Model directory: {model_dir}")
        print(f"Using {'FULL' if USE_FULL_DATASET else MAX_VOLUMES} volumes")
        print(f"Target: WT Dice ≥ 90, BraTS Avg ≥ 80")

        # Create directories
        output_dir = base_dir / 'outputs'
        for directory in [base_dir, output_dir, model_dir, checkpoint_dir]:
            directory.mkdir(exist_ok=True, parents=True)

        # Reproducibility and GPU optimization
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")

        # AMP settings
        amp_enabled = torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        print(f"Using {device} - {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        print(f"AMP: {amp_dtype}")

        # Configuration
        patch_size = (96, 96, 96)
        batch_size = 2
        in_channels = 4
        out_channels = 4
        base_filters = 32

        # Training parameters
        if USE_FULL_DATASET:
            epochs = 1000
            patches_per_epoch = 250
            save_every = 1
            lr = 3e-4
        else:
            epochs = 100
            patches_per_epoch = 50
            save_every = 1
            lr = 1e-3

        weight_decay = 3e-5
        dice_weight = 0.5
        ce_weight = 0.5

        # Class weights
        class_weights = torch.tensor([0.0, 1.0, 1.5, 2.0], device=device)
        ce_weights = torch.tensor([0.1, 1.0, 1.5, 2.0], device=device)

        print(f"Config - Mode: {'FULL' if USE_FULL_DATASET else 'TEST'}, Patch: {patch_size}")

        # Load preprocessed data
        npz_files = list(data_dir.glob('*_preprocessed.npz'))
        if len(npz_files) == 0:
            raise RuntimeError("No preprocessed .npz files found in data_dir")

        # Dataset split
        if USE_FULL_DATASET:
            # Reproducible splitting
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            random.shuffle(npz_files)

            # 80/10/10 split for medical AI standard
            n = len(npz_files)
            train_end = int(0.8 * n)
            val_end = train_end + int(0.1 * n)

            train_files = npz_files[:train_end]
            val_files = npz_files[train_end:val_end]
            test_files = npz_files[val_end:]

            print(f"Dataset split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

            # Save dataset split
            with open(model_dir / "dataset_split.json", "w") as f:
                json.dump({
                    "train": [str(p) for p in train_files],
                    "val": [str(p) for p in val_files],
                    "test": [str(p) for p in test_files]
                }, f, indent=2)
            print("Dataset split saved")
        else:
            # Single volume testing
            train_files = npz_files[:MAX_VOLUMES]
            val_files = None
            test_files = None
            print(f"Test mode - Using {len(train_files)} volumes")

        # Create datasets and loaders
        train_dataset = BrainTumorDataset(train_files, patch_size, patches_per_epoch)

        # Worker configuration
        cpu_cnt = os.cpu_count() or 2
        NUM_WORKERS = 2 if cpu_cnt <= 4 else min(8, cpu_cnt // 2)
        PREFETCH = 2

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
            pin_memory=True, prefetch_factor=PREFETCH, persistent_workers=(NUM_WORKERS > 0),
            worker_init_fn=worker_init_fn, drop_last=True
        )

        # Validation loader for full dataset only
        val_loader = None
        if USE_FULL_DATASET and val_files:
            val_dataset = BrainTumorDataset(val_files, patch_size, patches_per_epoch)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS,
                pin_memory=True, prefetch_factor=PREFETCH, persistent_workers=(NUM_WORKERS > 0),
                worker_init_fn=worker_init_fn, drop_last=False
            )
            print(f"Validation loader ready - {len(val_dataset)} patches")

        print(f"Training loader ready - Workers: {NUM_WORKERS}, Prefetch: {PREFETCH}")
        print(f"Train: {len(train_dataset)} patches, {len(train_loader)} batches per epoch")

    if __name__ == "__main__":
        main()
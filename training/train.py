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


if __name__ == "__main__":
    main()
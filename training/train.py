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
#!/usr/bin/env python3
"""
🧠 Clinical-Grade Multi-Site Brain Tumor Segmentation with AI - SageMaker Training Script
by Ridwan Oladipo, MD | AI Specialist

Professional nnU-Net v2 preprocessing pipeline for SageMaker Training Jobs
Targeting WT Dice ≥ 90% and BraTS Avg ≥ 80%
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import nibabel as nib
from tqdm import tqdm
import warnings
import SimpleITK as sitk
import boto3
from datetime import datetime
from scipy import ndimage
from sklearn.model_selection import KFold
import shutil
import gc

warnings.filterwarnings('ignore')


def main():
    # Parse SageMaker arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
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

    # Dataset configuration
    USE_FULL_DATASET = args.use_full_dataset
    MAX_VOLUMES = args.max_volumes
    RANDOM_SEED = args.random_seed

    print(f"🧠 SageMaker Brain Tumor Preprocessing Pipeline")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {base_dir}")
    print(f"Using {'FULL' if USE_FULL_DATASET else MAX_VOLUMES} volumes")
    print(f"🎯 Target: WT Dice ≥ 90, BraTS Avg ≥ 80")

    # Create directories
    processed_dir = base_dir / 'processed'
    nnunet_dir = base_dir / 'nnUNet_raw' / 'Dataset001_BrainTumor'
    results_dir = base_dir / 'results'

    for directory in [base_dir, processed_dir, nnunet_dir, nnunet_dir / 'imagesTr',
                      nnunet_dir / 'labelsTr', nnunet_dir / 'imagesTs',
                      results_dir, model_dir]:
        directory.mkdir(exist_ok=True, parents=True)

    # Load dataset metadata
    print("📊 Loading dataset metadata...")

    with open(data_dir / 'dataset.json', 'r') as f:
        dataset_metadata = json.load(f)

    print(f"📊 DATASET INFORMATION:")
    print(f"Name: {dataset_metadata['name']}")
    print(f"Description: {dataset_metadata['description']}")
    print(f"Training cases: {dataset_metadata['numTraining']}")
    print(f"Modalities: {', '.join([f'{k}: {v}' for k, v in dataset_metadata['modality'].items()])}")
    print(f"Labels: {', '.join([f'{k}: {v}' for k, v in dataset_metadata['labels'].items()])}")

    # Collect files - filter hidden files
    train_image_files = sorted([f for f in (data_dir / 'imagesTr').glob('*.nii*') if not f.name.startswith('._')])
    train_label_files = sorted([f for f in (data_dir / 'labelsTr').glob('*.nii*') if not f.name.startswith('._')])
    test_image_files = sorted([f for f in (data_dir / 'imagesTs').glob('*.nii*') if not f.name.startswith('._')])

    print(f"Found {len(train_image_files)} training images")
    print(f"Found {len(train_label_files)} training labels")
    print(f"Found {len(test_image_files)} test images")

    # Validate dataset
    if len(train_image_files) == 0:
        raise RuntimeError("No training images found in data_dir. Check data path and file formats.")

    # Apply volume limitation if specified
    if not USE_FULL_DATASET and MAX_VOLUMES:
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(train_image_files), min(MAX_VOLUMES, len(train_image_files)), replace=False)
        train_image_files = [train_image_files[i] for i in sorted(indices)]
        train_label_files = [train_label_files[i] for i in sorted(indices)]
        print(f"🎯 Limited to {len(train_image_files)} volumes for testing")
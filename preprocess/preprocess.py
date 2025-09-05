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

    # nnU-Net Dataset Fingerprinting & Target Spacing Calibration
    print("🔬 nnU-Net DATASET FINGERPRINTING:")

    dataset_properties = {
        'spacings': [],
        'shapes': [],
        'modalities': ['FLAIR', 'T1w', 'T1Gd', 'T2w'],
        'intensity_properties': {mod: {'percentiles': [], 'mean': [], 'std': []} for mod in
                                 ['FLAIR', 'T1w', 'T1Gd', 'T2w']},
        'label_properties': {'labels': [], 'volumes': []}
    }

    # Extract comprehensive dataset fingerprint
    for idx in tqdm(range(len(train_image_files)), desc="Fingerprinting dataset"):
        img_path = train_image_files[idx]
        label_path = train_label_files[idx]

        # Load volumes
        img_nifti = nib.load(img_path)
        label_nifti = nib.load(label_path)

        img_data = img_nifti.get_fdata()
        label_data = label_nifti.get_fdata()
        spacing = img_nifti.header.get_zooms()[:3]

        # Collect spacing and shape information
        dataset_properties['spacings'].append(spacing)
        dataset_properties['shapes'].append(img_data.shape[:3])

        # Analyze intensity properties per modality
        for mod_idx, modality in enumerate(dataset_properties['modalities']):
            mod_data = img_data[..., mod_idx]

            # Create brain mask using Otsu thresholding
            sitk_img = sitk.GetImageFromArray(mod_data.astype(np.float32))
            otsu_filter = sitk.OtsuThresholdImageFilter()
            otsu_filter.SetInsideValue(0)
            otsu_filter.SetOutsideValue(1)
            brain_mask_sitk = otsu_filter.Execute(sitk_img)
            brain_mask = sitk.GetArrayFromImage(brain_mask_sitk) > 0

            brain_voxels = mod_data[brain_mask]

            if len(brain_voxels) > 1000:
                # nnU-Net style percentile analysis
                percentiles = np.percentile(brain_voxels, [0.5, 10, 50, 90, 99.5])
                dataset_properties['intensity_properties'][modality]['percentiles'].append(percentiles)
                dataset_properties['intensity_properties'][modality]['mean'].append(np.mean(brain_voxels))
                dataset_properties['intensity_properties'][modality]['std'].append(np.std(brain_voxels))

        # Analyze label properties
        unique_labels = np.unique(label_data)
        dataset_properties['label_properties']['labels'].append(unique_labels)

        # Calculate label volumes
        voxel_volume = np.prod(spacing)
        label_volumes = {}
        for label_val in unique_labels:
            if label_val > 0:  # Skip background
                volume = np.sum(label_data == label_val) * voxel_volume / 1000  # Convert to cm³
                label_volumes[int(label_val)] = volume
        dataset_properties['label_properties']['volumes'].append(label_volumes)

    # Calculate nnU-Net target properties
    spacings_array = np.array(dataset_properties['spacings'])
    median_spacing = np.median(spacings_array, axis=0)
    target_spacing = [float(x) for x in median_spacing]

    # Calculate intensity normalization parameters per modality
    normalization_params = {}
    for modality in dataset_properties['modalities']:
        all_percentiles = np.array(dataset_properties['intensity_properties'][modality]['percentiles'])
        all_means = np.array(dataset_properties['intensity_properties'][modality]['mean'])
        all_stds = np.array(dataset_properties['intensity_properties'][modality]['std'])

        if len(all_percentiles) > 0:
            # nnU-Net normalization scheme
            normalization_params[modality] = {
                'clip_lower': float(np.median(all_percentiles[:, 0])),  # 0.5th percentile
                'clip_upper': float(np.median(all_percentiles[:, 4])),  # 99.5th percentile
                'mean_intensity': float(np.median(all_means)),
                'std_intensity': float(np.median(all_stds))
            }

    print(f"✅ Dataset fingerprinting complete")
    print(f"📏 Target spacing: {target_spacing} mm")
    print(f"🎯 Modalities analyzed: {len(dataset_properties['modalities'])}")

    # Save dataset fingerprint
    fingerprint = {
        'target_spacing': target_spacing,
        'normalization_params': normalization_params,
        'dataset_properties': {
            'num_cases': len(train_image_files),
            'modalities': dataset_properties['modalities'],
            'median_spacing': target_spacing
        }
    }

    with open(results_dir / 'nnunet_fingerprint.json', 'w') as f:
        json.dump(fingerprint, f, indent=2)

    # Multi-Site MRI Harmonization Configuration
    print("🔧 MULTI-SITE HARMONIZATION PIPELINE:")

    harmonization_methods = ['n4_bias_correction', 'z_score_harmonization']
    print(f"Implementing: {', '.join(harmonization_methods)}")

    # Create harmonization reference for z-score standardization
    harmonization_reference = {}
    for modality in dataset_properties['modalities']:
        if modality in normalization_params:
            harmonization_reference[modality] = {
                'target_mean': 0.0,
                'target_std': 1.0,
                'clip_percentiles': [
                    normalization_params[modality]['clip_lower'],
                    normalization_params[modality]['clip_upper']
                ]
            }

    # Save harmonization parameters
    harmonization_config = {
        'methods': harmonization_methods,
        'harmonization_reference': harmonization_reference,
        'n4_parameters': {
            'max_iterations': [50, 50, 30, 20],
            'convergence_threshold': 1e-6,
            'bspline_fitting_distance': 300,
            'shrink_factor': 3
        }
    }

    with open(results_dir / 'harmonization_config.json', 'w') as f:
        json.dump(harmonization_config, f, indent=2)

    print("✅ Multi-site harmonization parameters configured")

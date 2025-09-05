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

    # Professional nnU-Net v2 Preprocessing Pipeline
    print("🚀 PROFESSIONAL nnU-Net v2 PREPROCESSING:")
    print(f"Processing {len(train_image_files)} volumes with full harmonization pipeline")
    print("=" * 60)

    processing_stats = []

    for idx in tqdm(range(len(train_image_files)),
                    desc="nnU-Net v2 preprocessing",
                    disable=not sys.stdout.isatty()):

        img_path = train_image_files[idx]
        label_path = train_label_files[idx]
        case_id = img_path.stem.split('.')[0]

        try:
            # Load volumes and extract metadata
            img_nifti = nib.load(img_path)
            label_nifti = nib.load(label_path)

            img_data = img_nifti.get_fdata().astype(np.float32)
            label_data = label_nifti.get_fdata().astype(np.uint8)
            original_spacing = img_nifti.header.get_zooms()[:3]
            original_affine = img_nifti.affine

            # Optimized brain extraction
            all_voxels = img_data[img_data != 0]
            low_thresh = np.percentile(all_voxels, 2.0)

            # Create initial mask from FLAIR (modality 0)
            brain_mask = img_data[..., 0] > low_thresh

            # Apply dilation with proven parameters
            brain_mask = ndimage.binary_dilation(brain_mask, structure=np.ones((5, 5, 5)))

            # Fill holes
            brain_mask = ndimage.binary_fill_holes(brain_mask)

            # Keep largest connected component
            labeled_mask, num_labels = ndimage.label(brain_mask)
            if num_labels > 1:
                sizes = ndimage.sum(brain_mask, labeled_mask, range(1, num_labels + 1))
                largest_label = np.argmax(sizes) + 1
                brain_mask = labeled_mask == largest_label

            # Calculate coverage using bounding box method
            bbox = np.argwhere(brain_mask)
            if len(bbox) > 0:
                mins, maxs = bbox.min(axis=0), bbox.max(axis=0) + 1
                bbox_volume = np.prod(maxs - mins)
                brain_voxels_in_bbox = np.sum(brain_mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]])
                brain_coverage = (brain_voxels_in_bbox / bbox_volume) * 100
            else:
                brain_coverage = 0.0

            # N4 bias field correction using pre-extracted mask
            img_corrected = np.zeros_like(img_data)
            spacing = list(map(float, img_nifti.header.get_zooms()[:3]))

            for modality_idx in range(img_data.shape[-1]):
                mod_data = img_data[..., modality_idx]

                # Convert to SimpleITK images
                sitk_img_mod = sitk.GetImageFromArray(mod_data.astype(np.float32))
                sitk_mask_mod = sitk.GetImageFromArray(brain_mask.astype(np.uint8))

                # Preserve original NIfTI spacing/orientation
                sitk_img_mod.SetSpacing(spacing)
                sitk_mask_mod.SetSpacing(spacing)

                # Shrink for computational efficiency
                shrink_factor = 4
                img_shrunk = sitk.Shrink(sitk_img_mod, [shrink_factor] * 3)
                mask_shrunk = sitk.Shrink(sitk_mask_mod, [shrink_factor] * 3)

                # Enhanced N4 correction
                corrector = sitk.N4BiasFieldCorrectionImageFilter()
                corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
                corrector.SetConvergenceThreshold(1e-6)

                try:
                    corrected_shrunk = corrector.Execute(img_shrunk, mask_shrunk)

                    # Get bias field and apply to full resolution
                    log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img_mod)
                    bias_field = np.exp(sitk.GetArrayFromImage(log_bias_field))

                    corrected_fullres = mod_data / (bias_field + 1e-8)
                    img_corrected[..., modality_idx] = corrected_fullres.astype(np.float32)

                except Exception as e:
                    print(f"    ⚠️ N4 failed for modality {modality_idx}: {e}")
                    img_corrected[..., modality_idx] = mod_data


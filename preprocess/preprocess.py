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

            # nnU-Net intensity normalization with clipping
            img_normalized = np.zeros_like(img_corrected)

            for modality_idx, modality in enumerate(dataset_properties['modalities']):
                mod_data = img_corrected[..., modality_idx]

                if modality not in normalization_params:
                    img_normalized[..., modality_idx] = mod_data * brain_mask
                    continue

                # nnU-Net style percentile clipping on full image
                clip_lower = normalization_params[modality]['clip_lower']
                clip_upper = normalization_params[modality]['clip_upper']

                mod_data_clipped = np.clip(mod_data, clip_lower, clip_upper)

                # Z-score normalization using global parameters
                brain_voxels_clipped = mod_data_clipped[brain_mask]
                if len(brain_voxels_clipped) > 0:
                    mean_val = normalization_params[modality]['mean_intensity']
                    std_val = normalization_params[modality]['std_intensity']

                    if std_val > 0:
                        # Normalize full image, then apply brain mask
                        normalized_full = (mod_data_clipped - mean_val) / std_val
                        img_normalized[..., modality_idx] = normalized_full * brain_mask
                    else:
                        img_normalized[..., modality_idx] = (mod_data_clipped - mean_val) * brain_mask
                else:
                    img_normalized[..., modality_idx] = mod_data_clipped * brain_mask

            # Spatial resampling to nnU-Net target
            zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]

            # Calculate new shape with safe rounding
            new_shape = np.maximum(1, np.round(np.array(img_normalized.shape[:3]) * np.array(zoom_factors))).astype(int)

            # Resample images with cubic interpolation
            img_resampled = np.zeros((*new_shape, img_normalized.shape[-1]), dtype=np.float32)
            for modality_idx in range(img_normalized.shape[-1]):
                img_resampled[..., modality_idx] = ndimage.zoom(
                    img_normalized[..., modality_idx], zoom_factors, order=3
                )

            # Resample labels with nearest neighbor
            label_resampled = ndimage.zoom(label_data, zoom_factors, order=0).astype(np.uint8)

            # Resample brain mask
            brain_mask_resampled = ndimage.zoom(
                brain_mask.astype(np.uint8), zoom_factors, order=0
            ).astype(bool)

            # Recalculate bounding box for resampled brain mask
            bbox_resampled = np.argwhere(brain_mask_resampled)
            if len(bbox_resampled) > 0:
                mins_resampled, maxs_resampled = bbox_resampled.min(axis=0), bbox_resampled.max(axis=0) + 1
            else:
                mins_resampled, maxs_resampled = [0, 0, 0], brain_mask_resampled.shape

            # Quality control and validation
            if np.any(np.isnan(img_resampled)) or np.any(np.isinf(img_resampled)):
                img_resampled = np.nan_to_num(img_resampled, nan=0.0, posinf=0.0, neginf=0.0)

            # Validate label integrity
            valid_labels = [0, 1, 2, 3]
            invalid_voxels = ~np.isin(label_resampled, valid_labels)
            if np.any(invalid_voxels):
                label_resampled = np.where(np.isin(label_resampled, valid_labels), label_resampled, 0)

            # Check tumor preservation
            tumor_before = np.sum(label_data > 0)
            tumor_after = np.sum(label_resampled > 0)
            preservation_ratio = tumor_after / (tumor_before + 1e-8)

            # Update affine matrix for new spacing
            new_affine = original_affine.copy()
            new_affine[0, 0] = target_spacing[0] if new_affine[0, 0] > 0 else -target_spacing[0]
            new_affine[1, 1] = target_spacing[1] if new_affine[1, 1] > 0 else -target_spacing[1]
            new_affine[2, 2] = target_spacing[2] if new_affine[2, 2] > 0 else -target_spacing[2]

            # Save per-modality 3D files for nnU-Net v2
            for modality_idx, modality_suffix in enumerate(["0000", "0001", "0002", "0003"]):
                modality_img = nib.Nifti1Image(
                    img_resampled[..., modality_idx].astype(np.float32),
                    new_affine
                )
                modality_img.set_sform(new_affine, code=1)
                modality_img.set_qform(new_affine, code=1)

                modality_path = nnunet_dir / 'imagesTr' / f"{case_id}_{modality_suffix}.nii.gz"
                nib.save(modality_img, modality_path)

            # Save label with proper spatial metadata
            label_nifti_out = nib.Nifti1Image(label_resampled.astype(np.uint8), new_affine)
            label_nifti_out.set_sform(new_affine, code=1)
            label_nifti_out.set_qform(new_affine, code=1)

            output_label_path = nnunet_dir / 'labelsTr' / f"{case_id}.nii.gz"
            nib.save(label_nifti_out, output_label_path)

            # Save preprocessing metadata
            np.savez_compressed(
                processed_dir / f"{case_id}_preprocessed.npz",
                image=img_resampled.astype(np.float32),
                label=label_resampled.astype(np.uint8),
                brain_mask_full=brain_mask_resampled,
                brain_mask_roi=brain_mask_resampled[mins_resampled[0]:maxs_resampled[0],
                               mins_resampled[1]:maxs_resampled[1],
                               mins_resampled[2]:maxs_resampled[2]],
                brain_mask_bbox_mins=mins_resampled,
                brain_mask_bbox_maxs=maxs_resampled,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                original_shape=img_data.shape[:3],
                final_shape=img_resampled.shape[:3],
                normalization_applied=True,
                harmonization_applied=True,
                bias_correction_applied=True,
                histogram_matching_applied=False
            )

            # Record processing statistics
            processing_stats.append({
                'case_id': case_id,
                'success': True,
                'original_shape': img_data.shape[:3],
                'final_shape': img_resampled.shape[:3],
                'original_spacing': original_spacing,
                'final_spacing': target_spacing,
                'tumor_voxels_before': int(tumor_before),
                'tumor_voxels_after': int(tumor_after),
                'tumor_preservation_ratio': float(preservation_ratio),
                'brain_mask_coverage': float(brain_coverage),
                'intensity_range_per_modality': [
                    [float(img_resampled[..., i].min()), float(img_resampled[..., i].max())]
                    for i in range(img_resampled.shape[-1])
                ]
            })

            print(f"  ✅ {case_id} processing completed successfully!")

            # Progress logging for SageMaker CloudWatch
            if (idx + 1) % 10 == 0 or (idx + 1) == len(train_image_files):
                current_success_rate = sum(1 for s in processing_stats if s['success']) / (idx + 1) * 100
                print(f"📊 Progress: {idx + 1}/{len(train_image_files)} cases processed")
                print(f"✅ Success rate: {current_success_rate:.1f}%")
                print(f"🧠 Recent brain coverage: {brain_coverage:.1f}%")
                print(f"🎯 Recent tumor preservation: {preservation_ratio:.3f}")

                # Memory cleanup every 10 cases
                gc.collect()

        except Exception as e:
            print(f"  ❌ Error processing {case_id}: {e}")
            processing_stats.append({
                'case_id': case_id,
                'success': False,
                'error': str(e)
            })

    # Save comprehensive processing statistics
    processing_df = pd.DataFrame(processing_stats)
    processing_df.to_csv(results_dir / 'nnunet_preprocessing_stats.csv', index=False)

    successful_cases = processing_df['success'].sum()
    failed_cases = len(processing_df) - successful_cases

    print("\n" + "=" * 60)
    print("🎯 PROFESSIONAL nnU-Net v2 PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful_cases}/{len(processing_df)} volumes")
    print(f"❌ Failed cases: {failed_cases}")
    print(f"📊 Success rate: {successful_cases / len(processing_df) * 100:.1f}%")

    if successful_cases > 0:
        successful_stats = processing_df[processing_df['success']].copy()
        avg_preservation = successful_stats['tumor_preservation_ratio'].mean()
        avg_brain_coverage = successful_stats['brain_mask_coverage'].mean()

        print(f"📊 Average tumor preservation: {avg_preservation:.3f}")
        print(f"🧠 Average brain mask coverage: {avg_brain_coverage:.1f}%")

    # Create nnU-Net dataset JSON with correct training entries
    all_successful_cases = [str(stats['case_id']) for stats in processing_stats if stats['success']]

    nnunet_dataset_json = {
        "channel_names": {
            "0": "FLAIR",
            "1": "T1w",
            "2": "T1Gd",
            "3": "T2w"
        },
        "labels": {
            "background": 0,
            "edema": 1,
            "non_enhancing_tumor": 2,
            "enhancing_tumor": 3
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": int(successful_cases),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
        "nnUNet_version": "2.0",
        "dataset_name": "Dataset001_BrainTumor",
        "description": "Brain tumor segmentation from Medical Segmentation Decathlon with professional preprocessing",
        "reference": "BraTS challenge - targeting WT Dice ≥ 90, BraTS Avg ≥ 80",
        "tensorImageSize": "4D",
        "training": [{"image": f"./imagesTr/{case_id}",
                      "label": f"./labelsTr/{case_id}.nii.gz"}
                     for case_id in all_successful_cases]
    }

    with open(nnunet_dir / 'dataset.json', 'w') as f:
        json.dump(nnunet_dataset_json, f, indent=2)

    # Create cross-validation splits
    splits = []
    if len(all_successful_cases) > 1:
        kfold = KFold(n_splits=min(5, len(all_successful_cases)), shuffle=True, random_state=42)
        for train_idx, val_idx in kfold.split(all_successful_cases):
            train_cases = [all_successful_cases[i] for i in train_idx]
            val_cases = [all_successful_cases[i] for i in val_idx]
            splits.append({
                "train": train_cases,
                "val": val_cases
            })
    else:
        splits = [{"train": all_successful_cases, "val": all_successful_cases}]

    with open(nnunet_dir / 'splits_final.json', 'w') as f:
        json.dump(splits, f, indent=2)


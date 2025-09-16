#!/usr/bin/env python3
"""
Brain Tumor Segmentation Training Pipeline - SageMaker Script
by Ridwan Oladipo, MD | AI Specialist

Submit training job to SageMaker with spot instances for cost optimization
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from datetime import datetime
import os


def submit_brain_training_job(test_mode=False):
    """Submit brain tumor training job to SageMaker"""

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()

    # SageMaker execution role
    role = "arn:aws:iam::098824477125:role/service-role/AmazonSageMaker-ExecutionRole-20250810T134954"

    # Job configuration
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"brain-tumor-training-{'test' if test_mode else 'full'}-{timestamp}"

    # S3 paths using preprocessed data
    s3_bucket = "ridwan-md-brain-tumor-ai-2025"
    s3_input_data = f"s3://{s3_bucket}/outputs/20250812-100047/processed/"
    s3_output_path = f"s3://{s3_bucket}/training_outputs/"

    print(f"Brain Tumor Training Job: {job_name}")
    print(f"Input data: {s3_input_data}")
    print(f"Output path: {s3_output_path}")
    print(f"Mode: {'TEST (1 volume)' if test_mode else 'FULL DATASET (484 volumes)'}")

    # Configure parameters based on mode
    if test_mode:
        hyperparameters = {
            'no-use-full-dataset': '',
            'max-volumes': 1,
            'random-seed': 42
        }
        instance_type = 'ml.g4dn.xlarge'
        max_run = 3600
        max_wait = 7200
    else:
        hyperparameters = {
            'use-full-dataset': '',
            'random-seed': 42
        }
        instance_type = 'ml.g5.4xlarge'
        max_run = 86400
        max_wait = 172800

    # Environment variables
    environment = {
        'PYTHONUNBUFFERED': '1',
    }

    # Tags for cost tracking
    tags = [
        {'Key': 'Project', 'Value': 'BrainTumorSegmentation'},
        {'Key': 'Owner', 'Value': 'RidwanOladipo'},
        {'Key': 'Environment', 'Value': 'Research'},
        {'Key': 'Phase', 'Value': 'Training'},
        {'Key': 'Mode', 'Value': 'Test' if test_mode else 'Production'}
    ]

    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='../training/',
        role=role,
        instance_count=1,
        instance_type=instance_type,

        # Cost optimization with spot instances
        use_spot_instances=True,
        max_wait=max_wait,
        max_run=max_run,

        # Checkpointing for spot interruption recovery
        checkpoint_s3_uri=f"s3://{s3_bucket}/training_checkpoints/",
        checkpoint_local_path="/opt/ml/checkpoints",

        # Framework configuration
        framework_version='1.12',
        py_version='py38',

        # Volume configuration
        volume_size=60,

        # Job configuration
        job_name=job_name,
        output_path=s3_output_path,
        hyperparameters=hyperparameters,
        environment=environment,
        tags=tags
    )

    # Start training job
    estimator.fit({
        'training': TrainingInput(s3_input_data)
    }, wait=False)

    print(f"Job submitted successfully!")
    print(f"Job Name: {job_name}")
    print(f"Instance: {instance_type}")
    print(f"Using spot instances")
    print(f"Checkpointing enabled")
    print(f"Monitor progress in SageMaker Console")
    print(f"Direct link: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    return estimator


def check_preprocessed_data(bucket, prefix):
    """Verify that preprocessed data exists in S3"""
    s3 = boto3.client('s3')

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)

        if 'Contents' in response:
            # Count .npz files specifically
            npz_files = [obj for obj in response['Contents'] if obj['Key'].endswith('.npz')]

            print(f"Found {len(npz_files)} preprocessed .npz files in s3://{bucket}/{prefix}")

            # Show sample files
            print("Sample preprocessed files:")
            for obj in npz_files[:5]:
                print(f"  • {obj['Key']} ({obj['Size'] / 1024 / 1024:.1f} MB)")

            if len(npz_files) > 0:
                return True
            else:
                print("No .npz files found - training requires preprocessed data")
                return False
        else:
            print(f"No data found in s3://{bucket}/{prefix}")
            return False

    except Exception as e:
        print(f"Error checking S3 data: {e}")
        return False


def upload_training_source():
    """Verify training source code files are present"""
    print("Verifying training source code in ./training/ directory:")
    print("  • ./training/train.py")
    print("  • ./training/requirements.txt")

    required_files = ['../training/train.py', '../training/requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False

    print("All required training source files found")
    return True


if __name__ == "__main__":
    print("Brain Tumor Segmentation - SageMaker Training Job Submission")
    print("=" * 70)

    # Pre-flight checks
    print("Pre-flight checks...")

    # Check training source code
    if not upload_training_source():
        print("Source code check failed. Please ensure files are in training/ directory.")
        exit(1)

    # Check preprocessed data in S3
    bucket = "ridwan-md-brain-tumor-ai-2025"
    prefix = "outputs/20250812-100047/processed/"

    print(f"\nChecking preprocessed data in S3...")
    if not check_preprocessed_data(bucket, prefix):
        print("Preprocessed data check failed.")
        print("Ensure preprocessing pipeline completed successfully")
        print("Check the correct S3 path for your preprocessed .npz files")
        exit(1)

    # Choose mode
    print(f"\nSelect training mode:")
    print(f"1. TEST MODE (1 volume, ml.g4dn.xlarge)")
    print(f"2. FULL TRAINING (484 volumes, ml.g5.4xlarge)")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        test_mode = True
        print(f"\nSubmitting TEST training job...")
    elif choice == '2':
        test_mode = False
        print(f"\nSubmitting FULL DATASET training job...")
    else:
        print("Invalid choice. Defaulting to TEST mode.")
        test_mode = True

    try:
        estimator = submit_brain_training_job(test_mode=test_mode)

        print(f"\nNext steps:")
        print(f"1. Monitor job progress in SageMaker Console")
        print(f"2. Check CloudWatch logs for detailed training metrics")
        print(f"3. Watch for WT Dice scores approaching 90% target")
        print(f"4. Review trained models in S3 when complete")
        print(f"5. Proceed to evaluation notebook with best model")

        if not test_mode:
            print(f"\nFULL TRAINING TARGET METRICS:")
            print(f"   • Whole Tumor (WT) Dice Score ≥ 90%")
            print(f"   • BraTS Average Score ≥ 80%")
            print(f"   • Training with 484 preprocessed volumes")
            print(f"   • 80/10/10 train/val/test split")

        print(f"\nTraining job submitted successfully!")

    except Exception as e:
        print(f"Job submission failed: {e}")
        print(f"Common fixes:")
        print(f"  • Ensure SageMaker execution role is configured")
        print(f"  • Check AWS credentials: aws configure")
        print(f"  • Verify S3 bucket permissions")
        print(f"  • Check region settings")
        print(f"  • Ensure preprocessed data exists in S3")
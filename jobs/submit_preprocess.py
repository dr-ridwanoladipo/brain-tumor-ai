#!/usr/bin/env python3
"""
🧠 SageMaker Preprocessing Job Submission Script
Brain Tumor Segmentation Preprocessing Pipeline
by Ridwan Oladipo, MD | AI Specialist

Submit preprocessing job to SageMaker with spot instances for cost optimization
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from datetime import datetime
import os


def submit_brain_preprocess_job(test_mode=True):
    """Submit brain tumor preprocessing job to SageMaker"""

    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()

    # Your SageMaker execution role
    role = "arn:aws:iam::098824477125:role/service-role/AmazonSageMaker-ExecutionRole-20250810T134954"

    # Job configuration
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"brain-tumor-preprocess-{'test' if test_mode else 'full'}-{timestamp}"

    # S3 paths
    s3_bucket = "ridwan-md-brain-tumor-ai-2025"
    s3_input_data = f"s3://{s3_bucket}/brain-tumor-data/"
    s3_output_path = f"s3://{s3_bucket}/outputs/"

    print(f"🧠 Submitting Brain Tumor Preprocessing Job: {job_name}")
    print(f"📊 Input data: {s3_input_data}")
    print(f"📁 Output path: {s3_output_path}")
    print(f"🔬 Mode: {'TEST (1 volume)' if test_mode else 'FULL DATASET'}")

    # Hyperparameters - TEST vs FULL mode
    if test_mode:
        hyperparameters = {
            'no-use-full-dataset': '',
            'max-volumes': 1,
            'random-seed': 42
        }
    else:
        hyperparameters = {
            'use-full-dataset': '',
            'random-seed': 42
        }

    # Environment variables
    environment = {
        'PYTHONUNBUFFERED': '1',
    }

    # Tags for cost tracking
    tags = [
        {'Key': 'Project', 'Value': 'BrainTumorSegmentation'},
        {'Key': 'Owner', 'Value': 'RidwanOladipo'},
        {'Key': 'Environment', 'Value': 'Research'},
        {'Key': 'Mode', 'Value': 'Test' if test_mode else 'Production'}
    ]

    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='preprocess.py',
        source_dir='preprocess/',
        role=role,
        instance_count=1,
        instance_type='ml.r5.2xlarge',

        # Cost optimization with spot instances
        use_spot_instances=True,
        max_wait=39600,  # 11 hours
        max_run=36000 if not test_mode else 1800,  # 10 hours for full, 30 min for test

        # Checkpointing
        checkpoint_s3_uri=f"s3://{s3_bucket}/checkpoints/",
        checkpoint_local_path="/opt/ml/checkpoints",

        # Framework configuration
        framework_version='1.12',
        py_version='py38',

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

    print(f"✅ Job submitted successfully!")
    print(f"📊 Job Name: {job_name}")
    print(f"💰 Using spot instances")
    print(f"🛡️ Checkpointing enabled")
    print(f"📈 Monitor progress in SageMaker Console")
    print(f"🔗 Direct link: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")

    return estimator


def check_s3_data(bucket, prefix):
    """Verify that input data exists in S3"""
    s3 = boto3.client('s3')

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)

        if 'Contents' in response:
            print(f"✅ Found {len(response['Contents'])} objects in s3://{bucket}/{prefix}")

            # Show sample files
            print("📁 Sample files:")
            for obj in response['Contents'][:5]:
                print(f"  • {obj['Key']} ({obj['Size'] / 1024 / 1024:.1f} MB)")

            return True
        else:
            print(f"❌ No data found in s3://{bucket}/{prefix}")
            return False

    except Exception as e:
        print(f"❌ Error checking S3 data: {e}")
        return False


def upload_source_code():
    """Verify source code files are present"""
    print("📦 Verifying source code in ./preprocess/ directory:")
    print("  • ./preprocess/preprocess.py")
    print("  • ./preprocess/requirements.txt")

    # Check if files exist locally
    required_files = ['preprocess/preprocess.py', 'preprocess/requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    print("✅ All required source files found")
    return True


if __name__ == "__main__":
    print("🧠 Brain Tumor Segmentation - SageMaker Preprocess Job Submission")
    print("=" * 60)

    # Pre-flight checks
    print("🔍 Pre-flight checks...")

    # Check source code
    if not upload_source_code():
        print("❌ Source code check failed. Please fix and retry.")
        exit(1)

    # Check S3 data
    bucket = "ridwan-md-brain-tumor-ai-2025"
    prefix = "brain-tumor-data/"

    print(f"\n📊 Checking input data in S3...")
    if not check_s3_data(bucket, prefix):
        print("❌ S3 data check failed. Please upload data and retry.")
        print("💡 Run this to upload:")
        print(f"aws s3 sync brain-tumor-data/ s3://{bucket}/{prefix} --exact-timestamps")
        exit(1)

    # Submit job
    print(f"\n🚀 Submitting FULL DATASET preprocessing job...")

    try:
        estimator = submit_brain_preprocess_job(test_mode=False)

        print(f"\n📋 Next steps:")
        print(f"1. ⏳ Monitor job progress in SageMaker Console")
        print(f"2. 📊 Check CloudWatch logs for detailed output")
        print(f"3. 📁 Review results in S3 when complete")
        print(f"4. 🧠 Proceed to nnU-Net training with preprocessed volumes")

        print(f"\n✅ Job submitted successfully!")

    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        print(f"💡 Common fixes:")
        print(f"  • Ensure SageMaker execution role is configured")
        print(f"  • Check AWS credentials: aws configure")
        print(f"  • Verify S3 bucket permissions")
        print(f"  • Check region settings")
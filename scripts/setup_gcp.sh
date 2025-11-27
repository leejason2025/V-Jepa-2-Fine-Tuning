#!/bin/bash
# Setup script for GCP authentication and DROID dataset access

set -e

echo "=== GCP Setup for V-JEPA2-AC Fine-tuning ==="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI not found. Please install Google Cloud SDK:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil not found. Please install Google Cloud SDK:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "1. Authenticating with Google Cloud..."
gcloud auth application-default login

echo ""
echo "2. Testing access to DROID dataset..."

# Test access to DROID bucket
if gsutil ls gs://gresearch/robotics/droid > /dev/null 2>&1; then
    echo "✓ Successfully accessed DROID dataset bucket"
else
    echo "✗ Could not access DROID dataset bucket"
    echo "  Make sure you have permissions to access gs://gresearch/robotics/"
    exit 1
fi

# Test access to debug dataset
if gsutil ls gs://gresearch/robotics/droid_100 > /dev/null 2>&1; then
    echo "✓ Successfully accessed DROID debug dataset (100 episodes)"
else
    echo "⚠ Could not access DROID debug dataset"
fi

echo ""
echo "3. GCP Project Configuration"
read -p "Enter your GCP project ID (or press Enter to skip): " PROJECT_ID

if [ -n "$PROJECT_ID" ]; then
    gcloud config set project "$PROJECT_ID"
    echo "✓ Set GCP project to: $PROJECT_ID"

    # Ask about checkpoint bucket
    read -p "Enter GCS bucket name for checkpoints (or press Enter to skip): " CHECKPOINT_BUCKET

    if [ -n "$CHECKPOINT_BUCKET" ]; then
        # Check if bucket exists
        if gsutil ls "gs://$CHECKPOINT_BUCKET" > /dev/null 2>&1; then
            echo "✓ Checkpoint bucket exists: gs://$CHECKPOINT_BUCKET"
        else
            read -p "Bucket doesn't exist. Create it? [y/N]: " CREATE_BUCKET
            if [ "$CREATE_BUCKET" = "y" ] || [ "$CREATE_BUCKET" = "Y" ]; then
                gsutil mb "gs://$CHECKPOINT_BUCKET"
                echo "✓ Created checkpoint bucket: gs://$CHECKPOINT_BUCKET"
            fi
        fi

        # Update config file
        if [ -f "configs/default_config.yaml" ]; then
            echo ""
            echo "Update configs/default_config.yaml with:"
            echo "  gcp:"
            echo "    project_name: \"$PROJECT_ID\""
            echo "    checkpoint_bucket: \"$CHECKPOINT_BUCKET\""
        fi
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Install Python dependencies: pip install -r requirements.txt"
echo "2. Update configs/default_config.yaml with your settings"
echo "3. Run training: python train.py --config configs/default_config.yaml --debug"
echo ""

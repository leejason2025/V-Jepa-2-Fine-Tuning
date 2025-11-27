#!/usr/bin/env python3
"""Test DROID dataset loading."""

import tensorflow as tf
from google.cloud import storage
import os

# Anonymous GCS access
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''

print("Testing DROID dataset loading...")

# Test 1: Check files exist
print("\n1. Checking files...")
file_pattern = "gs://gresearch/robotics/droid_100/1.0.0/r2d2_faceblur-train.tfrecord-*"
print(f"File pattern: {file_pattern}")

try:
    files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    file_list = list(files.take(5).as_numpy_iterator())
    print(f"✓ Found {len(file_list)} files (showing first 5)")
    for f in file_list:
        print(f"  - {f.decode()}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test 2: Load first TFRecord
print("\n2. Loading first TFRecord...")
try:
    first_file = file_list[0].decode()
    dataset = tf.data.TFRecordDataset(first_file)

    # Get first record
    for i, record in enumerate(dataset.take(1)):
        print(f"✓ Successfully loaded record")
        print(f"  Record type: {type(record)}")
        print(f"  Record size: {len(record.numpy())} bytes")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✓ Basic dataset loading works!")
print("\nNote: Full RLDS parsing not implemented yet - that's the next step.")

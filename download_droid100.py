#!/usr/bin/env python3
"""
Download DROID-100 dataset from GCS to local storage.
"""

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pickle

# Create data directory
data_dir = "/workspace/V-Jepa-2-Fine-Tuning/data/droid_100"
os.makedirs(data_dir, exist_ok=True)

print(f"Downloading DROID-100 to {data_dir}...")
print("This will download ~5-10 GB of data")

# GCS path
gcs_path = "gs://gresearch/robotics/droid_100/1.0.0/"
pattern = f"{gcs_path}r2d2_faceblur-train.tfrecord-*"

# Get files
print(f"Finding files at {pattern}...")
files = tf.io.gfile.glob(pattern)
print(f"Found {len(files)} TFRecord files")

if len(files) == 0:
    print("ERROR: No files found!")
    exit(1)

# Download and convert to simpler format
dataset = tf.data.TFRecordDataset(files)

episodes = []
count = 0
max_episodes = 100  # DROID-100 has 100 episodes

print(f"\nDownloading {max_episodes} episodes...")

for serialized_example in tqdm(dataset, total=max_episodes):
    if count >= max_episodes:
        break

    # Parse
    feature_desc = {
        'steps/observation/wrist_image_left': tf.io.VarLenFeature(tf.string),
        'steps/action': tf.io.VarLenFeature(tf.float32),
        'steps/observation/cartesian_position': tf.io.VarLenFeature(tf.float32),
        'steps/observation/gripper_position': tf.io.VarLenFeature(tf.float32),
    }

    parsed = tf.io.parse_single_example(serialized_example, feature_desc)

    # Extract
    images_bytes = tf.sparse.to_dense(parsed['steps/observation/wrist_image_left']).numpy()
    cart_pos = tf.sparse.to_dense(parsed['steps/observation/cartesian_position']).numpy()
    grip_pos = tf.sparse.to_dense(parsed['steps/observation/gripper_position']).numpy()

    # Save episode
    episode_data = {
        'images_bytes': images_bytes,  # Keep as bytes to save space
        'cartesian_position': cart_pos,
        'gripper_position': grip_pos,
    }

    episode_file = os.path.join(data_dir, f"episode_{count:04d}.pkl")
    with open(episode_file, 'wb') as f:
        pickle.dump(episode_data, f)

    count += 1

print(f"\n✓ Downloaded {count} episodes to {data_dir}")
print(f"Total size: {os.popen(f'du -sh {data_dir}').read().split()[0]}")

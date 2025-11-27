#!/usr/bin/env python3
"""Test script to verify DROID dataset loading."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from utils.config import load_config
from data import create_dataloader


def test_dataloader(debug_mode=True):
    """Test DROID dataloader.

    Args:
        debug_mode: If True, use droid_100 (smaller dataset)
    """
    print("="*60)
    print("Testing DROID Dataset Loader")
    print("="*60)

    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'default_config.yaml'
    print(f"\nLoading config from: {config_path}")
    config = load_config(str(config_path))

    # Override debug mode
    config.data.debug_mode = debug_mode
    config.training.per_device_batch_size = 1

    print(f"\nDataset mode: {'Debug (100 episodes)' if debug_mode else 'Full dataset'}")
    print(f"Batch size: {config.training.per_device_batch_size}")
    print(f"Camera view: {config.data.camera_view}")
    print(f"Frames per clip: {config.data.frames_per_clip}")
    print(f"Resolution: {config.data.video_resolution}x{config.data.video_resolution}")

    # Create dataloader
    print("\n" + "-"*60)
    print("Creating dataloader...")
    print("-"*60)

    try:
        dataloader = create_dataloader(config, split='train', shuffle=True)
        print(f"✓ Dataloader created successfully")
        print(f"  Estimated batches per epoch: ~{len(dataloader)}")

    except Exception as e:
        print(f"✗ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading a batch
    print("\n" + "-"*60)
    print("Loading test batch...")
    print("-"*60)

    try:
        batch = next(iter(dataloader))

        print(f"✓ Successfully loaded batch")
        print(f"\nBatch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

        # Validate shapes
        B, T, C, H, W = batch['frames'].shape
        expected_T = config.data.frames_per_clip
        expected_HW = config.data.video_resolution

        print(f"\nShape validation:")
        print(f"  Batch size: {B} ✓")
        print(f"  Timesteps: {T} {'✓' if T == expected_T else '✗ (expected ' + str(expected_T) + ')'}")
        print(f"  Channels: {C} {'✓' if C == 3 else '✗ (expected 3)'}")
        print(f"  Height: {H} {'✓' if H == expected_HW else '✗ (expected ' + str(expected_HW) + ')'}")
        print(f"  Width: {W} {'✓' if W == expected_HW else '✗ (expected ' + str(expected_HW) + ')'}")

        _, T_a, A = batch['actions'].shape
        print(f"  Action dim: {A} {'✓' if A == 7 else '✗ (expected 7)'}")

        _, T_s, S = batch['states'].shape
        print(f"  State dim: {S} {'✓' if S == 7 else '✗ (expected 7)'}")

        # Test loading multiple batches
        print("\n" + "-"*60)
        print("Testing multiple batches...")
        print("-"*60)

        num_test_batches = 3
        for i in range(num_test_batches):
            batch = next(iter(dataloader))
            print(f"  Batch {i+1}/{num_test_batches}: ✓")

        print(f"\n✓ Successfully loaded {num_test_batches} batches")

    except Exception as e:
        print(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*60)
    print("Dataset test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test DROID dataset loading')
    parser.add_argument(
        '--full',
        action='store_true',
        help='Test with full dataset instead of debug (100 episodes)'
    )
    args = parser.parse_args()

    test_dataloader(debug_mode=not args.full)

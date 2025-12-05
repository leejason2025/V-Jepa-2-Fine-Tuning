#!/usr/bin/env python3
"""Test DROID GCS dataloader."""

from droid_src.data import init_data

print("Testing DROID dataloader...")
dataloader = init_data(
    debug_mode=True,  # Use droid_100
    batch_size=2,
    frames_per_clip=8,
    fps=4,
    crop_size=256,
    camera_view="wrist",
    shuffle_buffer_size=10,
    num_workers=0,  # Single process for testing
)

print("✓ Dataloader created")
print("Fetching first batch...")

for batch_idx, (buffer, actions, states) in enumerate(dataloader):
    print(f"\nBatch {batch_idx}:")
    print(f"  buffer shape: {buffer.shape}  # [B, T, H, W, C]")
    print(f"  actions shape: {actions.shape}  # [B, T-1, 7]")
    print(f"  states shape: {states.shape}  # [B, T, 7]")
    print(f"  buffer dtype: {buffer.dtype}")
    print(f"  actions dtype: {actions.dtype}")

    if batch_idx == 0:
        print("\n✓ Dataloader working! Stopping after first batch.")
        break

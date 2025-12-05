#!/usr/bin/env python3
"""
V-JEPA2-AC + LoRA training with LOCAL DROID-100 data.
This uses the LocalDROIDDataset to load from local pickle files.
"""

import sys
sys.path.insert(0, '/workspace/vjepa2')

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import copy
import time
import os

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit

# Import our local DROID dataset
from droid_local_dataset import create_dataloader


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    print("="*60)
    print("V-JEPA2-AC + LoRA Training (Local DROID-100)")
    print("="*60)

    # Create local DROID dataloader
    print("\nInitializing local DROID-100 dataloader...")
    dataloader = create_dataloader(
        data_dir="/workspace/V-Jepa-2-Fine-Tuning/data/droid_100",
        batch_size=2,
        frames_per_clip=8,
        crop_size=256,
        num_workers=0,
    )
    print(f"✓ Dataloader created")

    # Initialize models
    print("\nInitializing models...")
    encoder = video_vit.vit_giant_xformers(
        patch_size=16, num_frames=8, tubelet_size=2,
        uniform_power=True, use_sdpa=True, use_rope=True,
        use_silu=False, checkpoint_activations=True
    )

    predictor = vit_ac_pred.vit_ac_predictor(
        img_size=256, patch_size=16, num_frames=8, tubelet_size=2,
        embed_dim=encoder.embed_dim, predictor_embed_dim=1024,
        action_embed_dim=7, depth=24, is_frame_causal=True,
        num_heads=16, uniform_power=True, use_rope=True,
        use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=True
    )

    target_encoder = copy.deepcopy(encoder)

    # Move to device
    encoder = encoder.to(device).to(dtype)
    predictor = predictor.to(device).to(dtype)
    target_encoder = target_encoder.to(device).to(dtype)

    # Freeze encoders
    for param in encoder.parameters():
        param.requires_grad = False
    for param in target_encoder.parameters():
        param.requires_grad = False

    # Add LoRA to predictor
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        bias="none", task_type=None
    )
    predictor = get_peft_model(predictor, lora_config)
    predictor.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-5, weight_decay=0.04)

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    sys.stdout.flush()

    # Training loop
    max_steps = 1250
    save_every = 250
    step = 0
    data_iter = iter(dataloader)
    start_time = time.time()

    print("Fetching first batch from local data...")
    sys.stdout.flush()

    while step < max_steps:
        try:
            video_batch, actions, states = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            video_batch, actions, states = next(data_iter)

        # Convert [B,T,H,W,C] -> [B,C,T,H,W]
        video_batch = video_batch.permute(0, 4, 1, 2, 3).to(device, dtype=dtype)
        actions = actions.to(device, dtype=dtype)
        states = states.to(device, dtype=dtype)

        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            # Target
            with torch.no_grad():
                h_target = target_encoder(video_batch)

            # Context
            h_enc = encoder(video_batch)

            # Predictor
            h_pred = predictor(h_enc, actions, states)

            # Loss
            loss = F.smooth_l1_loss(h_pred, h_target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=0.5)

        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p_enc, p_tgt in zip(encoder.parameters(), target_encoder.parameters()):
                p_tgt.data.mul_(0.996).add_(0.004 * p_enc.data)

        step += 1

        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
            sys.stdout.flush()

        # Save checkpoint
        if step % save_every == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            checkpoint = {
                'step': step,
                'predictor': predictor.state_dict(),
                'encoder': encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            path = f'./checkpoints/droid_local_step{step}.pt'
            torch.save(checkpoint, path)
            print(f"\n✓ Saved checkpoint: {path}")
            sys.stdout.flush()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save final checkpoint
    checkpoint = {
        'step': step,
        'predictor': predictor.state_dict(),
        'encoder': encoder.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    path = f'./checkpoints/droid_local_step{step}_final.pt'
    torch.save(checkpoint, path)
    print(f"✓ Saved final checkpoint: {path}")


if __name__ == '__main__':
    main()

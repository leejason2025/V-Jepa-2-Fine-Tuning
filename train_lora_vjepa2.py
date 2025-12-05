#!/usr/bin/env python3
"""
V-JEPA2-AC fine-tuning with LoRA on DROID (GCS streaming).

This script:
1. Loads V-JEPA2-AC pretrained model from vjepa2 repo
2. Wraps predictor with LoRA adapters (encoder frozen)
3. Uses our GCS streaming DROID dataloader
4. Implements V-JEPA2's training loop (simplified for single-GPU)
"""

import argparse
import copy
import os
import sys
import time

# Add vjepa2 to path
sys.path.insert(0, '/workspace/vjepa2')

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model

from droid_src.data import init_data

# Import V-JEPA2 model initialization
import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit

# Seed for reproducibility
_GLOBAL_SEED = 239
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def init_models(device='cuda', dtype=torch.bfloat16, pretrained_path=None):
    """Initialize V-JEPA2-AC models and add LoRA to predictor."""
    print("Initializing V-JEPA2-AC models...")

    # Initialize encoder (vit_giant)
    encoder = video_vit.vit_giant_xformers(
        patch_size=16,
        num_frames=8,  # 8 frames
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_rope=True,
        use_silu=False,
        checkpoint_activations=True,
    )

    # Initialize predictor (action-conditioned)
    predictor = vit_ac_pred.vit_ac_predictor(
        img_size=256,
        patch_size=16,
        num_frames=8,
        tubelet_size=2,
        embed_dim=encoder.embed_dim,  # Use encoder's actual embed dim (1408 for giant)
        predictor_embed_dim=1024,
        action_embed_dim=7,
        depth=24,
        is_frame_causal=True,
        num_heads=16,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=True,
    )

    # Create target encoder (EMA copy)
    target_encoder = copy.deepcopy(encoder)

    # Move to device and dtype
    encoder = encoder.to(device).to(dtype)
    predictor = predictor.to(device).to(dtype)
    target_encoder = target_encoder.to(device).to(dtype)

    # Load pretrained weights if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Load encoder
        if 'target_encoder' in checkpoint:
            encoder_dict = {k.replace('module.', ''): v for k, v in checkpoint['target_encoder'].items()}
            encoder.load_state_dict(encoder_dict, strict=False)
            target_encoder.load_state_dict(encoder_dict, strict=False)
            print("✓ Loaded pretrained encoder")

        del checkpoint

    # Freeze encoder and target encoder
    for param in encoder.parameters():
        param.requires_grad = False
    for param in target_encoder.parameters():
        param.requires_grad = False

    print("✓ Encoder and target_encoder frozen")

    # Add LoRA to predictor
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,  # Reduced from 32
        lora_dropout=0.05,
        target_modules=["qkv", "proj", "fc1", "fc2"],  # Attention + MLP
        bias="none",
        task_type=None,
    )

    predictor = get_peft_model(predictor, lora_config)
    predictor.print_trainable_parameters()

    return encoder, predictor, target_encoder


def apply_masks(x, masks_enc, masks_pred):
    """Apply encoder and predictor masks (from V-JEPA2)."""
    # masks_enc: list of tensors for encoder
    # masks_pred: list of tensors for predictor
    # For now, simplified - V-JEPA2 uses complex masking strategy
    return x, None, None


def forward_target(target_encoder, video_batch):
    """Forward pass through target encoder."""
    with torch.no_grad():
        h = target_encoder(video_batch)
    return h


def forward_context(encoder, video_batch, masks_enc):
    """Forward pass through context encoder."""
    h = encoder(video_batch)
    return h


def forward_predictor(predictor, h_enc, masks_enc, masks_pred, actions, states):
    """Forward pass through predictor with actions."""
    # V-JEPA2-AC predictor takes:
    # - context embeddings
    # - actions
    # - states
    h_pred = predictor(h_enc, actions, states)
    return h_pred


def loss_fn(h_pred, h_target, masks_pred=None):
    """V-JEPA2 loss: smooth L1 between predicted and target representations."""
    loss = F.smooth_l1_loss(h_pred, h_target, reduction='mean')
    return loss


def train_one_step(encoder, predictor, target_encoder, optimizer, scaler,
                    video_batch, actions, states, device, dtype, use_amp=True):
    """Single training step."""

    # Convert from [B, T, H, W, C] to [B, C, T, H, W]
    video_batch = video_batch.permute(0, 4, 1, 2, 3)
    video_batch = video_batch.to(device, dtype=dtype, non_blocking=True)
    actions = actions.to(device, dtype=dtype, non_blocking=True)
    states = states.to(device, dtype=dtype, non_blocking=True)

    # For simplicity, we'll skip the complex masking strategy
    # In full V-JEPA2, this uses predictor-based masking
    masks_enc, masks_pred = None, None

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype):
        # Target encoder (no grad)
        h_target = forward_target(target_encoder, video_batch)

        # Context encoder (frozen)
        h_enc = forward_context(encoder, video_batch, masks_enc)

        # Predictor (LoRA trainable)
        h_pred = forward_predictor(predictor, h_enc, masks_enc, masks_pred, actions, states)

        # Loss
        loss = loss_fn(h_pred, h_target, masks_pred)

    # Backward
    optimizer.zero_grad()
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return loss.item()


def update_target_encoder(encoder, target_encoder, momentum=0.996):
    """Update target encoder with EMA of encoder."""
    with torch.no_grad():
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)


def save_checkpoint(encoder, predictor, target_encoder, optimizer, step, config, save_dir):
    """Save checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'step': step,
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),  # Includes LoRA weights
        'target_encoder': target_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    path = os.path.join(save_dir, f'checkpoint_step{step}.pt')
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Extract params
    data_cfg = config['data']
    train_cfg = config['training']
    lora_cfg = config.get('lora', {})

    debug_mode = data_cfg.get('debug_mode', False)
    batch_size = data_cfg.get('batch_size', 8)
    frames_per_clip = data_cfg.get('frames_per_clip', 8)
    fps = data_cfg.get('fps', 4)
    crop_size = data_cfg.get('crop_size', 256)
    shuffle_buffer_size = data_cfg.get('shuffle_buffer_size', 100)
    num_workers = data_cfg.get('num_workers', 4)

    max_steps = train_cfg.get('max_steps', 1250)
    save_every_n_steps = train_cfg.get('save_every_n_steps', 250)
    lr = train_cfg.get('learning_rate', 1e-4)
    checkpoint_dir = train_cfg.get('checkpoint_dir', 'checkpoints')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    use_amp = True

    print("=" * 60)
    print("V-JEPA2-AC + LoRA Fine-Tuning on DROID (GCS)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Debug mode: {debug_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Frames per clip: {frames_per_clip}")
    print(f"Max steps: {max_steps}")
    print(f"Checkpoint every: {save_every_n_steps} steps")
    print("=" * 60)

    # Initialize dataloader
    print("\nInitializing GCS DROID dataloader...")
    dataloader = init_data(
        debug_mode=debug_mode,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        fps=fps,
        crop_size=crop_size,
        camera_view='wrist',
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_mem=True,
    )
    print("✓ Dataloader initialized")

    # Initialize models
    pretrained_path = config.get('pretrained_checkpoint', None)
    encoder, predictor, target_encoder = init_models(
        device=device,
        dtype=dtype,
        pretrained_path=pretrained_path
    )

    # Initialize optimizer (only predictor parameters)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.04)
    # Note: GradScaler not compatible with bfloat16, only use with float16
    scaler = None  # Disabled for bfloat16

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Training loop
    step = 0
    data_iter = iter(dataloader)
    start_time = time.time()

    while step < max_steps:
        try:
            video_batch, actions, states = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            video_batch, actions, states = next(data_iter)

        # Train step
        loss = train_one_step(
            encoder, predictor, target_encoder,
            optimizer, scaler,
            video_batch, actions, states,
            device, dtype, use_amp
        )

        # Update target encoder (EMA)
        update_target_encoder(encoder, target_encoder, momentum=0.996)

        step += 1

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{max_steps} | Loss: {loss:.4f} | Time: {elapsed:.1f}s")

        # Save checkpoint
        if step % save_every_n_steps == 0:
            save_checkpoint(
                encoder, predictor, target_encoder,
                optimizer, step, config, checkpoint_dir
            )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Save final checkpoint
    save_checkpoint(
        encoder, predictor, target_encoder,
        optimizer, step, config, checkpoint_dir
    )


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
V-JEPA2-AC + LoRA training with CORRECT loss function.
Based on official implementation in /workspace/vjepa2/app/vjepa_droid/train.py
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

from droid_local_dataset import create_dataloader


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    # Configuration
    batch_size = 2
    max_steps = 50  # Start with 50 steps to test
    save_every = 50
    lr = 2e-5

    # V-JEPA2-AC specific params
    tokens_per_frame = 256  # 16x16 grid
    auto_steps = 2  # Autoregressive rollout steps
    normalize_reps = True
    loss_exp = 1.0  # L1 loss

    print("="*60)
    print("V-JEPA2-AC + LoRA Training (CORRECT Loss Function)")
    print("="*60)

    # Create local DROID dataloader
    print("\nInitializing local DROID-100 dataloader...")
    dataloader = create_dataloader(
        data_dir="/workspace/V-Jepa-2-Fine-Tuning/data/droid_100",
        batch_size=batch_size,
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
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2,  # num_frames=16 for larger attention mask
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
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=0.04)

    # Loss function (from official implementation)
    def loss_fn(z, h):
        """
        Args:
            z: Predicted representations [B, T*H*W, D]
            h: Target representations [B, (T+1)*H*W, D] (includes first frame)
        """
        # Skip first frame in target (line 440)
        _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

    def forward_target(video_batch):
        """
        Encode video frames with frozen target encoder.
        From official implementation lines 408-415.

        Args:
            video_batch: [B, T, H, W, C]
        Returns:
            h: [B, T*tokens_per_frame, D]
        """
        with torch.no_grad():
            # Permute to [B, C, T, H, W]
            c = video_batch.permute(0, 4, 1, 2, 3)
            # Flatten batch and time: [B*T, C, H, W]
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1)
            # Add tubelet dimension and repeat: [B*T, C, 2, H, W]
            c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
            # Encode
            h = target_encoder(c)
            # Reshape back to [B, T, tokens_per_frame, D]
            h = h.view(batch_size, 8, -1, h.size(-1))
            # Flatten tokens: [B, T*tokens_per_frame, D]
            h = h.flatten(1, 2)
            # Normalize
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_predictions(h, actions, states):
        """
        Run predictor in both teacher-forcing and autoregressive modes.
        From official implementation lines 417-437.

        Args:
            h: Target representations [B, T*tokens_per_frame, D]
            actions: [B, T-1, 7]
            states: [B, T, 7]
        Returns:
            z_tf: Teacher-forcing predictions
            z_ar: Autoregressive rollout predictions
        """
        def _step_predictor(_z, _a, _s):
            _z = predictor(_z, _a, _s, extrinsics=None)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        # TEACHER-FORCING: Use all but last frame as context (line 426)
        _z = h[:, :-tokens_per_frame]  # Remove last frame
        _a = actions  # All actions [B, 7, 7]
        _s = states[:, :-1]  # All but last state [B, 7, 7]
        z_tf = _step_predictor(_z, _a, _s)

        # AUTOREGRESSIVE ROLLOUT: Start with first frame + first predicted frame (lines 429-435)
        # Initial context: real first frame + predicted second frame
        _z = torch.cat([
            h[:, :tokens_per_frame],  # Real first frame
            z_tf[:, :tokens_per_frame]  # Predicted second frame
        ], dim=1)

        # Roll out for auto_steps iterations
        for n in range(1, auto_steps):
            _a = actions[:, :n+1]  # Actions up to step n+1
            _s = states[:, :n+1]   # States up to step n+1
            # Predict next frame
            _z_nxt = _step_predictor(_z, _a, _s)[:, -tokens_per_frame:]
            # Append to context
            _z = torch.cat([_z, _z_nxt], dim=1)

        # Return predictions (skip first frame which was ground truth)
        z_ar = _z[:, tokens_per_frame:]

        return z_tf, z_ar

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    sys.stdout.flush()

    # Training loop
    step = 0
    data_iter = iter(dataloader)
    start_time = time.time()

    print("Fetching first batch...")
    sys.stdout.flush()

    while step < max_steps:
        try:
            video_batch, actions, states = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            video_batch, actions, states = next(data_iter)

        # Move to device
        video_batch = video_batch.to(device)
        actions = actions.to(device, dtype=dtype)
        states = states.to(device, dtype=dtype)

        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            # Forward target encoder (frozen)
            h = forward_target(video_batch)

            # Forward predictor (teacher-forcing + autoregressive)
            z_tf, z_ar = forward_predictions(h, actions, states)

            # Compute losses
            tf_loss = loss_fn(z_tf, h)
            ar_loss = loss_fn(z_ar, h)
            loss = tf_loss + ar_loss

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
            print(f"Step {step}/{max_steps} | TF Loss: {tf_loss.item():.4f} | AR Loss: {ar_loss.item():.4f} | Total: {loss.item():.4f} | Time: {elapsed:.1f}s")
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
            path = f'./checkpoints/correct_loss_step{step}.pt'
            torch.save(checkpoint, path)
            print(f"\n✓ Saved checkpoint: {path}")
            sys.stdout.flush()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()

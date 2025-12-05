#!/usr/bin/env python3
"""
Simplified V-JEPA2-AC + LoRA training with dummy data to test the training loop.
Once this works, we can add real DROID data loading.
"""

import sys
sys.path.insert(0, '/workspace/vjepa2')

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
import copy
import time

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit

# Dummy dataset
class DummyDROIDDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy data matching DROID format
        # Video: [T, H, W, C]
        video = torch.randn(8, 256, 256, 3)
        # Actions: [4, 7] (subsampled due to tubelet pooling)
        actions = torch.randn(4, 7)
        # States: [4, 7]
        states = torch.randn(4, 7)
        return video, actions, states


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    print("="*60)
    print("V-JEPA2-AC + LoRA Training (Dummy Data Test)")
    print("="*60)

    # Create dummy dataloader
    dataset = DummyDROIDDataset(num_samples=1000)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(f"✓ Dummy dataloader created ({len(dataset)} samples)")

    # Initialize models
    print("Initializing models...")
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
    max_steps = 50
    step = 0
    data_iter = iter(dataloader)
    start_time = time.time()

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

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save checkpoint
    checkpoint = {
        'step': step,
        'predictor': predictor.state_dict(),
    }
    torch.save(checkpoint, './checkpoints/test_dummy_step50.pt')
    print("✓ Saved checkpoint to ./checkpoints/test_dummy_step50.pt")


if __name__ == '__main__':
    main()

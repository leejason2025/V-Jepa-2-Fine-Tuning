#!/usr/bin/env python3
"""Test script to verify training script can initialize."""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from models import load_vjepa2_ac

def main():
    print("="*70)
    print("Testing Training Script Initialization")
    print("="*70)

    # Load config
    config_path = 'configs/default_config.yaml'
    print(f"\n1. Loading config from {config_path}...")
    config = load_config(config_path)
    print(f"   ✓ Config loaded")

    # Test V-JEPA2-AC loading
    checkpoint_path = 'pretrained_models/vjepa2-ac-vitg.pt'
    print(f"\n2. Loading V-JEPA2-AC from {checkpoint_path}...")

    lora_config = {
        'r': config.lora.r,
        'lora_alpha': config.lora.lora_alpha,
        'lora_dropout': config.lora.lora_dropout,
        'use_rslora': config.lora.use_rslora,
    }

    encoder, predictor = load_vjepa2_ac(
        checkpoint_path=checkpoint_path,
        lora_config=lora_config,
        device='cpu',
        freeze_encoder=config.model.freeze_encoder,
        use_gradient_checkpointing=False,  # Disable for testing
    )

    print(f"\n3. Model loaded successfully!")
    print(f"   Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"   Predictor: {sum(p.numel() for p in predictor.parameters()):,} params")
    print(f"   Trainable: {sum(p.numel() for p in predictor.parameters() if p.requires_grad):,} params")

    # Test forward pass
    print(f"\n4. Testing forward pass...")
    B, C, T, H, W = 1, 3, 16, 256, 256
    dummy_video = torch.randn(B, C, T, H, W)
    T_tubelet = T // 2
    dummy_actions = torch.randn(B, T_tubelet, 7)
    dummy_states = torch.randn(B, T_tubelet, 7)

    with torch.no_grad():
        features = encoder(dummy_video)
        predictions = predictor(features, dummy_actions, dummy_states)

    print(f"   Input: {list(dummy_video.shape)}")
    print(f"   Features: {list(features.shape)}")
    print(f"   Predictions: {list(predictions.shape)}")

    print(f"\n{'='*70}")
    print("✓ All initialization tests passed!")
    print("="*70)

if __name__ == '__main__':
    main()

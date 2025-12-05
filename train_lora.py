#!/usr/bin/env python3
"""
Minimal LoRA fine-tuning for V-JEPA2-AC on DROID

Uses torch.hub to load pretrained V-JEPA2-AC and adds LoRA adapters.
For full training, this would integrate with V-JEPA2's training loop.
"""

import torch
from peft import LoraConfig, get_peft_model

# Load pretrained V-JEPA2-AC via torch.hub
print("Loading V-JEPA2-AC from torch.hub...")
encoder, predictor = torch.hub.load(
    'facebookresearch/vjepa2',
    'vjepa2_ac_vit_giant',
    trust_repo=True
)

# Freeze encoder
for param in encoder.parameters():
    param.requires_grad = False
print("✓ Encoder frozen")

# Add LoRA to predictor
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["qkv", "proj", "fc1", "fc2"],  # Attention + MLP
    lora_dropout=0.05,
    bias="none",
    task_type=None,  # Custom task
    use_rslora=True,
)

predictor_lora = get_peft_model(predictor, lora_config)
predictor_lora.print_trainable_parameters()

print("\n" + "=" * 60)
print("V-JEPA2-AC loaded with LoRA!")
print("=" * 60)
print("\nNext: Integrate with V-JEPA2's training loop from:")
print("  /workspace/vjepa2/app/vjepa_droid/train.py")
print("\nOr use this predictor_lora in your custom training loop.")

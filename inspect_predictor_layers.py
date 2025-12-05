#!/usr/bin/env python3
"""
Inspect predictor architecture to find action/state projection layers.
"""

import sys
sys.path.insert(0, '/workspace/vjepa2')

import torch
import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer as video_vit

def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    print("=" * 80)
    print("Inspecting V-JEPA2-AC Predictor Architecture")
    print("=" * 80)

    # Initialize encoder (needed for embed_dim)
    encoder = video_vit.vit_giant_xformers(
        patch_size=16, num_frames=8, tubelet_size=2,
        uniform_power=True, use_sdpa=True, use_rope=True,
        use_silu=False, checkpoint_activations=True
    )

    # Initialize predictor
    predictor = vit_ac_pred.vit_ac_predictor(
        img_size=256, patch_size=16, num_frames=16, tubelet_size=2,
        embed_dim=encoder.embed_dim, predictor_embed_dim=1024,
        action_embed_dim=7, depth=24, is_frame_causal=True,
        num_heads=16, uniform_power=True, use_rope=True,
        use_sdpa=True, use_silu=False, wide_silu=True,
        use_activation_checkpointing=True
    )

    print("\n=== ALL Linear Layers in Predictor ===\n")

    action_layers = []
    state_layers = []
    embed_layers = []
    transformer_layers = []
    other_layers = []

    for name, module in predictor.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Categorize layers
            if 'action' in name.lower():
                action_layers.append((name, module.weight.shape))
            elif 'state' in name.lower() or 'pose' in name.lower():
                state_layers.append((name, module.weight.shape))
            elif any(x in name for x in ['embed', 'input', 'proj']) and 'blocks' not in name:
                embed_layers.append((name, module.weight.shape))
            elif 'blocks' in name:
                transformer_layers.append((name, module.weight.shape))
            else:
                other_layers.append((name, module.weight.shape))

    # Print categorized layers
    print("\n" + "=" * 80)
    print("ACTION/STATE INPUT PROJECTION LAYERS (TARGET THESE!)")
    print("=" * 80)

    if action_layers:
        print("\n🎯 ACTION-RELATED LAYERS:")
        for name, shape in action_layers:
            print(f"  {name}: {shape}")
    else:
        print("\n⚠️  No layers with 'action' in name found")

    if state_layers:
        print("\n🎯 STATE-RELATED LAYERS:")
        for name, shape in state_layers:
            print(f"  {name}: {shape}")
    else:
        print("\n⚠️  No layers with 'state' or 'pose' in name found")

    print("\n" + "=" * 80)
    print("TOP-LEVEL EMBEDDING/PROJECTION LAYERS")
    print("=" * 80)
    if embed_layers:
        for name, shape in embed_layers:
            print(f"  {name}: {shape}")
    else:
        print("  None found")

    print("\n" + "=" * 80)
    print("TRANSFORMER BLOCK LAYERS (already in config)")
    print("=" * 80)
    # Just show first few as sample
    print(f"  Total transformer layers: {len(transformer_layers)}")
    print(f"  Sample (first 5):")
    for name, shape in transformer_layers[:5]:
        print(f"    {name}: {shape}")

    if other_layers:
        print("\n" + "=" * 80)
        print("OTHER LINEAR LAYERS")
        print("=" * 80)
        for name, shape in other_layers:
            print(f"  {name}: {shape}")

    # Summary recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDED target_modules UPDATE")
    print("=" * 80)

    target_modules = ["qkv", "proj", "fc1", "fc2"]  # Current

    # Add action/state layers if found
    for name, _ in action_layers + state_layers:
        # Extract the module name (last part before the weight)
        parts = name.split('.')
        if len(parts) > 0:
            module_name = parts[-1] if parts[-1] != 'weight' else parts[-2]
            if module_name not in target_modules:
                target_modules.append(module_name)

    print("\ntarget_modules = [")
    for module in target_modules:
        print(f'    "{module}",')
    print("]")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()

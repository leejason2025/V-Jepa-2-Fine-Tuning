"""Load V-JEPA2-AC model with optional LoRA fine-tuning."""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from peft import LoraConfig, get_peft_model


def load_vjepa2_ac(
    checkpoint_path: str,
    lora_config: dict = None,
    device: str = "cuda",
    freeze_encoder: bool = True,
    use_gradient_checkpointing: bool = False,
):
    """Load V-JEPA2-AC model from checkpoint with optional LoRA.

    Args:
        checkpoint_path: Path to vjepa2-ac-vitg.pt checkpoint
        lora_config: LoRA configuration dict (r, alpha, target_modules, etc.)
        device: Device to load model on
        freeze_encoder: Whether to freeze the encoder (recommended)
        use_gradient_checkpointing: Enable activation checkpointing for memory savings

    Returns:
        Tuple of (encoder, predictor) models
    """
    # Add vjepa2 src to path (parent/parent/vjepa2_src from src/models/)
    vjepa2_src_path = str(Path(__file__).parent.parent.parent / 'vjepa2_src')
    if vjepa2_src_path not in sys.path:
        sys.path.insert(0, vjepa2_src_path)

    # Verify path was added correctly
    assert Path(vjepa2_src_path).exists(), f"vjepa2_src path does not exist: {vjepa2_src_path}"
    assert (Path(vjepa2_src_path) / 'vjepa2_models' / 'vision_transformer.py').exists(), f"vision_transformer.py not found"

    # Import V-JEPA2 models after adding to path (renamed to vjepa2_models to avoid conflicts)
    from vjepa2_models.vision_transformer import vit_giant
    from vjepa2_models.ac_predictor import VisionTransformerPredictorAC

    print(f"Loading V-JEPA2-AC from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    # Create encoder (ViT-Giant for the AC model)
    print("\nCreating encoder (ViT-Giant)...")
    encoder = vit_giant(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=True,
        use_silu=False,  # No wide SiLU (no fc3)
        wide_silu=False,  # Confirmed from checkpoint
        use_rope=True,  # AC model uses RoPE instead of absolute pos embeddings
        use_activation_checkpointing=use_gradient_checkpointing,
    )

    # Load encoder weights
    encoder_state_dict = checkpoint['encoder']
    # Remove 'module.' prefix if present (from DataParallel)
    encoder_state_dict = {k.replace('module.', ''): v for k, v in encoder_state_dict.items()}
    encoder.load_state_dict(encoder_state_dict)

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
        print("  ✓ Encoder loaded and frozen")
    else:
        print("  ✓ Encoder loaded (trainable)")

    # Create predictor
    print("\nCreating predictor...")
    predictor = VisionTransformerPredictorAC(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1536,  # ViT-Giant encoder embedding dimension
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        use_silu=False,  # Match encoder
        wide_silu=False,  # Match encoder
        is_frame_causal=True,
        use_rope=True,
        action_embed_dim=7,
        use_extrinsics=False,
        use_activation_checkpointing=use_gradient_checkpointing,
    )

    # Load predictor weights
    predictor_state_dict = checkpoint['predictor']
    # Remove 'module.' prefix
    predictor_state_dict = {k.replace('module.', ''): v for k, v in predictor_state_dict.items()}
    predictor.load_state_dict(predictor_state_dict)

    print(f"  ✓ Predictor loaded")

    # Count parameters
    total_params = sum(p.numel() for p in predictor.parameters())
    print(f"  Total predictor parameters: {total_params:,}")

    # Debug: print module names for LoRA targeting
    if lora_config:
        print(f"\n  Available modules for LoRA targeting:")
        for name, module in predictor.named_modules():
            if 'predictor_blocks' in name and any(x in name for x in ['qkv', 'proj', 'fc1', 'fc2']):
                print(f"    {name}")

    # Apply LoRA if requested
    if lora_config:
        print("\nApplying LoRA to predictor...")

        # Generate target module names for all 24 predictor blocks
        # PEFT doesn't support regex patterns, so we need to list them explicitly
        default_targets = []
        for i in range(24):  # predictor has 24 blocks
            default_targets.extend([
                f"predictor_blocks.{i}.attn.qkv",
                f"predictor_blocks.{i}.attn.proj",
                f"predictor_blocks.{i}.mlp.fc1",
                f"predictor_blocks.{i}.mlp.fc2",
            ])

        # Create LoRA config
        # Note: use_rslora not supported in peft 0.7.0
        peft_config = LoraConfig(
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', default_targets),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
        )

        predictor = get_peft_model(predictor, peft_config)

        trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        print(f"  ✓ LoRA applied")
        print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)

    print(f"\n✓ Models loaded on {device}")

    return encoder, predictor


if __name__ == '__main__':
    # Test loading
    checkpoint_path = 'pretrained_models/vjepa2-ac-vitg.pt'

    # Test without LoRA
    print("="*70)
    print("Test 1: Loading without LoRA")
    print("="*70)
    encoder, predictor = load_vjepa2_ac(
        checkpoint_path,
        lora_config=None,
        device='cpu',
        freeze_encoder=True
    )

    print("\n" + "="*70)
    print("Test 2: Loading with LoRA")
    print("="*70)
    encoder, predictor = load_vjepa2_ac(
        checkpoint_path,
        lora_config={'r': 16, 'lora_alpha': 32, 'use_rslora': True},
        device='cpu',
        freeze_encoder=True
    )

    # Test forward pass
    print("\n" + "="*70)
    print("Test 3: Forward pass")
    print("="*70)

    B, T, H, W, C = 1, 16, 256, 256, 3
    dummy_video = torch.randn(B, C, T, H, W)  # [B, C, T, H, W] for 3D conv

    print(f"Input video shape: {list(dummy_video.shape)}")

    # Encode
    with torch.no_grad():
        features = encoder(dummy_video)
    print(f"Encoder output shape: {list(features.shape)}")

    # Expected: [B, num_patches, embed_dim]
    # num_patches = (T / tubelet_size) * (H / patch_size) * (W / patch_size) = 8 * 16 * 16 = 2048

    # Predict
    # Actions/states should match the tubelet dimension (T / tubelet_size = 16 / 2 = 8)
    T_tubelet = T // 2
    dummy_actions = torch.randn(B, T_tubelet, 7)
    dummy_states = torch.randn(B, T_tubelet, 7)

    with torch.no_grad():
        predictions = predictor(features, dummy_actions, dummy_states)
    print(f"Predictor output shape: {list(predictions.shape)}")
    print(f"Expected: [B={B}, num_patches=2048, embed_dim=1536]")

    print("\n✓ All tests passed!")

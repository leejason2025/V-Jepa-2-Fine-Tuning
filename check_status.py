#!/usr/bin/env python3
"""Check setup status for V-JEPA2-AC fine-tuning."""

import sys
from pathlib import Path

def print_status(name, status, details=""):
    """Print status line."""
    symbol = "✓" if status else "⚠"
    status_text = "READY" if status else "PENDING"
    print(f"{symbol} {name:.<50} {status_text}")
    if details:
        print(f"  {details}")

def main():
    print("="*70)
    print(" V-JEPA2-AC Fine-Tuning Setup Status")
    print("="*70)
    print()

    # Check dependencies
    print("Python Dependencies:")
    try:
        import torch
        print_status("PyTorch", True, f"v{torch.__version__}")
    except:
        print_status("PyTorch", False, "Run: pip install -r requirements.txt")

    try:
        import transformers
        print_status("Transformers", True, f"v{transformers.__version__}")
    except:
        print_status("Transformers", False)

    try:
        import peft
        print_status("PEFT (LoRA)", True, f"v{peft.__version__}")
    except:
        print_status("PEFT (LoRA)", False)

    try:
        import tensorflow
        print_status("TensorFlow (for TFDS)", True, f"v{tensorflow.__version__}")
    except:
        print_status("TensorFlow (for TFDS)", False)

    print()
    print("GCP Access:")

    # Check gsutil
    import subprocess
    try:
        result = subprocess.run(
            ['/root/google-cloud-sdk/bin/gsutil', 'ls', 'gs://gresearch/robotics/droid_100/'],
            capture_output=True,
            timeout=10
        )
        print_status("DROID Dataset Access", result.returncode == 0,
                    "gs://gresearch/robotics/droid_100/ is accessible")
    except:
        print_status("DROID Dataset Access", False,
                    "Install gcloud SDK or check connection")

    print()
    print("Code Components:")

    # Check code modules
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    try:
        from utils.config import load_config
        config = load_config('configs/default_config.yaml')
        print_status("Configuration System", True, "configs/default_config.yaml loaded")
    except Exception as e:
        print_status("Configuration System", False, str(e))

    try:
        from models import create_lora_predictor
        model = create_lora_predictor(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print_status("Model Architecture", True,
                    f"{trainable:,} trainable / {total:,} total params")
    except Exception as e:
        print_status("Model Architecture", False, str(e))

    try:
        from training import create_loss_function
        loss_fn = create_loss_function(config)
        print_status("Loss Functions", True,
                    "Teacher-forcing (T=15) + Rollout (T=2)")
    except Exception as e:
        print_status("Loss Functions", False, str(e))

    print()
    print("Required for Training:")

    # Check for pretrained weights
    encoder_path = config.model.encoder_path
    predictor_path = config.model.predictor_path

    print_status("V-JEPA2 Encoder Weights", encoder_path is not None,
                "Set 'model.encoder_path' in config when available")

    print_status("V-JEPA2-AC Predictor Weights", predictor_path is not None,
                "Set 'model.predictor_path' in config (optional - can train from scratch)")

    print()
    print("="*70)

    if encoder_path is None:
        print()
        print("NEXT STEPS:")
        print("1. Obtain V-JEPA2 encoder pretrained weights")
        print("2. Update configs/default_config.yaml:")
        print("   model:")
        print("     encoder_path: '/path/to/vjepa2_encoder.pt'")
        print("3. (Optional) Set predictor_path if you have pretrained predictor")
        print("4. Run training:")
        print("   python train.py --config configs/default_config.yaml --debug")
        print()
        print("NOTE: Currently the pipeline will use placeholder encoder features")
        print("      for testing purposes. Training will work but won't be meaningful")
        print("      until real encoder weights are provided.")
    else:
        print()
        print("READY TO TRAIN!")
        print("Run: python train.py --config configs/default_config.yaml --debug")

    print("="*70)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Main training script for V-JEPA2-AC LoRA fine-tuning."""

import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config import load_config
from utils.checkpoint import CheckpointManager, create_optimizer, create_scheduler
from data import create_dataloader
from models import load_vjepa2_ac
from training import create_loss_function
from training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='V-JEPA2-AC LoRA Fine-tuning')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--vjepa2-ac-checkpoint',
        type=str,
        default='pretrained_models/vjepa2-ac-vitg.pt',
        help='Path to pretrained V-JEPA2-AC checkpoint (contains both encoder and predictor)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Use debug mode (droid_100 dataset)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to train on'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.debug:
        config.data.debug_mode = True

    # Print configuration
    print("\n" + "="*50)
    print("Configuration:")
    print("="*50)
    print(f"Model: {config.model.num_layers} layers, {config.model.hidden_dim} hidden dim")
    print(f"LoRA: r={config.lora.r}, alpha={config.lora.lora_alpha}, RSLoRA={config.lora.use_rslora}")
    print(f"Data: {config.data.dataset} ({'debug' if config.data.debug_mode else 'full'})")
    print(f"Training: batch_size={config.training.per_device_batch_size}, "
          f"grad_accum={config.training.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.training.per_device_batch_size * config.training.gradient_accumulation_steps}")
    print(f"Max steps: {config.training.max_steps}")
    print(f"Device: {args.device}")
    print("="*50 + "\n")

    # Load V-JEPA2-AC model (encoder + predictor)
    print(f"Loading V-JEPA2-AC from {args.vjepa2_ac_checkpoint}...")

    # Prepare LoRA config
    lora_config = {
        'r': config.lora.r,
        'lora_alpha': config.lora.lora_alpha,
        'lora_dropout': config.lora.lora_dropout,
        'use_rslora': config.lora.use_rslora,
    }

    encoder, predictor = load_vjepa2_ac(
        checkpoint_path=args.vjepa2_ac_checkpoint,
        lora_config=lora_config,
        device=args.device,
        freeze_encoder=config.model.freeze_encoder,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
    )

    print(f"✓ V-JEPA2-AC loaded successfully with LoRA")
    if config.training.gradient_checkpointing:
        print("  ✓ Gradient checkpointing enabled")

    # Create data loaders
    print("Creating data loaders...")
    train_dataloader = create_dataloader(config, split='train', shuffle=True)
    # DROID dataset only has train split, use it for validation too (we'll split it later)
    val_dataloader = create_dataloader(config, split='train', shuffle=False)

    # Note: IterableDataset doesn't have a length
    print(f"✓ Data loaders created (streaming from GCS)")

    # Create loss function
    loss_fn = create_loss_function(config)

    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(config, predictor)

    # Create scheduler
    scheduler = create_scheduler(config, optimizer)

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        local_dir=config.training.checkpoint_dir,
        gcs_bucket=config.gcp.get('checkpoint_bucket'),
        gcs_prefix='vjepa2-checkpoints',
        sync_interval_minutes=30
    )

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_manager.load_checkpoint(
            args.resume,
            predictor,
            optimizer,
            scheduler
        )
    elif checkpoint_manager.get_latest_checkpoint() is not None:
        # Auto-resume from latest checkpoint
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        print(f"Found existing checkpoint: {latest_checkpoint}")
        response = input("Resume from this checkpoint? [y/N]: ")
        if response.lower() == 'y':
            checkpoint_manager.load_checkpoint(
                latest_checkpoint,
                predictor,
                optimizer,
                scheduler
            )

    # Create trainer
    trainer = Trainer(
        config=config,
        predictor=predictor,
        encoder=encoder,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
    )

    # Start training
    try:
        trainer.train()

        # Final GCS sync
        checkpoint_manager.sync_to_gcs(force=True)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        checkpoint_manager.sync_to_gcs(force=True)

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        checkpoint_manager.sync_to_gcs(force=True)
        raise


if __name__ == '__main__':
    main()

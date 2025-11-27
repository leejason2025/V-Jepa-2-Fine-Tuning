"""Checkpoint management and GCS syncing utilities."""

import os
import torch
from pathlib import Path
from typing import Optional, Dict
import subprocess
import time
from datetime import datetime


class CheckpointManager:
    """Manages checkpoints with optional GCS backup."""

    def __init__(
        self,
        local_dir: str,
        gcs_bucket: Optional[str] = None,
        gcs_prefix: Optional[str] = None,
        sync_interval_minutes: int = 30,
    ):
        """Initialize checkpoint manager.

        Args:
            local_dir: Local directory for checkpoints
            gcs_bucket: GCS bucket name for backup (optional)
            gcs_prefix: Prefix path in GCS bucket (optional)
            sync_interval_minutes: Interval for syncing to GCS
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.sync_interval = sync_interval_minutes * 60  # Convert to seconds
        self.last_sync_time = 0

        # Check if GCS sync is enabled and gsutil is available
        self.gcs_enabled = gcs_bucket is not None
        if self.gcs_enabled:
            try:
                subprocess.run(['gsutil', '--version'], capture_output=True, check=True)
                print(f"GCS sync enabled: gs://{gcs_bucket}/{gcs_prefix}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: gsutil not found. GCS sync disabled.")
                self.gcs_enabled = False

    def sync_to_gcs(self, force: bool = False):
        """Sync local checkpoints to GCS.

        Args:
            force: Force sync regardless of time interval
        """
        if not self.gcs_enabled:
            return

        current_time = time.time()
        if not force and (current_time - self.last_sync_time) < self.sync_interval:
            return

        print("Syncing checkpoints to GCS...")

        try:
            # Construct GCS path
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}"

            # Use gsutil rsync for efficient sync
            cmd = [
                'gsutil',
                '-m',  # Multi-threaded
                'rsync',
                '-r',  # Recursive
                '-d',  # Delete extra files in destination
                str(self.local_dir),
                gcs_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Successfully synced to {gcs_path}")
            self.last_sync_time = current_time

        except subprocess.CalledProcessError as e:
            print(f"Error syncing to GCS: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state dict into
            optimizer: Optimizer to load state dict into (optional)
            scheduler: Scheduler to load state dict into (optional)

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint_path = Path(checkpoint_path)

        # Check if path is local or GCS
        if str(checkpoint_path).startswith('gs://'):
            # Download from GCS
            local_path = self.local_dir / checkpoint_path.name
            print(f"Downloading checkpoint from {checkpoint_path}...")

            try:
                subprocess.run(
                    ['gsutil', 'cp', str(checkpoint_path), str(local_path)],
                    check=True,
                    capture_output=True
                )
                checkpoint_path = local_path
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to download checkpoint from GCS: {e}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = sorted(
            self.local_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        if checkpoints:
            return checkpoints[-1]
        return None


def create_optimizer(config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer with optional 8-bit Adam.

    Args:
        config: Configuration object
        model: Model to optimize

    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.training.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            print("Using 8-bit AdamW optimizer")
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        except ImportError:
            print("Warning: bitsandbytes not available. Using standard AdamW.")
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

    return optimizer


def create_scheduler(
    config,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with warmup.

    Args:
        config: Configuration object
        optimizer: Optimizer instance

    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        """Compute learning rate multiplier."""
        warmup_steps = config.training.warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(
                max(1, config.training.max_steps - warmup_steps)
            )
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))).item()

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

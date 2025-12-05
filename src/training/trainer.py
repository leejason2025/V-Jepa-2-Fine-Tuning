"""Main training loop for V-JEPA2-AC fine-tuning."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict
from pathlib import Path
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Trainer:
    """Trainer class for V-JEPA2-AC predictor fine-tuning."""

    def __init__(
        self,
        config,
        predictor: nn.Module,
        encoder: Optional[nn.Module],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            config: Configuration object
            predictor: Predictor model with LoRA
            encoder: V-JEPA2 encoder (frozen)
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.config = config
        self.predictor = predictor
        self.encoder = encoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Move models to device
        self.predictor.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
            self.encoder.eval()  # Frozen encoder

        # Mixed precision setup
        self.use_amp = config.training.mixed_precision in ["fp16", "bf16"]
        if self.use_amp:
            self.scaler = GradScaler() if config.training.mixed_precision == "fp16" else None
            self.dtype = torch.float16 if config.training.mixed_precision == "fp16" else torch.bfloat16
        else:
            self.scaler = None
            self.dtype = torch.float32

        # Gradient accumulation
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Logging and checkpointing
        self.log_every_n_steps = config.training.log_every_n_steps
        self.eval_every_n_steps = config.training.eval_every_n_steps
        self.save_every_n_steps = config.training.save_every_n_steps
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Weights & Biases setup
        self.use_wandb = config.wandb.get('enabled', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.wandb.get('project', 'vjepa2-ac-finetune'),
                entity=config.wandb.get('entity'),
                name=config.wandb.get('run_name'),
                config=vars(config)
            )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames using V-JEPA2 encoder.

        Args:
            frames: [B, T, C, H, W] video frames

        Returns:
            [B, T, patch_H, patch_W, encoder_dim] encoder features
        """
        if self.encoder is None:
            # Placeholder: return random features for testing
            B, T, C, H, W = frames.shape
            patch_size = 16
            encoder_dim = 1408
            patch_H = H // patch_size
            patch_W = W // patch_size
            return torch.randn(B, T, patch_H, patch_W, encoder_dim, device=frames.device)

        # Dataset returns [B, C, T, H, W]
        B, C, T, H, W = frames.shape

        # V-JEPA2 encoder processes video with tubelet_size=2, so we encode the full video
        # Then extract per-frame features from the output
        with torch.no_grad():
            # Encode full video: [B, C, T, H, W] -> [B, num_patches, encoder_dim]
            encoded = self.encoder(frames)  # [B, num_patches_total, D]

            # V-JEPA2 uses tubelet_size=2, so temporal dimension is T//2
            # num_patches_total = (T//2) * (H//patch_size) * (W//patch_size)
            B_cur, num_patches_total, D = encoded.shape

            # Calculate spatial dimensions
            tubelet_size = 2
            T_encoded = T // tubelet_size  # 16 // 2 = 8 temporal tokens
            pH = pW = int((num_patches_total // T_encoded) ** 0.5)

            # Reshape to [B, T_encoded, pH, pW, D]
            features_encoded = encoded.view(B_cur, T_encoded, pH, pW, D)

            # Upsample temporal dimension to match original: [B, T_encoded, pH, pW, D] -> [B, T, pH, pW, D]
            # Each tubelet token represents 2 frames, so we repeat each token twice
            features = features_encoded.repeat_interleave(tubelet_size, dim=1)  # [B, T, pH, pW, D]

        return features

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with frames, actions, states

        Returns:
            Dictionary with loss values
        """
        # Move batch to device
        frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
        actions = batch['actions'].to(self.device)  # [B, T, action_dim]
        states = batch['states'].to(self.device)  # [B, T, state_dim]

        # Encode frames
        # Note: older PyTorch versions don't support device_type parameter
        with autocast(dtype=self.dtype, enabled=self.use_amp):
            encoder_features = self.encode_frames(frames)  # [B, T, pH, pW, D]

            # Compute loss
            loss, loss_dict = self.loss_fn(
                self.predictor,
                encoder_features,
                actions,
                states
            )

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss_dict

    def train_epoch(self):
        """Train for one epoch."""
        self.predictor.train()

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")

        for step, batch in enumerate(progress_bar):
            # Training step
            loss_dict = self.train_step(batch)

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.gradient_clipping > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.predictor.parameters(),
                        self.config.training.gradient_clipping
                    )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Logging
                if self.global_step % self.log_every_n_steps == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': loss_dict['total_loss'],
                        'lr': lr
                    })

                    if self.use_wandb:
                        wandb.log({
                            **loss_dict,
                            'learning_rate': lr,
                            'epoch': self.epoch,
                            'step': self.global_step
                        })

                # Evaluation
                if self.global_step % self.eval_every_n_steps == 0 and self.val_dataloader is not None:
                    val_metrics = self.evaluate()
                    if self.use_wandb:
                        wandb.log({f'val/{k}': v for k, v in val_metrics.items()})

                # Checkpointing
                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint()

            # Max steps check
            if self.global_step >= self.config.training.max_steps:
                break

        self.epoch += 1

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        self.predictor.eval()

        total_loss = 0.0
        total_tf_loss = 0.0
        total_rollout_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            # Move batch to device
            frames = batch['frames'].to(self.device)
            actions = batch['actions'].to(self.device)
            states = batch['states'].to(self.device)

            # Encode frames
            with autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_amp):
                encoder_features = self.encode_frames(frames)

                # Compute loss
                _, loss_dict = self.loss_fn(
                    self.predictor,
                    encoder_features,
                    actions,
                    states
                )

            total_loss += loss_dict['total_loss']
            total_tf_loss += loss_dict['teacher_forcing_loss']
            total_rollout_loss += loss_dict['rollout_loss']
            num_batches += 1

        # Average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'teacher_forcing_loss': total_tf_loss / num_batches,
            'rollout_loss': total_rollout_loss / num_batches,
        }

        self.predictor.train()
        return metrics

    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"

        # Prepare checkpoint
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.config)
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save config
        config_path = self.checkpoint_dir / f"config_step_{self.global_step}.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)

        # Remove old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )

        if len(checkpoints) > self.config.training.save_total_limit:
            for checkpoint in checkpoints[:-self.config.training.save_total_limit]:
                checkpoint.unlink()
                # Also remove corresponding config
                config_file = checkpoint.parent / f"config_step_{checkpoint.stem.split('_')[-1]}.json"
                if config_file.exists():
                    config_file.unlink()

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.max_steps} steps...")

        while self.global_step < self.config.training.max_steps:
            self.train_epoch()

        # Final checkpoint
        self.save_checkpoint()

        print("Training complete!")

        if self.use_wandb:
            wandb.finish()

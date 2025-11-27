"""V-JEPA2 Encoder wrapper for feature extraction."""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel, AutoConfig


class VJEPA2Encoder(nn.Module):
    """Wrapper for V-JEPA2 encoder from HuggingFace.

    Extracts spatial features from video frames for use in predictor training.
    """

    def __init__(
        self,
        model_path: str,
        freeze: bool = True,
        output_patch_features: bool = True,
    ):
        """Initialize V-JEPA2 encoder.

        Args:
            model_path: Path to pretrained V-JEPA2 model
            freeze: If True, freeze encoder weights (recommended for fine-tuning predictor)
            output_patch_features: If True, return patch-level features instead of pooled
        """
        super().__init__()

        print(f"Loading V-JEPA2 encoder from {model_path}...")

        # Load config and model
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        self.output_patch_features = output_patch_features
        self.freeze = freeze

        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("  Encoder weights frozen")

        # Print model info
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Patch size: {self.config.patch_size}")
        print(f"  Image size: {self.config.image_size}")
        print(f"  Num layers: {self.config.num_hidden_layers}")

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features from video frames.

        Args:
            videos: [B, T, C, H, W] video tensor
                   B = batch size
                   T = number of frames
                   C = channels (3 for RGB)
                   H, W = height, width (256x256)

        Returns:
            features: [B, T, patch_H, patch_W, hidden_size]
                     Patch-level features for each frame
        """
        B, T, C, H, W = videos.shape

        # V-JEPA2 model expects [B, T, C, H, W] format (same as our input)
        # The model handles internal permutation to [B, C, T, H, W]

        # Extract features using V-JEPA2
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.model(videos, output_hidden_states=True)

        # Get patch features
        # V-JEPA2 outputs shape: [B, num_tokens, hidden_size]
        # where num_tokens = num_temporal_patches * num_spatial_patches
        hidden_states = outputs.last_hidden_state  # [B, num_tokens, hidden_size]

        # Calculate dimensions
        patch_size = self.config.patch_size
        tubelet_size = self.config.tubelet_size  # Temporal patch size

        patch_H = H // patch_size  # 256 // 16 = 16
        patch_W = W // patch_size  # 256 // 16 = 16
        patch_T = T // tubelet_size  # Temporal patches
        hidden_size = self.config.hidden_size

        # Total spatial-temporal patches
        num_patches = patch_T * patch_H * patch_W

        # Remove CLS token if present (first token)
        if hidden_states.shape[1] == num_patches + 1:
            hidden_states = hidden_states[:, 1:, :]  # [B, num_patches, hidden_size]

        # Reshape to [B, patch_T, patch_H, patch_W, hidden_size]
        features = hidden_states.view(B, patch_T, patch_H, patch_W, hidden_size)

        # Upsample temporal dimension back to T if needed
        # For now, we'll repeat features to match the input temporal dimension
        if patch_T < T:
            # Repeat each temporal patch to match original frame count
            repeat_factor = T // patch_T
            features = features.repeat_interleave(repeat_factor, dim=1)

        # Ensure we have exactly T frames
        features = features[:, :T, :, :, :]

        return features

    @torch.no_grad()
    def extract_features(self, videos: torch.Tensor) -> torch.Tensor:
        """Extract features without gradients (for inference).

        Args:
            videos: [B, T, C, H, W] video tensor

        Returns:
            features: [B, T, patch_H, patch_W, hidden_size]
        """
        was_training = self.training
        self.eval()
        features = self.forward(videos)
        if was_training:
            self.train()
        return features


def load_vjepa2_encoder(
    model_path: str,
    freeze: bool = True,
    device: str = "cuda",
) -> VJEPA2Encoder:
    """Load V-JEPA2 encoder from HuggingFace checkpoint.

    Args:
        model_path: Path to pretrained model directory or HuggingFace model ID
        freeze: Whether to freeze encoder weights
        device: Device to load model on

    Returns:
        VJEPA2Encoder instance
    """
    encoder = VJEPA2Encoder(
        model_path=model_path,
        freeze=freeze,
        output_patch_features=True,
    )

    encoder = encoder.to(device)

    return encoder

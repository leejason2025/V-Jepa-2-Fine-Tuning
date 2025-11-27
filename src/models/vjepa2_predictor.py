"""V-JEPA2-AC Predictor model with LoRA adaptation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from peft import LoraConfig, get_peft_model, PeftModel
import math


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for video patches."""

    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings.

        Args:
            positions: [batch, seq_len, 3] positions (t, h, w)

        Returns:
            cos, sin embeddings for rotary attention
        """
        # positions: [B, N, 3] - (time, height, width)
        freqs = torch.einsum("...i,j->...ij", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


class BlockCausalAttention(nn.Module):
    """Block-causal attention where each patch can attend to current and previous timesteps."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply block-causal attention.

        Args:
            x: [B, N, D] input features
            attention_mask: [B, N, N] attention mask (1 for allowed, 0 for masked)

        Returns:
            [B, N, D] attended features
        """
        B, N, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = BlockCausalAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden = hidden_dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Attention with residual
        x = x + self.attn(self.norm1(x), attention_mask)
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VJEPA2Predictor(nn.Module):
    """V-JEPA2-AC Predictor network (300M parameters).

    Predicts future V-JEPA2 encoder representations given:
    - Past encoder representations (16x16x1408 per frame)
    - Action sequences (7D action vectors)
    - End-effector states (7D state vectors)
    """

    def __init__(
        self,
        encoder_dim: int = 1408,  # V-JEPA2 encoder output dimension
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        action_dim: int = 7,
        state_dim: int = 7,
        patch_size: int = 16,  # Spatial patches per frame
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.patch_size = patch_size
        self.num_patches = patch_size * patch_size  # 16x16 = 256 patches per frame

        # Input projections
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Position embeddings
        self.rope_3d = RotaryPositionEmbedding3D(hidden_dim // num_heads)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output projection back to encoder dimension
        self.output_proj = nn.Linear(hidden_dim, encoder_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def create_block_causal_mask(self, T: int, num_patches: int, device: torch.device) -> torch.Tensor:
        """Create block-causal attention mask.

        Each patch at time t can attend to:
        - All patches at time t
        - All patches, actions, states at times < t

        Args:
            T: Number of timesteps
            num_patches: Number of spatial patches per frame
            device: Torch device

        Returns:
            [seq_len, seq_len] attention mask
        """
        # Sequence: [patches_0, action_0, state_0, patches_1, action_1, state_1, ...]
        tokens_per_timestep = num_patches + 2  # patches + action + state
        seq_len = T * tokens_per_timestep

        mask = torch.zeros(seq_len, seq_len, device=device)

        for t in range(T):
            start_t = t * tokens_per_timestep
            end_t = (t + 1) * tokens_per_timestep

            # Current timestep can attend to itself and all previous timesteps
            mask[start_t:end_t, :end_t] = 1

        return mask

    def forward(
        self,
        encoder_features: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        predict_steps: int = 1,
    ) -> torch.Tensor:
        """Forward pass through predictor.

        Args:
            encoder_features: [B, T, H, W, encoder_dim] encoder representations
            actions: [B, T, action_dim] action sequences
            states: [B, T, state_dim] state sequences
            predict_steps: Number of steps to predict (1 for teacher-forcing, >1 for rollout)

        Returns:
            [B, predict_steps, H, W, encoder_dim] predicted representations
        """
        B, T, H, W, _ = encoder_features.shape
        device = encoder_features.device

        # Flatten spatial dimensions: [B, T, H*W, encoder_dim]
        encoder_flat = encoder_features.view(B, T, H * W, self.encoder_dim)

        # Project inputs to hidden dimension
        encoder_proj = self.encoder_proj(encoder_flat)  # [B, T, H*W, hidden_dim]
        action_proj = self.action_proj(actions).unsqueeze(2)  # [B, T, 1, hidden_dim]
        state_proj = self.state_proj(states).unsqueeze(2)  # [B, T, 1, hidden_dim]

        # Interleave: [patches_0, action_0, state_0, patches_1, action_1, state_1, ...]
        # Shape: [B, T*(H*W + 2), hidden_dim]
        sequence = []
        for t in range(T):
            sequence.append(encoder_proj[:, t])  # [B, H*W, hidden_dim]
            sequence.append(action_proj[:, t])   # [B, 1, hidden_dim]
            sequence.append(state_proj[:, t])    # [B, 1, hidden_dim]
        sequence = torch.cat(sequence, dim=1)  # [B, T*(H*W+2), hidden_dim]

        # Create block-causal attention mask
        attn_mask = self.create_block_causal_mask(T, H * W, device)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, seq_len]

        # Apply transformer blocks
        x = sequence
        for block in self.blocks:
            x = block(x, attn_mask)

        x = self.norm(x)

        # Extract predicted patch features for next timestep(s)
        # For teacher-forcing: predict next frame features
        # For rollout: predict multiple future frames

        predictions = []
        for step in range(predict_steps):
            # Get features at the end of sequence for prediction
            # Extract patch tokens (skip action/state tokens)
            step_idx = (T - 1) * (H * W + 2)  # Start of last timestep
            patch_features = x[:, step_idx:step_idx + H * W]  # [B, H*W, hidden_dim]

            # Project to encoder dimension
            pred = self.output_proj(patch_features)  # [B, H*W, encoder_dim]
            pred = pred.view(B, H, W, self.encoder_dim)  # [B, H, W, encoder_dim]
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=1)  # [B, predict_steps, H, W, encoder_dim]

        return predictions


def create_lora_predictor(config, pretrained_path: Optional[str] = None) -> nn.Module:
    """Create V-JEPA2 Predictor with LoRA adaptation.

    Args:
        config: Configuration object
        pretrained_path: Path to pretrained predictor checkpoint

    Returns:
        Predictor model with LoRA adapters
    """
    # Create base predictor model
    predictor = VJEPA2Predictor(
        encoder_dim=1408,  # V-JEPA2 encoder dimension
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        action_dim=7,
        state_dim=7,
        patch_size=16,
    )

    # Load pretrained weights if provided
    if pretrained_path:
        print(f"Loading pretrained predictor from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        predictor.load_state_dict(checkpoint['model_state_dict'])

    # Apply LoRA if enabled
    if config.lora.enabled:
        print("Applying LoRA adaptation...")

        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            use_rslora=config.lora.use_rslora,
        )

        # Apply LoRA to model
        predictor = get_peft_model(predictor, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in predictor.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return predictor

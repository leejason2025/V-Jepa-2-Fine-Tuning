"""Configuration utilities for loading and validating config files."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_rslora: bool = True
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    bias: str = "none"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    predictor_path: Optional[str] = None
    encoder_path: Optional[str] = None
    freeze_encoder: bool = True
    num_layers: int = 24
    num_heads: int = 16
    hidden_dim: int = 1024
    activation: str = "gelu"


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset: str = "droid"
    bucket_name: str = "gresearch"
    bucket_prefix: str = "robotics"
    debug_mode: bool = False
    video_resolution: int = 256
    fps: int = 4
    clip_length_sec: int = 4
    frames_per_clip: int = 16
    camera_view: str = "wrist"
    filter_idle_actions: bool = True
    idle_threshold: float = 0.01  # Threshold for filtering idle actions
    clip_stride: int = 8  # Stride for sliding window clips
    shuffle_buffer_size: int = 100000
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 250
    max_steps: int = 2000
    teacher_forcing_steps: int = 15
    rollout_steps: int = 2
    loss_weight_tf: float = 1.0
    loss_weight_rollout: float = 1.0
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    use_8bit_adam: bool = True
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 250
    save_every_n_steps: int = 250
    save_total_limit: int = 5
    checkpoint_dir: str = "./checkpoints"
    ddp_backend: str = "nccl"
    gradient_clipping: float = 1.0


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    training: TrainingConfig
    wandb: Dict[str, Any] = field(default_factory=dict)
    gcp: Dict[str, Any] = field(default_factory=dict)
    eval: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with all parameters
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Parse nested configs
    model_config = ModelConfig(**yaml_config.get('model', {}))
    lora_config = LoRAConfig(**yaml_config.get('lora', {}))
    data_config = DataConfig(**yaml_config.get('data', {}))
    training_config = TrainingConfig(**yaml_config.get('training', {}))

    return Config(
        model=model_config,
        lora=lora_config,
        data=data_config,
        training=training_config,
        wandb=yaml_config.get('wandb', {}),
        gcp=yaml_config.get('gcp', {}),
        eval=yaml_config.get('eval', {})
    )


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file.

    Args:
        config: Config object to save
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'model': vars(config.model),
        'lora': vars(config.lora),
        'data': vars(config.data),
        'training': vars(config.training),
        'wandb': config.wandb,
        'gcp': config.gcp,
        'eval': config.eval
    }

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

# V-JEPA2-AC LoRA Fine-Tuning

Fine-tune the V-JEPA2-AC world model predictor on the DROID robotics dataset using LoRA (Low-Rank Adaptation) for parameter-efficient training.

## Project Overview

This project aims to discover if more training data yields positive effects on V-JEPA2-AC in order to determine if data is a bottleneck for world model robot planning.

### Key Features

- **LoRA Fine-tuning**: Efficient adaptation of the 300M parameter V-JEPA2-AC predictor
- **DROID Dataset**: Large-scale robot manipulation dataset (76k+ episodes)
- **GCS Streaming**: Efficient data loading directly from Google Cloud Storage
- **Memory Optimized**: Designed for 24GB VRAM GPUs with gradient checkpointing, mixed precision, and 8-bit Adam
- **Teacher-Forcing + Rollout Loss**: Combined training objective for robust world modeling

## Architecture

### V-JEPA2-AC Predictor (300M parameters)
- 24-layer transformer with 16 attention heads
- 1024 hidden dimensions
- Block-causal attention pattern
- 3D rotary position embeddings
- Predicts future V-JEPA2 encoder representations given actions and states

### LoRA Configuration
- Rank: 16-32 (recommended)
- Alpha: 32-64
- Target modules: Q, K, V projections and MLP layers
- Rank-Stabilized LoRA (RSLoRA) enabled

### Training Loss
Combined loss with equal weighting:
- **Teacher-forcing** (T=15): Predict next frame features autoregressively
- **Rollout** (T=2): Multi-step prediction with feedback

Both losses use L1 distance between predicted and target representations.

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training)
- 24GB+ VRAM GPU (recommended)
- Google Cloud SDK (for GCS access)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/leejason2025/V-Jepa-2-Fine-Tuning.git
cd V-Jepa-2-Fine-Tuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate with Google Cloud (for DROID dataset access):
```bash
gcloud auth application-default login
```

4. Set up GCP project (optional, for checkpoint backup):
```bash
export GCP_PROJECT="your-project-id"
export GCS_CHECKPOINT_BUCKET="your-checkpoint-bucket"
```

## Configuration

Edit [configs/default_config.yaml](configs/default_config.yaml) to customize training parameters:

### Key Parameters

```yaml
model:
  predictor_path: null  # Set when you have pretrained weights
  encoder_path: null    # Set when you have V-JEPA2 encoder

lora:
  r: 16                 # LoRA rank (16-32 recommended)
  lora_alpha: 32        # Scaling factor
  use_rslora: true      # Rank-stabilized LoRA

data:
  debug_mode: false     # Set true to use droid_100 (100 episodes)
  camera_view: "wrist"  # "wrist", "exterior_1", "exterior_2"

training:
  per_device_batch_size: 2          # Fits in 24GB VRAM
  gradient_accumulation_steps: 16   # Effective batch = 32
  learning_rate: 1.0e-4
  max_steps: 100000

  # Memory optimization
  gradient_checkpointing: true
  mixed_precision: "bf16"  # or "fp16"
  use_8bit_adam: true
```

## Usage

### Basic Training

```bash
python train.py --config configs/default_config.yaml
```

### Debug Mode (100 episodes)

```bash
python train.py --config configs/default_config.yaml --debug
```

### Resume from Checkpoint

```bash
python train.py --config configs/default_config.yaml --resume checkpoints/checkpoint_step_10000.pt
```

### With Pretrained Weights

```bash
python train.py \
  --config configs/default_config.yaml \
  --predictor-checkpoint /path/to/predictor.pt \
  --encoder-checkpoint /path/to/encoder.pt
```

## Data

### DROID Dataset

The DROID dataset contains 76,000+ robot manipulation episodes in RLDS format.

- **Full dataset**: `gs://gresearch/robotics/droid` (1.7TB)
- **Debug dataset**: `gs://gresearch/robotics/droid_100` (2GB, 100 episodes)

### Data Format

Each episode contains:
- **Video frames**: 256×256 RGB at 4 fps
- **Actions**: 7D end-effector deltas (position + orientation + gripper)
- **States**: 7D end-effector states
- **Metadata**: Episode ID, agent ID, environment info

### Preprocessing

- Frames are encoded using V-JEPA2 encoder to 16×16×1408 feature maps
- Actions and states are projected to predictor hidden dimension
- Idle actions are filtered based on magnitude thresholds

## Training Details

### Memory Usage (24GB VRAM)

- Batch size: 2 per device
- Gradient accumulation: 16 steps
- Effective batch size: 32
- Mixed precision: BF16
- Gradient checkpointing: Enabled
- 8-bit AdamW: Enabled

### Expected Performance

- Training time: ~3-5 days for 100k steps on A100
- LoRA trainable parameters: ~20M (6.7% of 300M)
- Checkpoint size: ~80MB (LoRA adapters only)

### Checkpointing

- Checkpoints saved every 1000 steps
- Keep last 5 checkpoints
- Auto-sync to GCS every 30 minutes
- Includes: model state, optimizer state, scheduler state, config

## Monitoring

### Weights & Biases

Enable W&B logging in config:

```yaml
wandb:
  enabled: true
  project: "vjepa2-ac-finetune"
  entity: "your-username"
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

## Project Structure

```
V-Jepa-2-Fine-Tuning/
├── configs/
│   └── default_config.yaml      # Training configuration
├── src/
│   ├── data/
│   │   ├── droid_dataset.py     # DROID dataset loader
│   │   └── __init__.py
│   ├── models/
│   │   ├── vjepa2_predictor.py  # Predictor architecture
│   │   └── __init__.py
│   ├── training/
│   │   ├── losses.py            # Training losses
│   │   ├── trainer.py           # Training loop
│   │   └── __init__.py
│   └── utils/
│       ├── config.py            # Configuration utilities
│       ├── checkpoint.py        # Checkpoint management
│       └── __init__.py
├── train.py                      # Main training script
├── requirements.txt              # Python dependencies
└── README.md
```

## Troubleshooting

### Out of Memory

1. Reduce `per_device_batch_size` to 1
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable gradient checkpointing (should already be enabled)
4. Use FP16 instead of BF16 if supported
5. Reduce LoRA rank to 8

### GCS Access Issues

1. Ensure you're authenticated:
   ```bash
   gcloud auth application-default login
   ```

2. Check bucket permissions:
   ```bash
   gsutil ls gs://gresearch/robotics/
   ```

3. Use debug mode to test with smaller dataset:
   ```bash
   python train.py --debug
   ```

### Data Loading Slow

1. Increase `num_workers` in config (up to CPU cores)
2. Increase `prefetch_factor` to 4-8
3. Check network bandwidth to GCS
4. Use a VM in the same GCP region as the data

### V-JEPA2 Encoder Not Available

The current implementation uses a placeholder encoder that returns random features. To use the actual V-JEPA2 encoder:

1. Obtain V-JEPA2 pretrained weights
2. Implement encoder loading in `src/data/droid_dataset.py`
3. Pass encoder checkpoint via `--encoder-checkpoint`

## Citation

If you use this code, please cite:

```bibtex
@misc{vjepa2ac-finetune,
  author = {Your Name},
  title = {V-JEPA2-AC LoRA Fine-tuning on DROID},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/leejason2025/V-Jepa-2-Fine-Tuning}
}
```

## References

- [V-JEPA2-AC Paper](https://arxiv.org/abs/2412.09424)
- [DROID Dataset](https://droid-dataset.github.io/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## TODO

- [ ] Implement actual V-JEPA2 encoder loading
- [ ] Add RLDS format parsing for DROID episodes
- [ ] Implement idle action filtering logic
- [ ] Add evaluation metrics (prediction accuracy, FVD)
- [ ] Support multi-GPU training with DDP
- [ ] Add visualization tools for predictions
- [ ] Implement downstream task evaluation (robot control)

# V-JEPA2-AC Fine-Tuning with LoRA

Fine-tune V-JEPA2-AC (Action-Conditioned world model) on the DROID robotics dataset using LoRA for parameter-efficient training.

## Quick Start

### 1. Download Pretrained Model
```bash
# Download V-JEPA2-AC checkpoint (11GB)
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt -P pretrained_models/
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training

**Debug (50 steps on droid_100):**
```bash
python train.py --config configs/debug_config.yaml
```

**Production (1250 steps on full DROID):**
```bash
python train.py --config configs/default_config.yaml
```

## Data Streaming

**No download required!** Data streams directly from Google Cloud Storage:
- **Source**: `gs://gresearch/robotics/droid/` (public bucket)
- **Format**: TFRecord files in RLDS format
- **Access**: Anonymous, no authentication needed
- **Debug dataset**: `droid_100` (100 episodes) for quick testing
- **Full dataset**: `droid` (thousands of episodes) for production

The dataloader ([src/data/droid_dataset.py](src/data/droid_dataset.py)) uses TensorFlow's GCS support to stream episodes on-the-fly without local storage.

## What We Built

### Architecture
- **Encoder**: V-JEPA2 ViT-Giant (1.01B params, frozen)
- **Predictor**: 300M params with LoRA adapters
- **LoRA**: 6.3M trainable params (r=16, α=32) on attention and MLP layers
- **Training**: Autoregressive feature prediction conditioned on actions and states

### Key Implementations

1. **DROID Dataset Loader** ([src/data/droid_dataset.py](src/data/droid_dataset.py))
   - Parses flattened RLDS TFRecords with VarLenFeature
   - Streams 16-frame clips at 4fps from GCS
   - Handles wrist camera RGB + 7-DOF actions (position + orientation + gripper)
   - Configurable shuffle buffer for efficient randomization

2. **Encoder Integration** ([src/training/trainer.py](src/training/trainer.py))
   - Processes full 16-frame videos with tubelet_size=2 (Conv3d)
   - Outputs 8 temporal tokens, upsampled to 16 via repeat-interleave
   - Generates per-frame spatial feature maps (16×16 patches)

3. **Training Pipeline** ([train.py](train.py))
   - Mixed precision (BF16) with gradient accumulation (effective batch=32)
   - Gradient checkpointing for memory efficiency
   - Saves checkpoints every 250 steps with LoRA adapters

### Fixes Applied
- ✅ TFRecord parsing for RLDS flattened format
- ✅ Video tensor handling (B, C, T, H, W)
- ✅ Tubelet compression/decompression
- ✅ Feature flattening for predictor input
- ✅ PyTorch version compatibility (autocast)

## Output

Checkpoints saved to `./checkpoints/` containing:
- LoRA adapter weights (6.3M params)
- Optimizer state
- Training step and config

Use these for energy landscape analysis or downstream tasks.

## Config Files

- [configs/debug_config.yaml](configs/debug_config.yaml): Fast validation (50 steps, droid_100, minimal shuffling)
- [configs/default_config.yaml](configs/default_config.yaml): Production (1250 steps, full DROID, proper training)

## Requirements

- PyTorch 2.0+
- TensorFlow 2.x (for TFRecord/GCS)
- CUDA GPU (24GB+ VRAM recommended)
- Internet connection for GCS streaming

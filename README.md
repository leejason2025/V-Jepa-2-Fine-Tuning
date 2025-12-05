# V-JEPA2-AC Fine-Tuning with LoRA

Fine-tune V-JEPA2-AC (Action-Conditioned world model) on the DROID robotics dataset using LoRA for parameter-efficient training.

## Project Structure

```
.
├── configs/                  # Training configurations
│   ├── debug_config.yaml    # Quick validation (50 steps, droid_100)
│   └── default_config.yaml  # Production (1250 steps, full DROID)
├── src/
│   ├── models/
│   │   └── load_vjepa2_ac.py  # LoRA integration for V-JEPA2-AC
│   └── utils/
│       ├── config.py          # Config loading utilities
│       └── checkpoint.py      # Checkpoint management
├── vjepa2_src/              # V-JEPA2 source code (from official repo)
├── pretrained_models/       # Pretrained V-JEPA2-AC checkpoint
├── train.py                 # Main training script (LoRA-enabled)
└── README.md
```

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

**Debug (quick validation):**
```bash
python train.py --config configs/debug_config.yaml
```

**Production (1250 steps):**
```bash
python train.py --config configs/default_config.yaml
```

## What This Does

### LoRA Fine-Tuning
- **Encoder**: V-JEPA2 ViT-Giant (1.01B params) - **frozen**
- **Predictor**: 300M params - **fine-tuned with LoRA**
- **LoRA Adapters**: 6.3M trainable params (2.06% of predictor)
  - r=16, α=32 (Rank-Stabilized LoRA)
  - Applied to attention (qkv, proj) and MLP layers

### Training Details
- **Input**: 8 frames @ 4fps (2 seconds) from DROID wrist camera
- **Resolution**: 256×256
- **Actions**: 7-DOF (3D position + 3D Euler + gripper state)
- **Batch size**: 32 effective (2 per device × 16 gradient accumulation)
- **Precision**: BF16 mixed precision
- **Checkpoints**: Saved every 250 steps

### Key Discovery: Dimension Matching

The pretrained V-JEPA2-AC expects exactly **2064 tokens**:
- 8 frames → tubelet_size=2 → 4 temporal tokens → upsampled to 8
- 8 temporal × 256 spatial patches (16×16 grid) = 2048 tokens
- Plus 16 action/state conditioning tokens = **2064 total**

We initially used 16 frames (4128 tokens) which caused RoPE dimension mismatch. Fixed by matching V-JEPA2's configuration.

## Data Streaming

**No download required!** Data streams directly from Google Cloud Storage:
- **Source**: `gs://gresearch/robotics/droid/` (public bucket)
- **Format**: TFRecord files in RLDS format
- **Access**: Anonymous, no authentication needed
- **Debug dataset**: `droid_100` (100 episodes)
- **Full dataset**: `droid` (thousands of episodes)

Note: While we stream from GCS in our implementation, the official V-JEPA2 repo uses downloaded MP4 files with H5 trajectory data.

## What We Learned (Avoiding Redundant Work)

Initially we built custom implementations of:
- ❌ DROID dataloader (V-JEPA2 already provides: `vjepa2/app/vjepa_droid/droid.py`)
- ❌ TFRecord parsing (V-JEPA2 has correct implementation)
- ❌ Training loop (V-JEPA2 provides: `vjepa2/app/vjepa_droid/train.py`)
- ❌ Encoder/predictor wrappers (V-JEPA2 provides via torch.hub)

**What we kept (unique to LoRA approach):**
- ✅ LoRA integration (`src/models/load_vjepa2_ac.py`)
- ✅ Config management (`src/utils/`)
- ✅ Training configs with 8-frame setup (`configs/`)

## Official V-JEPA2 Resources

- **Repo**: https://github.com/facebookresearch/vjepa2
- **Paper**: https://arxiv.org/abs/2506.09985
- **Models**: https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6
- **DROID Training Config**: [configs/train/vitg16/droid-256px-8f.yaml](https://github.com/facebookresearch/vjepa2/blob/main/configs/train/vitg16/droid-256px-8f.yaml)

## Output

Checkpoints saved to `./checkpoints/` containing:
- LoRA adapter weights (6.3M params)
- Optimizer state
- Training configuration

Use these for:
- Energy landscape analysis
- Robot policy learning
- Transfer to new manipulation tasks

## Requirements

- Python 3.10+ (V-JEPA2 official repo requires 3.11+, but LoRA approach works with 3.10)
- PyTorch 2.0+
- TensorFlow 2.x (for TFRecord/GCS streaming)
- CUDA GPU (24GB+ VRAM recommended)
- Internet connection for GCS streaming

## Training Progress

**Completed:**
- ✅ Fixed dimension mismatch (16→8 frames)
- ✅ LoRA integration working
- ✅ Debug training validated (loss=1.22 at step 25)
- ✅ Codebase cleaned up (removed redundant implementations)

**Next:**
- [ ] Run full 1250-step training on complete DROID dataset
- [ ] Generate 2-3 checkpoints for energy landscape analysis
- [ ] Test energy landscape visualization

## Citation

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and others},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```

# V-JEPA2-AC Fine-Tuning with LoRA

Fine-tune V-JEPA2-AC (Action-Conditioned world model) on the DROID robotics dataset using LoRA for parameter-efficient training.

## Project Structure

```
.
├── configs/                  # Training configurations
│   ├── debug_config.yaml    # Quick validation (50 steps, droid_100)
│   └── default_config.yaml  # Production (1250 steps, full DROID)
├── droid_src/               # Source code for LoRA fine-tuning
│   └── models/
│       └── load_vjepa2_ac.py  # LoRA integration for V-JEPA2-AC
├── vjepa2_src/              # V-JEPA2 source code (not included due to ownership)
├── train_lora.py            # Main training script (LoRA-enabled)
├── v-jepa2_energy_landscape_analysis.ipynb  # Colab notebook for energy landscape analysis
└── README.md
```

## Resources Used

To fine-tune V-JEPA2-AC, you will need the following external resources:

### 1. V-JEPA2 Training Code
Clone the official V-JEPA2 repository for the training infrastructure:
- **Repo**: https://github.com/facebookresearch/vjepa2

### 2. Pretrained Model Weights
Download the open-source weights from Hugging Face:
- **V-JEPA2-AC-ViT-G**: Action-conditioned model weights
- **V-JEPA2 Base Encoder**: Base ViT-Giant encoder model

### 3. DROID Dataset
The DROID robotics dataset is required for training but not included, also used by official V-Jepa2 which also include their DROID data training. 
- **Source**: https://droid-dataset.github.io/


### LoRA Fine-Tuning
- **Encoder**: V-JEPA2 ViT-Giant (~1.1B params) - **frozen**
- **Predictor**: ~305M params - **fine-tuned with LoRA**
- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha (α): 32
  - Dropout: 0.05
  - RSLoRA (rank-stabilized): True
  - Target modules: attn.qkv, attn.proj, mlp.fc1, mlp.fc2
  - Bias: None
  - Checkpoint: LoRA_step_223.pt (step 223)

### Training Details
- **Input**: 16 frames @ 4fps (4 seconds) from DROID wrist camera
- **Resolution**: 256×256
- **Actions**: 7-DOF (3D position + 3D Euler + gripper state)
- **Batch size**: 32 effective (2 per device × 16 gradient accumulation)
- **Precision**: BF16 mixed precision
- **Checkpoints**: Saved every 250 steps

## Output

Checkpoints saved to `./checkpoints/` containing:
- LoRA adapter weights (6.3M params)
- Optimizer state
- Training configuration

Use these for:
- Energy landscape analysis
- Robot policy learning
- Transfer to new manipulation tasks

## LoRA Weights for Energy Analysis

The trained LoRA weights used for energy landscape analysis are available at:
- **Weights**: https://github.com/leejason2025/V-Jepa2-LoRa-Weights

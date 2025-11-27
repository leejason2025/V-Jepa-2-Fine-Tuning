# Quick Start Guide

Get up and running with V-JEPA2-AC fine-tuning in minutes.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU with 24GB+ VRAM
- Google Cloud account with access to DROID dataset

## 5-Minute Setup

### 1. Clone and Install

```bash
git clone https://github.com/leejason2025/V-Jepa-2-Fine-Tuning.git
cd V-Jepa-2-Fine-Tuning
pip install -r requirements.txt
```

### 2. GCP Authentication

```bash
# Run setup script
bash scripts/setup_gcp.sh

# Or manually:
gcloud auth application-default login
gsutil ls gs://gresearch/robotics/droid
```

### 3. Test Dataset Access

```bash
python scripts/test_dataset.py
```

This will verify you can access the DROID dataset and load batches correctly.

### 4. Start Training (Debug Mode)

```bash
# Train on 100 episodes for testing
python train.py --config configs/default_config.yaml --debug
```

### 5. Full Training

```bash
# Train on full DROID dataset
python train.py --config configs/default_config.yaml
```

## Next Steps

### Monitor Training

Enable Weights & Biases logging by editing `configs/default_config.yaml`:

```yaml
wandb:
  enabled: true
  project: "vjepa2-ac-finetune"
  entity: "your-username"
```

### Adjust for Your GPU

If you have less than 24GB VRAM, edit `configs/default_config.yaml`:

```yaml
training:
  per_device_batch_size: 1  # Reduce from 2
  gradient_accumulation_steps: 32  # Increase to maintain effective batch
  mixed_precision: "fp16"  # May save more memory than bf16
```

### Resume from Checkpoint

```bash
python train.py --config configs/default_config.yaml --resume checkpoints/checkpoint_step_10000.pt
```

## Common Issues

### Out of Memory
- Reduce `per_device_batch_size` to 1
- Enable gradient checkpointing (should be on by default)
- Use FP16 instead of BF16

### Can't Access DROID Dataset
- Run `gcloud auth application-default login`
- Verify with `gsutil ls gs://gresearch/robotics/droid`
- Try debug mode: `python train.py --debug`

### Slow Data Loading
- Increase `num_workers` in config (up to your CPU cores)
- Use a VM in the same GCP region as the data
- Check network bandwidth

## What's Next?

Once training is complete, you'll have:

1. **LoRA adapters**: Lightweight adapters (~80MB) in `checkpoints/`
2. **Training logs**: In `logs/` (view with TensorBoard)
3. **Checkpoints**: Saved every 1000 steps, synced to GCS

See [README.md](README.md) for complete documentation.

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review [configs/default_config.yaml](configs/default_config.yaml) for all options
- Open an issue on GitHub

## Training Overview

### What You're Training

- **Model**: V-JEPA2-AC Predictor (300M parameters)
- **Method**: LoRA fine-tuning (only ~20M trainable params)
- **Data**: DROID robot manipulation dataset (76k+ episodes)
- **Goal**: Predict future visual representations given actions

### Expected Timeline

| GPU | Batch Size | Time to 100k steps |
|-----|------------|-------------------|
| A100 (40GB) | 4 | ~2-3 days |
| A100 (24GB) | 2 | ~3-5 days |
| RTX 4090 | 1-2 | ~4-7 days |

### Memory Usage

- Base model: ~1.2GB
- LoRA adapters: ~80MB
- Optimizer states: ~2GB
- Activations (batch=2): ~18GB
- **Total**: ~22GB (fits in 24GB)

### Checkpointing

Checkpoints are saved:
- Every 1000 steps
- On training interruption (Ctrl+C)
- On error/crash
- Synced to GCS every 30 minutes

### Monitoring Metrics

Key metrics to watch:
- `total_loss`: Combined teacher-forcing + rollout loss
- `teacher_forcing_loss`: Single-step prediction accuracy
- `rollout_loss`: Multi-step prediction accuracy
- `learning_rate`: Should warmup then cosine decay

## Architecture Summary

```
Input:
  ├── Video frames [B, T, 3, 256, 256]
  ├── Actions [B, T, 7] (end-effector deltas)
  └── States [B, T, 7] (end-effector pose)

V-JEPA2 Encoder (frozen):
  └── Frame features [B, T, 16, 16, 1408]

Predictor (with LoRA):
  ├── Input projection
  ├── 24 Transformer layers
  │   ├── Block-causal attention (LoRA on Q,K,V)
  │   └── MLP (LoRA on fc1, fc2)
  └── Output projection

Output:
  └── Predicted features [B, predict_steps, 16, 16, 1408]

Loss:
  ├── Teacher-forcing (T=15 steps)
  └── Rollout (T=2 steps)
```

## File Structure After Training

```
V-Jepa-2-Fine-Tuning/
├── checkpoints/
│   ├── checkpoint_step_1000.pt
│   ├── checkpoint_step_2000.pt
│   └── ...
├── logs/
│   └── events.out.tfevents.*
└── wandb/
    └── run-*
```

## Tips for Best Results

1. **Start with debug mode** to verify everything works
2. **Monitor first 1000 steps** - losses should decrease
3. **Use wandb** for better visualization
4. **Keep checkpoints backed up** to GCS
5. **Validate on downstream tasks** after training

Good luck with your training!

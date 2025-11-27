# Implementation Notes

This document provides technical details about the V-JEPA2-AC LoRA fine-tuning implementation.

## Implementation Status

### ✅ Completed Components

1. **Project Structure**
   - Clean modular organization
   - Separation of concerns (data, models, training, utils)
   - Configuration-driven design

2. **Configuration System** ([src/utils/config.py](src/utils/config.py))
   - YAML-based configuration
   - Type-safe dataclasses
   - Easy to extend and modify

3. **Model Architecture** ([src/models/vjepa2_predictor.py](src/models/vjepa2_predictor.py))
   - V-JEPA2-AC Predictor (300M params)
   - Block-causal attention mechanism
   - 3D rotary position embeddings
   - LoRA integration via PEFT library

4. **Training Losses** ([src/training/losses.py](src/training/losses.py))
   - Teacher-forcing loss (T=15 steps)
   - Rollout loss (T=2 steps)
   - L1 distance metrics

5. **Training Loop** ([src/training/trainer.py](src/training/trainer.py))
   - Mixed precision training (FP16/BF16)
   - Gradient accumulation
   - Gradient checkpointing support
   - Automatic checkpointing and logging
   - Weights & Biases integration

6. **Checkpoint Management** ([src/utils/checkpoint.py](src/utils/checkpoint.py))
   - Local checkpoint saving
   - GCS sync with gsutil
   - Automatic cleanup of old checkpoints
   - 8-bit AdamW optimizer support

7. **Helper Scripts**
   - GCP setup script
   - Dataset testing script
   - Main training entry point

8. **Documentation**
   - Comprehensive README
   - Quick start guide
   - Implementation notes

### ⚠️ Placeholder Components (TODO)

These components are implemented with placeholders and need actual implementations when resources are available:

1. **V-JEPA2 Encoder Loading**
   - Location: [src/training/trainer.py:100](src/training/trainer.py#L100)
   - Current: Returns random features
   - Needed: Load actual V-JEPA2 encoder weights
   - Action: Implement encoder loading when weights available

2. **DROID Dataset RLDS Parsing**
   - Location: [src/data/droid_dataset.py:95](src/data/droid_dataset.py#L95)
   - Current: Simplified parser, returns dummy data
   - Needed: Full RLDS format parsing for DROID
   - Action: Parse actual episode structure, extract frames/actions/states

3. **Idle Action Filtering**
   - Location: [src/data/droid_dataset.py:111](src/data/droid_dataset.py#L111)
   - Current: Accepts all episodes
   - Needed: Filter based on action magnitudes
   - Action: Implement filtering logic based on action norms

## Architecture Details

### V-JEPA2-AC Predictor

```python
Input Format:
  - encoder_features: [B, T, 16, 16, 1408]  # V-JEPA2 encoded frames
  - actions: [B, T, 7]                       # End-effector deltas
  - states: [B, T, 7]                        # End-effector poses

Architecture:
  1. Input Projection (Linear layers)
     - encoder_features: 1408 → 1024
     - actions: 7 → 1024
     - states: 7 → 1024

  2. Sequence Construction
     - Interleave: [patches_t0, action_t0, state_t0, patches_t1, ...]
     - Shape: [B, T*(256+2), 1024] for 16x16 patches

  3. Block-Causal Attention
     - Each timestep attends to current + all previous
     - Prevents information leakage from future

  4. 24 Transformer Blocks
     - Multi-head attention (16 heads)
     - MLP with GELU activation
     - LayerNorm before each sub-layer
     - Residual connections

  5. Output Projection
     - 1024 → 1408 (back to encoder dimension)
     - Reshape to [B, predict_steps, 16, 16, 1408]

Output:
  - predicted_features: [B, predict_steps, 16, 16, 1408]
```

### LoRA Adaptation

LoRA is applied to these modules in each transformer block:
- `q_proj`: Query projection in attention
- `k_proj`: Key projection in attention
- `v_proj`: Value projection in attention
- `o_proj`: Output projection in attention
- `mlp.fc1`: First MLP layer
- `mlp.fc2`: Second MLP layer

Parameters:
- Rank (r): 16 (configurable 8-32)
- Alpha: 32 (configurable 16-64)
- Dropout: 0.05
- Rank-Stabilized: Yes (scaling = alpha/sqrt(r))

Trainable parameters: ~20M out of 300M (6.7%)

### Training Losses

#### Teacher-Forcing Loss

```python
for t in range(T-1):
    # Input: frames 0 to t
    predicted = predictor(features[:, :t+1], actions[:, :t+1], states[:, :t+1])

    # Target: frame t+1
    target = features[:, t+1]

    # L1 loss
    loss_t = ||predicted - target||_1

loss_tf = mean(loss_t for t in range(15))
```

#### Rollout Loss

```python
# Start from first frame
current = features[:, 0]

# Rollout for T steps
for step in range(2):
    predicted = predictor(current, actions[:, step], states[:, step])
    current = cat([current, predicted], dim=1)

# Compare final prediction to actual
target = features[:, 2]
loss_rollout = ||current[:, -1] - target||_1
```

## Memory Optimization Techniques

### 1. LoRA Fine-Tuning
- Only 6.7% of parameters trainable
- Reduces optimizer states significantly
- Smaller gradient buffers

### 2. Gradient Checkpointing
- Trades compute for memory
- Recomputes activations during backward pass
- ~50% memory reduction for ~20% speed decrease

### 3. Mixed Precision (BF16)
- Halves memory for activations
- Maintains numerical stability better than FP16
- No loss scaling needed

### 4. 8-bit AdamW
- Reduces optimizer state memory
- Uses bitsandbytes library
- Minimal impact on convergence

### 5. Gradient Accumulation
- Simulates larger batch sizes
- Accumulate gradients over multiple micro-batches
- Update once per N steps

### Memory Breakdown (24GB GPU)

```
Component                Memory
─────────────────────────────────
Model parameters         ~1.2 GB
LoRA adapters           ~0.08 GB
Optimizer states        ~2.0 GB
Activations (batch=2)   ~18 GB
CUDA context            ~1 GB
Buffer/overhead         ~2 GB
─────────────────────────────────
Total                   ~24.3 GB
```

## Data Pipeline

### DROID Dataset Structure

```
gs://gresearch/robotics/droid/
├── train/
│   ├── episode_0000.tfrecord
│   ├── episode_0001.tfrecord
│   └── ...
└── val/
    └── ...

Each episode contains:
- observations:
  - wrist_rgb: [T, 256, 256, 3]
  - exterior_rgb_1: [T, 256, 256, 3]
  - exterior_rgb_2: [T, 256, 256, 3]
- actions: [T, 7]  # delta end-effector pose
- states: [T, 7]   # absolute end-effector pose
- metadata: episode_id, agent_id, etc.
```

### Streaming Pipeline

1. **GCS Streaming**: Use TensorFlow Datasets to stream from GCS
2. **Shuffle**: Large buffer (100k) for randomization
3. **Filter**: Remove idle actions (optional)
4. **Batch**: Collate into batches
5. **Prefetch**: Pipeline next batches during training

### Data Processing

```python
1. Load episode from GCS (TFRecord)
2. Extract video clip (16 frames @ 4fps = 4 seconds)
3. Resize frames to 256x256
4. Normalize to [-1, 1]
5. Extract corresponding actions and states
6. Encode frames with V-JEPA2 (frozen)
7. Return: {features, actions, states}
```

## Configuration System

The configuration is hierarchical and type-safe:

```yaml
Config
├── ModelConfig
│   ├── predictor_path
│   ├── encoder_path
│   ├── num_layers
│   └── ...
├── LoRAConfig
│   ├── r
│   ├── lora_alpha
│   ├── target_modules
│   └── ...
├── DataConfig
│   ├── bucket_name
│   ├── camera_view
│   └── ...
└── TrainingConfig
    ├── batch_size
    ├── learning_rate
    └── ...
```

All configs are loaded from YAML and converted to Python dataclasses for type safety.

## Checkpoint Format

Checkpoints include:

```python
{
    'global_step': int,
    'epoch': int,
    'model_state_dict': OrderedDict,  # LoRA adapters + base model
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,  # Optional
    'scaler_state_dict': OrderedDict,     # Optional (for FP16)
    'config': dict
}
```

Size: ~80MB for LoRA adapters only, ~1.3GB with full model

## GCS Integration

### Checkpoint Syncing

```python
# Syncs checkpoints to GCS every 30 minutes
gsutil -m rsync -r -d ./checkpoints gs://bucket/vjepa2-checkpoints

# -m: multi-threaded
# -r: recursive
# -d: delete extra files in destination
```

### Data Streaming

```python
# TensorFlow Datasets automatically handles:
- Parallel downloads
- Caching
- Prefetching
- Retry logic
```

## Performance Optimizations

### DataLoader Settings

```yaml
num_workers: 4              # Parallel data loading
prefetch_factor: 2          # Batches to prefetch per worker
pin_memory: true            # Faster GPU transfer
```

### Training Settings

```yaml
gradient_accumulation_steps: 16  # Larger effective batch
gradient_clipping: 1.0           # Prevent exploding gradients
warmup_steps: 500                # LR warmup
max_steps: 100000                # Total training steps
```

### Logging Settings

```yaml
log_every_n_steps: 10       # Frequent logging
eval_every_n_steps: 500     # Periodic validation
save_every_n_steps: 1000    # Checkpoint frequency
```

## Testing the Implementation

### 1. Test Configuration Loading

```bash
python -c "from src.utils.config import load_config; config = load_config('configs/default_config.yaml'); print('Config loaded:', config.model.num_layers, 'layers')"
```

### 2. Test Model Creation

```bash
python -c "from src.utils.config import load_config; from src.models import create_lora_predictor; config = load_config('configs/default_config.yaml'); model = create_lora_predictor(config); print('Model created with', sum(p.numel() for p in model.parameters()), 'total params')"
```

### 3. Test Data Loading

```bash
python scripts/test_dataset.py
```

### 4. Test Training (Dry Run)

```bash
# This will fail at actual DROID loading but tests the pipeline
python train.py --config configs/default_config.yaml --debug
```

## Future Enhancements

### Short-term (When resources available)

1. **Implement V-JEPA2 Encoder Loading**
   - Load pretrained weights
   - Freeze encoder
   - Extract features efficiently

2. **Complete DROID Parsing**
   - Full RLDS format parsing
   - Extract videos, actions, states
   - Implement idle filtering

3. **Add Multi-GPU Support**
   - DistributedDataParallel
   - Gradient synchronization
   - Multi-node training

### Medium-term

1. **Evaluation Metrics**
   - FVD (Fréchet Video Distance)
   - LPIPS (perceptual similarity)
   - Action prediction accuracy

2. **Visualization Tools**
   - Predicted vs actual frames
   - Attention maps
   - Rollout trajectories

3. **Downstream Tasks**
   - Robot control evaluation
   - Planning benchmarks
   - Transfer learning tests

### Long-term

1. **Model Improvements**
   - Larger predictor variants
   - Different LoRA configurations
   - Alternative attention mechanisms

2. **Data Augmentation**
   - Temporal augmentation
   - Spatial augmentation
   - Action noise injection

3. **Scaling Studies**
   - Data scaling laws
   - Model size vs performance
   - Compute-optimal training

## Key Design Decisions

### Why LoRA?

- **Efficiency**: Only 6.7% parameters to train
- **Speed**: Faster training, less memory
- **Storage**: Small checkpoint files
- **Flexibility**: Easy to swap adapters

### Why Block-Causal Attention?

- **Temporal modeling**: Respects causality
- **Efficiency**: Reduces computation
- **Stability**: Prevents information leakage

### Why Teacher-Forcing + Rollout?

- **Balance**: Single-step accuracy + multi-step coherence
- **Stability**: Teacher-forcing provides stable gradients
- **Robustness**: Rollout handles error accumulation

### Why BF16?

- **Range**: Better than FP16 for large values
- **Stability**: No loss scaling needed
- **Speed**: Faster than FP32 on modern GPUs

## Common Issues and Solutions

### Issue: CUDA Out of Memory

Solutions:
1. Reduce `per_device_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing
4. Use FP16 instead of BF16
5. Reduce LoRA rank

### Issue: Slow Data Loading

Solutions:
1. Increase `num_workers` (up to CPU cores)
2. Increase `prefetch_factor`
3. Use VM in same GCP region
4. Check network bandwidth

### Issue: NaN Loss

Solutions:
1. Reduce learning rate
2. Enable gradient clipping
3. Check for numerical instability
4. Use BF16 instead of FP16

### Issue: Not Converging

Solutions:
1. Increase warmup steps
2. Adjust learning rate
3. Check data quality
4. Verify loss implementation

## References

- V-JEPA2-AC: https://arxiv.org/abs/2412.09424
- LoRA: https://arxiv.org/abs/2106.09685
- DROID: https://droid-dataset.github.io/
- PEFT: https://github.com/huggingface/peft
- Rotary Embeddings: https://arxiv.org/abs/2104.09864
- Rank-Stabilized LoRA: https://arxiv.org/abs/2312.03732

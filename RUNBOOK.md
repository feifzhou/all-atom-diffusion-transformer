# DiT-S Identity VAE Training Runbook

Experiment: All-atom Diffusion Transformer (DiT-S) with Identity VAE on QM9 dataset

## Changes from Original Repository

This fork (starting from commit `b9ce505f`) adds AMD GPU support and QM9-only training infrastructure. Summary of changes:

### Infrastructure Changes
1. **AMD GPU compatibility** (`9e6d5c9`): Added `torch_scatter` fallback for AMD GPUs lacking native scatter operations
2. **Optional dependencies** (`d8417ad`): Made crystal/MOF dependencies (CifFile, PyXtal, QMOF) optional for QM9-only workflows
3. **QM9-only training scripts** (`cdfdf55`, `1561dfd`): Added standalone training scripts with auto-resume for VAE and DiT
4. **CSV logging** (`configs/logger/csv.yaml`): Added CSV logger for local metric tracking without W&B

### Training Improvements
5. **Auto-resume logic**: All training scripts detect and load latest checkpoint automatically
6. **Checkpoint optimization** (`da82261`): 
   - Decoupled checkpoint saving from validation metrics (`monitor=null`)
   - Keep all epoch checkpoints (`save_top_k=-1`)
   - Save after training epochs, not validation (`save_on_train_epoch_end=True`)
7. **Flux scheduler integration**: Job submission scripts for LLNL Lassen cluster using Flux (not SLURM)

### Model Additions
8. **Identity VAE** (`d547ae8`): No-compression autoencoder for baseline DiT experiments
   - `src/models/identity_vae_module.py`: Identity encoder/decoder (latent = concat(one_hot, pos))
   - `scripts/make_identity_vae_ckpt.py`: Generate identity VAE checkpoint without training
   - `configs/autoencoder_module/identity_vae.yaml`: Config for identity VAE

### Training Configuration
9. **Cosine LR scheduler** (`05d54fc`): Added to reduce loss spikes observed with flat LR
   - `diffusion_module.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR`
   - `T_max=1200`, `eta_min=1e-5`
10. **Bug fixes** (`05d54fc`):
    - Added missing PoseBusters metrics (`no_radicals`, `non-aromatic_ring_non-flatness`) to fix validation crash
    - Fixed checkpoint auto-resume path

### Documentation
11. **RUNBOOK.md** (this file): Complete experiment documentation with hyperparameters, bug fixes, and reproducibility notes
12. **ARCHITECTURE_NOTES.md** (`ec49b77`): Analysis of position embedding issue in transformer VAE
13. **extract_val_metrics.sh**: Script to parse validation metrics from flux stdout

### File Statistics
- 28 files changed: +1523 insertions, -15 deletions
- 10 new scripts added (training, testing, utilities)
- 3 new model files (identity VAE, scatter fallback)
- 3 documentation files added

**Original repository**: https://github.com/facebookresearch/all-atom-diffusion-transformer (last upstream commit: `b9ce505f` - "Fix edge case for absurd MOFs", 2024)

---

## Experiment Overview

**Goal**: Establish baseline DiT training pipeline with no-compression identity VAE before scaling up to learned VAE.

**Model**: DiT-S (smallest variant)
- 12 layers, d_model=384, 6 heads
- ~33M trainable parameters
- Identity latent: d_x=8 (5-element one-hot + 3 coords, no compression)

**Dataset**: QM9 only (100k training molecules)

**Hardware**: AMD GPUs on LLNL Lassen cluster, 4 GPUs per job

## Key Hyperparameters

Matching the paper (arxiv:2410.01712) with modifications:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 → 1e-5 | Cosine annealing over 1200 epochs (added to reduce loss spikes) |
| Batch size | 256 | Paper value |
| Weight decay | 0.0 | Paper value |
| EMA decay | 0.9999 | Paper value |
| Gradient clip | 1.0 norm | Default |
| Validation frequency | Every 50 epochs | ~1x per hour |
| Checkpoint frequency | Every 30 epochs | ~2x per hour, keep 2 latest |
| Validation samples | 100 | Reduced from 1000 for speed |

**Deviations from paper**: 
1. Added cosine LR scheduler (paper used flat 1e-4) to stabilize training and reduce loss spikes observed at constant LR
2. Keep only 2 latest epoch checkpoints (`save_top_k=2`) to save disk space

## Training Script

Location: `/usr/WS2/zhou6/all-atom-diffusion-transformer/train_dit_identity_4gpu.sh`

```bash
#!/bin/bash
# DiT-S training with identity VAE on 4 GPUs - QM9 only with DDP
# Run with: flux batch -q pdebug -t 1h -N 1 -n 4 -g 1 train_dit_identity_4gpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
export WORK_DIR=/p/lustre5/zhou6/ADIT
cd $PROJECT_ROOT

echo "=== Starting DiT-S training with identity VAE (4 GPUs) ==="
echo "Time: $(date)"

# Identity autoencoder checkpoint
VAE_CHECKPOINT="$PROJECT_ROOT/checkpoints/identity_vae_qm9_std.ckpt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: Identity VAE checkpoint not found"
    exit 1
fi

# Auto-resume: find latest DiT checkpoint
CKPT_ARG=""
LATEST_CKPT="$WORK_DIR/logs/train_diffusion/checkpoints/last.ckpt"
if [ -f "$LATEST_CKPT" ]; then
    echo "Found DiT checkpoint: $LATEST_CKPT"
    echo "Resuming from checkpoint..."
    CKPT_ARG="ckpt_path=$LATEST_CKPT"
else
    echo "No DiT checkpoint found, starting from scratch"
fi

# DiT-S configuration: 12 layers, d_model=384, 6 heads (~33M params)
# LR: cosine annealing from 1e-4 to 1e-5 over 1200 epochs
python src/train_diffusion.py \
    data=qm9_only \
    callbacks=diffusion_qm9_only \
    trainer=ddp \
    trainer.devices=4 \
    logger=csv \
    test=False \
    name="dit_s_identity_qm9" \
    trainer.max_epochs=2000 \
    trainer.check_val_every_n_epoch=50 \
    paths.output_dir=$WORK_DIR/logs/train_diffusion \
    paths.log_dir=$WORK_DIR/logs/train_diffusion \
    diffusion_module.autoencoder_ckpt=$VAE_CHECKPOINT \
    diffusion_module.autoencoder_cls=src.models.identity_vae_module.IdentityAutoencoderLitModule \
    diffusion_module.sampling.num_samples=100 \
    diffusion_module.denoiser.d_x=8 \
    diffusion_module.denoiser.num_layers=12 \
    diffusion_module.denoiser.d_model=384 \
    diffusion_module.denoiser.nhead=6 \
    +diffusion_module.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR \
    +diffusion_module.scheduler.T_max=1200 \
    +diffusion_module.scheduler.eta_min=1e-5 \
    $CKPT_ARG

echo "=== Training completed ==="
echo "Time: $(date)"
```

## Checkpoint Configuration

Modified `configs/callbacks/diffusion_qm9_only.yaml`:

```yaml
model_checkpoint:
  dirpath: ${paths.output_dir}/checkpoints
  filename: "ldm-epoch@{epoch}-step@{step}"
  monitor: null  # Don't require validation metrics for saving
  mode: "max"
  save_last: True
  save_top_k: 2  # Keep only 2 latest epoch checkpoints to save disk space
  every_n_epochs: 30  # Save every 30 epochs (~30 min)
  save_on_train_epoch_end: True  # Save after training, not validation
```

**Key changes**:
- `save_top_k=2`: Keep only 2 latest epoch checkpoints (was -1 to keep all, then changed to save disk space)
- `monitor=null`: Decouple checkpoint saving from validation metrics
- `save_on_train_epoch_end=True`: Save after training epochs, not validation

## Bug Fixes Applied

### 1. Missing PoseBusters Metrics (Fixed)

**Problem**: Validation crashed with `KeyError: 'no_radicals'`

**Root cause**: PoseBusters evaluator returns 12 metrics, but only 10 were registered in `ldm_module.py`

**Fix**: Added missing metrics to QM9 validation dict:
```python
"no_radicals": MeanMetric(),
"non-aromatic_ring_non-flatness": MeanMetric(),
```

### 2. Checkpoint Auto-Resume Path (Fixed)

**Problem**: Auto-resume looked in wrong directory pattern

**Fix**: Changed from glob pattern to fixed path:
```bash
LATEST_CKPT="$WORK_DIR/logs/train_diffusion/checkpoints/last.ckpt"
```

### 3. Hydra Config Error for LR Scheduler (Fixed)

**Problem**: `diffusion_module.scheduler._target_=...` failed because scheduler is `null` by default

**Error**: `Key '_target_' is not in struct` - can't override fields in null config

**Fix**: Use `+` prefix to append/create config instead of override:
```bash
+diffusion_module.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR
+diffusion_module.scheduler.T_max=1200
+diffusion_module.scheduler.eta_min=1e-5
```

## Job Submission

### Single 1-hour Job
```bash
flux batch -q pdebug -t 1h -N 1 -n 4 -g 1 train_dit_identity_4gpu.sh
```

### Chained Jobs (12 hours)
```bash
PREV_JOB=$(flux batch -q pdebug -t 1h -N 1 -n 4 -g 1 train_dit_identity_4gpu.sh)
for i in {2..12}; do
  JOB_ID=$(flux batch -q pdebug -t 1h -N 1 -n 4 -g 1 --dependency=afterany:${PREV_JOB} train_dit_identity_4gpu.sh)
  echo "Job $i: $JOB_ID"
  PREV_JOB=$JOB_ID
done
```

**Note**: `-n 4 -g 1` means 4 tasks with 1 GPU each = 4 GPUs total. This was 3% faster than `-n 1 -g 4` in throughput tests.

## Training Performance

- **Training speed**: ~50 epochs/hour (~72 seconds/epoch)
- **Validation time**: ~2 minutes (100 samples)
- **Checkpoint save time**: ~30 seconds (498 MB per checkpoint)
- **Expected epochs per 1h job**: ~50 epochs

## File Locations

### Checkpoints
```
/p/lustre5/zhou6/ADIT/logs/train_diffusion/checkpoints/
├── last.ckpt                           # Latest checkpoint (always current)
├── ldm-epoch@119-step@11640.ckpt     # Epoch 119
├── ldm-epoch@149-step@14550.ckpt     # Epoch 149
├── ldm-epoch@179-step@17460.ckpt     # Epoch 179
└── ...                                 # Every 30 epochs
```

### Training Metrics (CSV)
```
/p/lustre5/zhou6/ADIT/logs/train_diffusion/csv_logs/
├── version_0/metrics.csv
├── version_1/metrics.csv
└── ...
```

**Note**: One version per training run. Training loss logged every epoch.

### Validation Metrics (Flux Output)
```
flux-<jobid>.out
```

**Note**: Validation metrics only printed to stdout (not properly logged to CSV). Use `extract_val_metrics.sh` to parse.

### Generated Structures (PDB)
```
/p/lustre5/zhou6/ADIT/logs/train_diffusion/train_diffusion/runs/
└── dit_s_identity_qm9_<timestamp>/
    ├── qm9_val_0/
    │   ├── molecule_0.pdb
    │   ├── molecule_1.pdb
    │   └── ...
    ├── qm9_val_1/
    ├── qm9_val_2/
    └── qm9_val_3/
```

**Note**: 100 molecules per validation run, split across 4 GPU ranks (25 per rank).

## Extracting Validation Metrics

Use the provided script:
```bash
./extract_val_metrics.sh flux-*.out > validation_metrics.csv
```

This extracts `valid_rate` from flux output files where validation actually logged correct values.

## Current Training Status

**As of 2026-06-22:**
- Training started from scratch at epoch 0
- Reached epoch 410 after 6 jobs (no LR scheduler)
- Restarted with LR scheduler at epoch 410
- Current chain: 12 jobs running with cosine annealing LR, expected final epoch ~1010
- Checkpoint strategy: Keep 2 latest + last.ckpt (saves ~1 GB per 60 epochs)

**Validation metrics trajectory:**
```
Epoch 149: valid_rate = 0.25%
Epoch 211: valid_rate = 0.0%
Epoch 311: valid_rate = 28.5%
Epoch 361: valid_rate = 46.8%
Epoch 411: valid_rate = 60.2%
```

**Training loss trajectory:**
- Epoch 0: ~2.6
- Epoch 100: ~0.8
- Epoch 200: ~0.5
- Epoch 400: ~0.4

Loss shows spiky behavior (factor of 2-3x above average at peaks), which motivated adding cosine LR scheduler.

## Next Steps

1. Complete current 12-job chain to ~1010 epochs
2. Evaluate generation quality at checkpoints: 500, 750, 1000 epochs
3. If quality plateaus before 1000 epochs, stop early
4. If quality still improving, extend to 1200 epochs
5. Switch to learned VAE (Equiformer-based) for compression experiments

## Reproducibility Notes

- Random seed: 9 (set in config)
- DDP with 4 GPUs, sync batchnorm enabled
- Flux scheduler on LLNL Lassen (pdebug queue, 1h time limit)
- All paths relative to `/p/lustre5/zhou6/ADIT/` for persistence

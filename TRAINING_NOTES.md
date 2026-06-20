# QM9-only Training Setup

## Overview
Training All-atom Diffusion Transformer (ADiT) on QM9 dataset only for molecule generation.
Target: 30-50% validity, not full convergence.

## Hardware
- 4 GPUs with 90GB RAM each
- Expected time: ~2 days total (VAE + DiT)

## Dataset
- QM9: 130,831 molecules
- Auto-downloaded on first run

## Model Configuration
- **VAE**: latent_dim=8, KL_weight=1e-5
- **DiT-S**: 12 layers, d_model=384, 6 heads (~30M params)
- Using smaller DiT-S instead of DiT-B for faster training

## Training Steps

### Step 1: Test VAE (Single GPU, 1 hour)
```bash
cd /usr/WS2/zhou6/all-atom-diffusion-transformer
flux batch -q pdebug -t 1h -N 1 -n 1 -g 1 test_vae_singlegpu.sh
```
Check: `flux jobs` and logs in `logs/train_autoencoder/`

### Step 2: Full VAE Training (4 GPUs, ~18 hours)
```bash
flux batch -q pbatch -t 24h -N 1 -n 4 -g 4 train_vae_4gpu.sh
```
Monitor: validation `match_rate` should reach >0.9 for good reconstruction

### Step 3: DiT Training (4 GPUs, ~36 hours)
First, edit `train_dit_4gpu.sh` and set VAE_CHECKPOINT path from step 2.
Then:
```bash
flux batch -q pbatch -t 24h -N 1 -n 4 -g 4 train_dit_4gpu.sh
```
Monitor: validation `valid_rate` for molecule validity (target: 0.3-0.5)

## Monitoring Progress
```bash
# Check running jobs
flux jobs

# View live logs (replace JOBID)
flux job attach JOBID

# Check output
tail -f logs/train_autoencoder/runs/LATEST_RUN/*.out
```

## Checkpoints
- VAE: `logs/train_autoencoder/runs/DATE_TIME/checkpoints/`
- DiT: `logs/train_diffusion/runs/DATE_TIME/checkpoints/`
- Saved every epoch with top-3 kept

## Parallel Training
Using PyTorch DDP (DistributedDataParallel) via Lightning.
If DDP fails on Flux, fall back to single GPU (change `trainer=gpu` and `trainer.devices=1`).

## Early Stopping
Manually stop if:
- VAE: match_rate > 0.9
- DiT: valid_rate > 0.3 (30% validity target met)

Use: `flux cancel JOBID`

## Configuration Notes
- W&B logging: DISABLED (logger=null)
- Validation: every epoch for quick feedback
- Data dir: auto-created in `data/`
- Output: `logs/train_{autoencoder,diffusion}/runs/`

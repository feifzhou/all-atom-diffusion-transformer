#!/bin/bash

# DiT-S training on 4 GPUs - QM9 only with DDP
# Run with: flux batch -q pbatch -t 24h -N 1 -n 4 -g 4 train_dit_4gpu.sh
# IMPORTANT: Set VAE_CHECKPOINT path below before running!

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
cd $PROJECT_ROOT

# Activate conda env
source ~/.bashrc
conda activate myenv || conda activate base

# SET THIS: Path to trained VAE checkpoint from previous step
VAE_CHECKPOINT="logs/train_autoencoder/runs/YYYY-MM-DD_HH-MM-SS/checkpoints/last.ckpt"

if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: VAE checkpoint not found at: $VAE_CHECKPOINT"
    echo "Please set VAE_CHECKPOINT path in this script"
    exit 1
fi

echo "=== Starting DiT-S training (4 GPUs) ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "VAE checkpoint: $VAE_CHECKPOINT"
echo "GPUs available:"
nvidia-smi -L

# DiT-S configuration: 12 layers, d_model=384, 6 heads (~30M params)
python src/train_diffusion.py \
    data=qm9_only \
    callbacks=diffusion_qm9_only \
    trainer=ddp \
    trainer.devices=4 \
    logger=csv \
    name="dit_s_qm9_latent8" \
    trainer.max_epochs=2000 \
    trainer.check_val_every_n_epoch=1 \
    ++diffusion_module.autoencoder_ckpt=$VAE_CHECKPOINT \
    ++diffusion_module.denoiser.d_x=8 \
    ++diffusion_module.denoiser.num_layers=12 \
    ++diffusion_module.denoiser.d_model=384 \
    ++diffusion_module.denoiser.nhead=6 \
    ++callbacks.model_checkpoint.every_n_epochs=1 \
    ++callbacks.model_checkpoint.save_top_k=3

echo "=== Training completed ==="
echo "Time: $(date)"
echo "Check for checkpoint in: logs/train_diffusion/runs/"

#!/bin/bash

# Test DiT training on single GPU - QM9 only, short run
# Run with: flux batch -q pdebug -t 30m -N 1 -n 1 -g 1 test_dit_singlegpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
cd $PROJECT_ROOT

echo "=== Starting DiT test training ==="
echo "Time: $(date)"

# Auto-detect latest VAE checkpoint
VAE_CHECKPOINT=$(find logs/train_autoencoder/runs/vae_qm9_4gpu_scaled_*/checkpoints/last.ckpt 2>/dev/null | sort -r | head -1)
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: No VAE checkpoint found"
    exit 1
fi
echo "Using VAE checkpoint: $VAE_CHECKPOINT"

# Auto-resume: find latest DiT checkpoint if available
CKPT_ARG=""
LATEST_CKPT=$(find logs/train_diffusion/runs/test_dit_qm9_*/checkpoints/last.ckpt 2>/dev/null | sort -r | head -1)
if [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    echo "Resuming from checkpoint..."
    CKPT_ARG="ckpt_path=$LATEST_CKPT"
else
    echo "No checkpoint found, starting from scratch"
fi

# Run training - single GPU, QM9 only, short test
python src/train_diffusion.py \
    data=qm9_only \
    callbacks=diffusion_qm9_only \
    trainer=gpu \
    logger=csv \
    test=False \
    name="test_dit_qm9" \
    trainer.max_epochs=3 \
    trainer.devices=1 \
    diffusion_module.autoencoder_ckpt=$VAE_CHECKPOINT \
    diffusion_module.denoiser.d_x=8 \
    diffusion_module.denoiser.num_layers=12 \
    diffusion_module.denoiser.d_model=384 \
    diffusion_module.denoiser.nhead=6 \
    $CKPT_ARG

echo "=== Training completed ==="
echo "Time: $(date)"

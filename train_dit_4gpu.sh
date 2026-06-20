#!/bin/bash

# DiT-S training on 4 GPUs - QM9 only with DDP
# Run with: flux batch -q pbatch -t 12h -N 1 -n 4 -g 1 train_dit_4gpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
cd $PROJECT_ROOT

echo "=== Starting DiT-S training (4 GPUs) ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "GPUs available:"
rocm-smi --showid || echo "rocm-smi failed"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"

# Auto-detect latest VAE checkpoint
VAE_CHECKPOINT=$(find logs/train_autoencoder/runs/vae_qm9_4gpu_scaled_*/checkpoints/last.ckpt 2>/dev/null | sort -r | head -1)
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: No VAE checkpoint found"
    echo "Expected pattern: logs/train_autoencoder/runs/vae_qm9_4gpu_scaled_*/checkpoints/last.ckpt"
    exit 1
fi
echo "Using VAE checkpoint: $VAE_CHECKPOINT"

# Auto-resume: find latest DiT checkpoint if available
CKPT_ARG=""
LATEST_CKPT=$(find logs/train_diffusion/runs/dit_s_qm9_latent8_*/checkpoints/last.ckpt 2>/dev/null | sort -r | head -1)
if [ -f "$LATEST_CKPT" ]; then
    echo "Found DiT checkpoint: $LATEST_CKPT"
    echo "Resuming from checkpoint..."
    CKPT_ARG="ckpt_path=$LATEST_CKPT"
else
    echo "No DiT checkpoint found, starting from scratch"
fi

# DiT-S configuration: 12 layers, d_model=384, 6 heads (~30M params)
python src/train_diffusion.py \
    data=qm9_only \
    callbacks=diffusion_qm9_only \
    trainer=ddp \
    trainer.devices=4 \
    logger=csv \
    test=False \
    name="dit_s_qm9_latent8" \
    trainer.max_epochs=2000 \
    trainer.check_val_every_n_epoch=1 \
    diffusion_module.autoencoder_ckpt=$VAE_CHECKPOINT \
    diffusion_module.denoiser.d_x=8 \
    diffusion_module.denoiser.num_layers=12 \
    diffusion_module.denoiser.d_model=384 \
    diffusion_module.denoiser.nhead=6 \
    callbacks.model_checkpoint.every_n_epochs=1 \
    callbacks.model_checkpoint.save_top_k=3 \
    $CKPT_ARG

echo "=== Training completed ==="
echo "Time: $(date)"
echo "Check for checkpoint in: logs/train_diffusion/runs/"

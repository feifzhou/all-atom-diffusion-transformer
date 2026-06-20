#!/bin/bash

# VAE training on 4 GPUs - QM9 only with DDP
# Run with: flux batch -q pbatch -t 24h -N 1 -n 4 -g 4 train_vae_4gpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
cd $PROJECT_ROOT

echo "=== Starting VAE training (4 GPUs) ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "GPUs available:"
rocm-smi --showid || echo "rocm-smi failed"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"

# Auto-resume: find latest checkpoint if available
CKPT_ARG=""
LATEST_CKPT=$(find logs/train_autoencoder/runs/vae_qm9_4gpu_scaled_*/checkpoints/last.ckpt 2>/dev/null | sort -r | head -1)
if [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    echo "Resuming from checkpoint..."
    CKPT_ARG="ckpt_path=$LATEST_CKPT"
else
    echo "No checkpoint found, starting from scratch"
fi

# Run training - 4 GPUs with DDP, QM9 only
# Properly scaled: batch_size=1024 (256 per GPU), LR=0.0004 (4x base)
python src/train_autoencoder.py \
    data=qm9_only \
    callbacks=autoencoder_qm9_only \
    trainer=ddp \
    logger=csv \
    test=False \
    name="vae_qm9_4gpu_scaled" \
    trainer.max_epochs=5000 \
    trainer.devices=4 \
    data.datamodule.batch_size.train=1024 \
    autoencoder_module.latent_dim=8 \
    autoencoder_module.optimizer.lr=0.0004 \
    autoencoder_module.loss_weights.loss_kl.qm9=0.00001 \
    $CKPT_ARG

echo "=== Training completed ==="
echo "Time: $(date)"
echo "Check for checkpoint in: logs/train_autoencoder/runs/"

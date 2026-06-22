#!/bin/bash

# DiT-B training with identity VAE on 4 GPUs - QM9 only with DDP
# Identity VAE: no compression, latent = concat(one_hot, pos)
# Run with: flux batch -q pdebug -t 1h -N 1 -n 1 -g 4 train_dit_identity_4gpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
export WORK_DIR=/p/lustre5/zhou6/ADIT
cd $PROJECT_ROOT

echo "=== Starting DiT-S training with identity VAE (4 GPUs) ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "GPUs available:"
rocm-smi --showid || echo "rocm-smi failed"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}')"

# Identity autoencoder checkpoint
VAE_CHECKPOINT="$PROJECT_ROOT/checkpoints/identity_vae_qm9_std.ckpt"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: Identity VAE checkpoint not found at $VAE_CHECKPOINT"
    exit 1
fi
echo "Using identity VAE checkpoint: $VAE_CHECKPOINT"

# Auto-resume: find latest DiT checkpoint if available
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
# Identity latent: d_x=8 (5 element one-hot + 3 coords)
# Checkpoint every 30 epochs (~15 min), validate every 50 epochs (~25 min)
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
echo "Check for checkpoint in: $WORK_DIR/logs/train_diffusion/runs/"

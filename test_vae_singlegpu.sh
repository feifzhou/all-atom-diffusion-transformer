#!/bin/bash

# Test VAE training on single GPU - QM9 only
# Run with: flux batch -q pdebug -t 1h -N 1 -n 1 -g 1 test_vae_singlegpu.sh

export PROJECT_ROOT=/usr/WS2/zhou6/all-atom-diffusion-transformer
cd $PROJECT_ROOT

echo "=== Starting VAE test training ==="
echo "Time: $(date)"
echo "Working directory: $(pwd)"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "GPUs available:"
rocm-smi --showid || echo "rocm-smi failed"
python -c "import torch; print(f'Device count: {torch.cuda.device_count()}'); print(f'Device 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run training - single GPU, QM9 only, DiT-S scale
# Validation enabled now that openbabel is installed
python src/train_autoencoder.py \
    data=qm9_only \
    callbacks=autoencoder_qm9_only \
    trainer=gpu \
    logger=csv \
    name="test_vae_qm9_latent8" \
    trainer.max_epochs=5 \
    test=False \
    autoencoder_module.latent_dim=8 \
    autoencoder_module.loss_weights.loss_kl.qm9=0.00001

echo "=== Training completed ==="
echo "Time: $(date)"

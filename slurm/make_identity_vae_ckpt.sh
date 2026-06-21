#!/bin/bash
# Generate the identity-VAE checkpoint with recommended first-try settings (QM9).
# CPU-only and quick; run on a node/login shell with internet (QM9 downloads on first run).
# Edit the env-activation line to match your cluster.
set -eo pipefail
source ~/.bashrc && mamba activate myenv
cd "$(dirname "$0")/.."
python scripts/make_identity_vae_ckpt.py \
    standardize=true center_pos=true center_types=false \
    out=checkpoints/identity_vae_qm9_std.ckpt

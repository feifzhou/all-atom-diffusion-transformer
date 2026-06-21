"""Produce a Lightning checkpoint for the identity (null-op) autoencoder.

The identity AE has no learnable parameters. With standardize=false no training/data is
needed: we just instantiate from config and write a checkpoint. With standardize=true we
additionally fit per-channel latent mean/std over the QM9 training split and bake them
into the checkpoint, so the diffusion model trains on ~unit-scale latents.

Usage:
    python scripts/make_identity_vae_ckpt.py
    python scripts/make_identity_vae_ckpt.py standardize=true
    python scripts/make_identity_vae_ckpt.py standardize=true out=checkpoints/identity_vae_qm9_std.ckpt
"""

import os
import sys
from functools import partial

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import lightning as L
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Data

DEFAULT_OUT = "checkpoints/identity_vae_qm9.ckpt"
QM9_ROOT = "data/qm9"


def custom_transform(data, removeHs=False):
    """Inlined copy of joint_datamodule.custom_transform (avoids the crystal import chain)."""
    atoms_to_keep = torch.ones_like(data.z, dtype=torch.bool)
    num_atoms = data.num_nodes
    if removeHs:
        atoms_to_keep = data.z != 1
        num_atoms = atoms_to_keep.sum().item()
    return Data(
        id=f"qm9_{data.name}",
        atom_types=data.z[atoms_to_keep],
        pos=data.pos[atoms_to_keep],
        frac_coords=torch.zeros_like(data.pos[atoms_to_keep]),
        cell=torch.zeros((1, 3, 3)),
        lattices=torch.zeros(1, 6),
        lattices_scaled=torch.zeros(1, 6),
        lengths=torch.zeros(1, 3),
        lengths_scaled=torch.zeros(1, 3),
        angles=torch.zeros(1, 3),
        angles_radians=torch.zeros(1, 3),
        num_atoms=torch.LongTensor([num_atoms]),
        num_nodes=torch.LongTensor([num_atoms]),
        spacegroup=torch.zeros(1, dtype=torch.long),
        token_idx=torch.arange(num_atoms),
        dataset_idx=torch.tensor([1], dtype=torch.long),
    )


def fit_qm9_standardization(module):
    """Fit latent mean/std over the QM9 training split (downloads QM9 on first run)."""
    from torch_geometric.datasets import QM9

    print("Loading QM9 to fit standardization stats (downloads on first run)...")
    dataset = QM9(root=QM9_ROOT, transform=partial(custom_transform, removeHs=False))
    train = dataset[:100000]  # matches JointDataModule train split size
    module.fit_standardization(train)
    print("latent_mean:", module.latent_mean.tolist())
    print("latent_std :", module.latent_std.tolist())


def main():
    overrides = [a for a in sys.argv[1:] if not a.startswith("out=")]
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("out=")), DEFAULT_OUT)

    with initialize(version_base=None, config_path="../configs/autoencoder_module"):
        cfg = compose(config_name="identity_vae", overrides=overrides)
    # The config interpolates ${trainer.check_val_every_n_epoch}; supply a plain value
    # and a concrete viz dir since we are instantiating the module standalone.
    OmegaConf.update(cfg, "scheduler_frequency", 1, force_add=True)
    OmegaConf.update(cfg, "visualization.save_dir", "/tmp/identity_vae", force_add=True)

    module = instantiate(cfg)
    n_learn = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Instantiated {type(module).__name__}: {n_learn} learnable params, "
          f"center_pos={module.center_pos}, center_types={module.center_types}, "
          f"standardize={module.standardize}")

    if module.standardize:
        module.eval()
        with torch.no_grad():
            fit_qm9_standardization(module)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    trainer = L.Trainer(accelerator="cpu", logger=False)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(out)
    print(f"Wrote checkpoint: {out}  ({os.path.getsize(out) / 1024:.1f} KB)")
    print(
        "\nTrain the diffusion generator on it with:\n"
        f"  python src/train_diffusion.py data=qm9_only callbacks=diffusion_qm9_only \\\n"
        f"    diffusion_module.autoencoder_ckpt={out} \\\n"
        f"    ++diffusion_module.autoencoder_cls=src.models.identity_vae_module.IdentityAutoencoderLitModule \\\n"
        f"    diffusion_module.denoiser.d_x=8 trainer=gpu logger=wandb name=DiT_qm9_identityAE"
    )


if __name__ == "__main__":
    main()

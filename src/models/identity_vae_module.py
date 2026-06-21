"""Copyright (c) Meta Platforms, Inc. and affiliates.

Identity ("null-op") autoencoder for QM9.

The bottleneck is simply ``z = concat(one_hot(atom_type), pos)`` with no learned
compression: 5 QM9 element classes (H, C, N, O, F) + 3 Cartesian coords = 8 dims.
Decoding is trivial: ``atom_type = argmax(z[:, :5])`` and ``pos = z[:, 5:]``.

This lets us train the latent-diffusion generator directly on (a lightly encoded)
data space, to test whether decent structures can be generated *without* the VAE's
learned latent space doing any heavy lifting.

It subclasses ``VariationalAutoencoderLitModule`` and overrides only ``encode`` and
``decode`` so that all of the loss / metric / eval / training machinery is reused
unchanged.
"""

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from torch_scatter import scatter
except ImportError:
    from src.models.components.scatter_fallback import scatter

from src.models.vae_module import (
    DiagonalGaussianDistribution,
    VariationalAutoencoderLitModule,
)

# QM9 elements (all-atom, hydrogens retained). Index in this list == one-hot class.
QM9_ATOMIC_NUMBERS = [1, 6, 7, 8, 9]  # H, C, N, O, F
NUM_QM9_CLASSES = len(QM9_ATOMIC_NUMBERS)  # 5


class _DeterministicPosterior(DiagonalGaussianDistribution):
    """Posterior whose mean *is* the identity latent: no sampling, zero KL.

    ``parameters`` is just the latent ``z`` repeated twice (mean, logvar) so that the
    parent's ``chunk(.., 2)`` recovers ``mean == z``; ``deterministic=True`` forces
    ``std == 0`` so both ``sample()`` and ``mode()`` return ``z`` exactly.
    """

    def __init__(self, z):
        super().__init__(torch.cat([z, z], dim=-1), deterministic=True)

    def kl(self, other=None):
        # No latent regularisation for an identity AE. Return device-correct zeros
        # with the per-node shape the parent class produces (n,).
        return torch.zeros(self.mean.shape[0], device=self.mean.device)


class _StubModule(torch.nn.Module):
    """Minimal stand-in exposing the attributes the base __init__ reads."""

    def __init__(self, max_num_elements: int) -> None:
        super().__init__()
        self.d_model = 1
        self.max_num_elements = max_num_elements


class IdentityAutoencoderLitModule(VariationalAutoencoderLitModule):
    """Null-op autoencoder: latent = concat(one_hot(type), pos). QM9 only."""

    def __init__(
        self,
        encoder=None,
        decoder=None,
        max_num_elements: int = 100,
        center_pos: bool = True,
        center_types: bool = False,
        standardize: bool = False,
        **kwargs,
    ) -> None:
        # The real encoder/decoder are never used; resolve max_num_elements (the one
        # attribute decode() needs) and feed the base only lightweight stubs so nothing
        # heavy is built or pickled into the checkpoint.
        if decoder is not None and hasattr(decoder, "max_num_elements"):
            max_num_elements = decoder.max_num_elements
        super().__init__(
            encoder=_StubModule(max_num_elements),
            decoder=_StubModule(max_num_elements),
            **kwargs,
        )
        # keep init args (minus the unused/heavy modules) in the checkpoint for reload
        self.save_hyperparameters(ignore=["encoder", "decoder"], logger=False)
        self.max_num_elements = max_num_elements
        # Per-molecule centering of the latent. Subtracting the coordinate centroid
        # removes the (chemically meaningless) absolute-translation DOF so the target
        # better matches the diffusion model's N(0,1) prior. center_types optionally
        # does the same for the one-hot channels (see class docstring / encode()).
        self.center_pos = center_pos
        self.center_types = center_types

        # Optional fixed global standardization of the latent: z <- (z - mean) / std,
        # per channel, using statistics over the (centered) training set. This puts the
        # one-hot and coordinate channels on a common ~unit scale that matches the
        # diffusion model's N(0,1) prior. Identity (mean=0, std=1) until fit_standardization
        # populates the buffers; they are saved in / restored from the checkpoint.
        self.standardize = standardize
        latent_dim = self.hparams.latent_dim
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

        # Drop all sub-modules: this AE has no parameters.
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()
        self.quant_conv = torch.nn.Identity()
        self.post_quant_conv = torch.nn.Identity()

        # atomic number <-> one-hot class lookups (registered so they follow .to(device))
        atomic_to_class = torch.full((self.max_num_elements,), -1, dtype=torch.long)
        for cls, z in enumerate(QM9_ATOMIC_NUMBERS):
            atomic_to_class[z] = cls
        self.register_buffer("atomic_to_class", atomic_to_class, persistent=False)
        self.register_buffer(
            "class_to_atomic",
            torch.tensor(QM9_ATOMIC_NUMBERS, dtype=torch.long),
            persistent=False,
        )

    def _encode_raw(self, batch) -> torch.Tensor:
        """Latent before optional standardization: concat(one_hot, pos), with centering."""
        # one-hot over the 5 QM9 element classes
        cls = self.atomic_to_class[batch.atom_types]  # (n,) in 0..4
        onehot = F.one_hot(cls, num_classes=NUM_QM9_CLASSES).to(batch.pos.dtype)  # (n, 5)
        # Cartesian coords in nm (matches the codebase's Angstrom/10 convention)
        pos_nm = batch.pos / 10.0  # (n, 3)

        # Per-molecule centering (translation-invariant for molecules; argmax-decodable
        # for types). Subtracting a per-graph constant preserves argmax of the one-hot.
        if self.center_pos:
            pos_nm = pos_nm - scatter(pos_nm, batch.batch, dim=0, reduce="mean")[batch.batch]
        if self.center_types:
            onehot = onehot - scatter(onehot, batch.batch, dim=0, reduce="mean")[batch.batch]

        return torch.cat([onehot, pos_nm], dim=-1)  # (n, 8)

    def encode(self, batch) -> Dict[str, torch.Tensor]:
        z = self._encode_raw(batch)
        if self.standardize:
            z = (z - self.latent_mean) / self.latent_std

        return {
            "x": z,
            "moments": torch.cat([z, z], dim=-1),
            "posterior": _DeterministicPosterior(z),
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            "token_idx": batch.token_idx,
        }

    def decode(self, encoded_batch) -> Dict[str, torch.Tensor]:
        z = encoded_batch["x"]  # (n, 8)
        if self.standardize:
            z = z * self.latent_std + self.latent_mean  # invert standardization
        type_logits = z[:, :NUM_QM9_CLASSES]  # (n, 5)
        pos_nm = z[:, NUM_QM9_CLASSES:]  # (n, 3)
        n = z.shape[0]
        n_graphs = encoded_batch["num_atoms"].shape[0]

        # Expand 5-class logits to atomic-number logits of width max_num_elements, so
        # the existing CE loss / argmax-eval (which work in atomic-number space) are reused.
        atom_types = z.new_full((n, self.max_num_elements), -1e4)
        atom_types[:, self.class_to_atomic] = type_logits

        return {
            "atom_types": atom_types,  # (n, max_num_elements); argmax -> atomic number
            "pos": pos_nm,  # (n, 3) in nm
            "frac_coords": torch.zeros_like(pos_nm),  # molecules: unused (loss weight 0)
            "lengths": z.new_zeros((n_graphs, 3)),
            "angles": z.new_zeros((n_graphs, 3)),
            "lattices": z.new_zeros((n_graphs, 6)),
        }

    @torch.no_grad()
    def fit_standardization(self, dataset, batch_size: int = 512, num_workers: int = 0) -> None:
        """Compute per-channel mean/std of the (centered) latent over a dataset and store
        them in the latent_mean / latent_std buffers. Stats are over all atoms (tokens),
        since the diffusion model operates per-atom. Call once before saving the checkpoint.
        """
        from torch_geometric.loader import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        device = self.latent_mean.device
        n = 0
        s = torch.zeros_like(self.latent_mean)
        ss = torch.zeros_like(self.latent_mean)
        for batch in loader:
            z = self._encode_raw(batch.to(device))  # pre-standardization latent
            n += z.shape[0]
            s += z.sum(dim=0)
            ss += (z * z).sum(dim=0)
        if n == 0:
            raise ValueError("fit_standardization received an empty dataset")
        mean = s / n
        var = (ss / n - mean**2).clamp_min(0.0)
        std = var.sqrt().clamp_min(1e-6)  # guard zero-variance channels
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

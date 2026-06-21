"""Smoke test for the identity (null-op) autoencoder.

Verifies that decode(encode(x)) reconstructs QM9 atom types exactly and coordinates
exactly up to the per-molecule centering applied by the AE, and that the LDM-style
path (posterior.sample() -> decode) round-trips too.

Run: python scripts/smoke_test_identity_vae.py
"""

import functools

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from torch_geometric.data import Batch, Data

from src.models.decoders.transformer import TransformerDecoder
from src.models.encoders.transformer import TransformerEncoder
from src.models.identity_vae_module import QM9_ATOMIC_NUMBERS, IdentityAutoencoderLitModule


def custom_transform(data):
    """Inlined copy of joint_datamodule.custom_transform (removeHs=False, molecule)."""
    n = data.num_nodes
    return Data(
        id=f"qm9_{data.name}",
        atom_types=data.z,
        pos=data.pos,
        frac_coords=torch.zeros_like(data.pos),
        cell=torch.zeros((1, 3, 3)),
        lattices=torch.zeros(1, 6),
        lattices_scaled=torch.zeros(1, 6),
        lengths=torch.zeros(1, 3),
        lengths_scaled=torch.zeros(1, 3),
        angles=torch.zeros(1, 3),
        angles_radians=torch.zeros(1, 3),
        num_atoms=torch.LongTensor([n]),
        num_nodes=torch.LongTensor([n]),
        spacegroup=torch.zeros(1, dtype=torch.long),
        token_idx=torch.arange(n),
        dataset_idx=torch.tensor([1], dtype=torch.long),
    )


def make_qm9_like_batch(num_mols=4, seed=0):
    """Build a batch matching the schema produced by joint_datamodule.custom_transform."""
    g = torch.Generator().manual_seed(seed)
    datas = []
    for i in range(num_mols):
        n = int(torch.randint(3, 12, (1,), generator=g).item())
        z = torch.tensor(
            [QM9_ATOMIC_NUMBERS[j] for j in torch.randint(0, 5, (n,), generator=g).tolist()],
            dtype=torch.long,
        )
        pos = torch.randn(n, 3, generator=g) * 3.0  # ~Angstrom scale
        raw = Data(z=z, pos=pos, name=f"mol{i}", num_nodes=n)
        datas.append(custom_transform(raw))
    return Batch.from_data_list(datas)


def build_module(center_pos=True, center_types=False):
    enc = TransformerEncoder(max_num_elements=100, d_model=64, nhead=4, num_layers=1)
    dec = TransformerDecoder(max_num_elements=100, d_model=64, nhead=4, num_layers=1)
    module = IdentityAutoencoderLitModule(
        encoder=enc,
        decoder=dec,
        latent_dim=8,
        center_pos=center_pos,
        center_types=center_types,
        optimizer=functools.partial(torch.optim.AdamW, lr=1e-4),
        scheduler=None,
        scheduler_frequency=1,
        loss_weights={
            "loss_atom_types": {"mp20": 0.0, "qm9": 1.0},
            "loss_lengths": {"mp20": 0.0, "qm9": 0.0},
            "loss_angles": {"mp20": 0.0, "qm9": 0.0},
            "loss_frac_coords": {"mp20": 0.0, "qm9": 0.0},
            "loss_pos": {"mp20": 0.0, "qm9": 10.0},
            "loss_kl": {"mp20": 0.0, "qm9": 0.0},
        },
        augmentations={"noise": 0.0, "frac_coords": False, "pos": False},
        visualization={"visualize": False, "save_dir": "/tmp"},
        compile=False,
    )
    return module.eval()


def center_per_mol(pos, batch_index):
    """Subtract each molecule's coordinate centroid (for translation-invariant compare)."""
    try:
        from torch_scatter import scatter
    except ImportError:
        from src.models.components.scatter_fallback import scatter

    return pos - scatter(pos, batch_index, dim=0, reduce="mean")[batch_index]


def main():
    torch.set_grad_enabled(False)
    batch = make_qm9_like_batch()
    module = build_module(center_pos=True, center_types=False)

    n_learn = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"Learnable parameters in identity AE: {n_learn}  "
          f"(center_pos={module.center_pos}, center_types={module.center_types})")

    # --- 1. latent shape / contents ---
    enc = module.encode(batch)
    z = enc["x"]
    assert z.shape == (batch.num_nodes, 8), z.shape
    onehot = z[:, :5]
    assert torch.all((onehot == 0) | (onehot == 1)), "type slice not one-hot (center_types off)"
    assert torch.all(onehot.sum(dim=-1) == 1), "type slice not a valid one-hot"
    # coordinate part must be zero-centroid per molecule
    com = center_per_mol(z[:, 5:], batch.batch)
    assert torch.allclose(z[:, 5:], com, atol=1e-6), "coords not centered per molecule"
    print(f"Latent z shape: {tuple(z.shape)}  (one-hot[5] + centered pos_nm[3]) -- OK")

    # --- 2. deterministic posterior: sample()==mode()==z, kl==0 ---
    post = enc["posterior"]
    assert torch.equal(post.sample(), z) and torch.equal(post.mode(), z)
    assert torch.count_nonzero(post.kl()) == 0 and post.kl().shape == (batch.num_nodes,)
    print("Posterior is deterministic (sample==mode==z) and KL==0 -- OK")

    # --- 3. reconstruction via decode (exact for types; up to translation for coords) ---
    out = module.decode(enc)
    recon_types = out["atom_types"].argmax(dim=-1)
    recon_pos = out["pos"] * 10.0  # nm -> Angstrom
    types_ok = torch.equal(recon_types, batch.atom_types)
    # compare centered coordinates (centering removes only the absolute translation)
    pos_err = (center_per_mol(recon_pos, batch.batch)
               - center_per_mol(batch.pos, batch.batch)).abs().max().item()
    print(f"Atom types exact match: {types_ok}")
    print(f"Max centered-coordinate abs error (Angstrom): {pos_err:.3e}")
    assert types_ok, "atom types not perfectly reconstructed"
    assert pos_err < 1e-5, "coords not reconstructed (up to translation)"

    # --- 4. full forward (as used in train/eval), deterministic ---
    out_fwd, _ = module.forward(batch, sample_posterior=True)
    assert torch.equal(out_fwd["atom_types"].argmax(dim=-1), batch.atom_types)
    print("Full forward() round-trips types and centered coords exactly -- OK")

    # --- 5. LDM-style path: decode arbitrary generated latents (continuous) ---
    fake_latent = torch.randn(batch.num_nodes, 8)
    gen = {"x": fake_latent, "num_atoms": batch.num_atoms,
           "batch": batch.batch, "token_idx": batch.token_idx}
    gout = module.decode(gen)
    gen_types = gout["atom_types"].argmax(dim=-1)
    assert torch.equal(gen_types, torch.tensor(QM9_ATOMIC_NUMBERS)[fake_latent[:, :5].argmax(-1)])
    assert torch.equal(gout["pos"], fake_latent[:, 5:])
    assert gen_types.min() >= 1, "decoded a null (0) atom type"
    print("Generated-latent decode maps to valid atomic numbers + passes coords -- OK")

    # --- 6. center_types=True variant still decodes types correctly via argmax ---
    mod_ct = build_module(center_pos=True, center_types=True)
    enc_ct = mod_ct.encode(batch)
    oh_ct = enc_ct["x"][:, :5]
    assert not torch.all((oh_ct == 0) | (oh_ct == 1)), "center_types did not shift one-hot"
    out_ct = mod_ct.decode(enc_ct)
    assert torch.equal(out_ct["atom_types"].argmax(-1), batch.atom_types), \
        "center_types broke argmax type decoding"
    print("center_types=True: one-hot shifted but argmax type decoding still exact -- OK")

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()

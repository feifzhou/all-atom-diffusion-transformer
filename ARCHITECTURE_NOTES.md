# ADiT Architecture Notes

## Transformer VAE Position Embedding Issue

### Paper vs Code Discrepancy

**Paper (Algorithms 1 & 2):** NO positional/index embedding mentioned
- Encoder: atom type embedding + coordinate MLPs → TransformerEncoder
- Decoder: latent → TransformerEncoder → prediction heads

**Code Implementation:** Positional embedding present
- Encoder: `src/models/encoders/transformer.py:120`
- Decoder: `src/models/decoders/transformer.py:97`
- Uses sinusoidal embedding: `get_index_embedding(batch.token_idx, d_model)`
- `token_idx = torch.arange(num_atoms)` - arbitrary ordering from dataset files

### Why This Is Problematic

- Atom ordering in molecular files is arbitrary (no canonical ordering)
- Position embedding breaks permutation equivariance
- No permutation augmentation in training code to compensate
- Molecular properties are permutation invariant - atom order shouldn't matter

**Contrast with:**
- **LLMs:** Word order matters ("dog bites man" ≠ "man bites dog") → position embedding appropriate
- **Image DiT:** Spatial position matters (top-left ≠ bottom-right) → position embedding appropriate
- **Molecules:** Atom order arbitrary → position embedding inappropriate

### Likely Origin

Model appears to be latent DiT with patch_size=1 (each atom = one patch):
- Inherited from image/text DiT template code
- Standard LLM practices: padding + positional embedding + masked attention
- Position embedding makes sense in source domain (images/text), not target domain (molecules)
- Paper correctly omits it; implementation kept legacy code

### What IS Appropriate (Standard LLM Practice)

- **Padding:** Variable num_atoms → pad to batch max → `to_dense_batch()`
- **Masked attention:** `src_key_padding_mask` prevents attending to padding tokens
- Both are correct for variable-length sequences

### Recommendation

Remove position embedding (comment out lines 120 in encoder, 97 in decoder) to restore permutation equivariance. Without it, latents would be equivariant to atom permutations, which is chemically appropriate.

## Model Architecture Summary

**ADiT = Latent DiT with patch_size=1**
- VAE compresses N atoms → N latent tokens (dim d=8)
- Only bottleneck: latent dimension, NOT spatial (still N tokens)
- Diffusion on latent space (not raw atom coordinates)
- Standard Transformer encoder/decoder architecture

## Identity ("Null-op") Autoencoder Experiment (QM9)

### Motivation

ADiT couples a learned VAE (first stage) with a latent diffusion DiT (second stage).
We want to isolate the contribution of the VAE: **can the DiT generate decent QM9
molecules with no learned compression at all** — i.e. diffusing directly in (a lightly
encoded) data space? If quality holds up, the latent-space formalism is doing less heavy
lifting than assumed; if it collapses, the VAE earns its keep.

### What the identity AE does

Replaces the VAE with a parameter-free map. For QM9 (all-atom, elements H,C,N,O,F):
- **Encode:** `z = concat(one_hot_5(atom_type), pos_nm)` → 8 dims (5 type + 3 coords).
- **Decode:** `atom_type = argmax(z[:, :5])`, `pos = z[:, 5:]` (argmax is invariant to any
  per-molecule additive shift, so centering/standardization stay decodable).
- Deterministic posterior (`sample()==mode()==z`, `KL==0`) so the DiT trains on `z` directly.

### Implementation choices (max reuse, non-invasive)

- `src/models/identity_vae_module.py`: `IdentityAutoencoderLitModule` **subclasses**
  `VariationalAutoencoderLitModule` and overrides only `encode`/`decode`. All loss / metric
  / eval / training machinery is inherited unchanged. Override is at the *LitModule* level
  (not the encoder/decoder `nn.Module`s) because `quant_conv` + the Gaussian sampler sit
  between the encoder output and the latent.
- The unused encoder/decoder/quant layers are replaced with `nn.Identity()`; `__init__`
  takes lightweight stubs so nothing heavy is built or pickled → checkpoint is ~15 KB,
  0 learnable params.
- Atom types use a fixed vocab `[1,6,7,8,9]`; decode scatters the 5 type-logits into a
  width-`max_num_elements` logit vector so the existing CE-loss / argmax-eval (which work
  in atomic-number space) are reused verbatim.
- DiT integration is **config-driven, not hardcoded**: `ldm_module.py` now loads the AE via
  `diffusion_module.autoencoder_cls` (`hydra.utils.get_class(...).load_from_checkpoint(...)`).
  The transformer and Equiformer VAEs both use the default class; only the identity AE
  overrides it. No editing required to switch back.

### Tunable parts (`configs/autoencoder_module/identity_vae.yaml`)

- `center_pos` (default **true**): subtract the per-molecule coordinate centroid. Removes the
  meaningless absolute-translation DOF and matches the interpolant's per-sample-centered
  Gaussian prior (`flow_matching.py:_centered_gaussian`, which centers *all* channels).
- `center_types` (default **false**): subtract the per-molecule mean one-hot. Off by default
  (keeps each element a fixed, composition-independent code). The interpolant centers the
  one-hot channels of the noise too, so there is a real *prior-consistency* argument for
  `true` — it is the single most informative follow-up ablation.
- `standardize` (default **false**, but **recommended true**): fixed global per-channel
  `(z - mean)/std`. Puts one-hot (~0.2/0.4) and coordinate (~0/0.24 nm) channels on a common
  unit scale matching the N(0,1) prior — the change most likely to decide success. Stats are
  fit over the QM9 train split and baked into the checkpoint (`fit_standardization`).

**Recommended first experiment:** `center_pos=true, center_types=false, standardize=true`,
DiT-B, QM9-only. First ablation: flip `center_types=true`.

### Instructions

```bash
# 1. Mint the checkpoint (CPU, minutes; QM9 downloads on first run).
bash slurm/make_identity_vae_ckpt.sh
#    -> writes checkpoints/identity_vae_qm9_std.ckpt (standardize=true)
#    Or directly:
#    python scripts/make_identity_vae_ckpt.py standardize=true center_pos=true center_types=false \
#        out=checkpoints/identity_vae_qm9_std.ckpt

# 2. Train the DiT on it (4 GPUs).
sbatch slurm/train_ddp_ldm_identity.sh

# Verify the AE logic any time (parameter-free, exact reconstruction):
python scripts/smoke_test_identity_vae.py
```

The diffusion run uses `data=qm9_only callbacks=diffusion_qm9_only`,
`++diffusion_module.autoencoder_cls=src.models.identity_vae_module.IdentityAutoencoderLitModule`,
and `++diffusion_module.denoiser.d_x=8` (must equal the identity latent dim). Note:
`JointDataModule.setup()` still instantiates MP20/QMOF even at proportion 0, so those
datasets must be loadable on the training node (or comment out their instantiation).

### Caveats

- Discrete types are diffused as continuous one-hot logits and decoded by argmax — validity
  may trail a learned latent; that gap is part of what the experiment measures.
- The index-based positional embedding issue documented above applies to the DiT here too:
  with raw coordinates in the tokens, the model could exploit dataset atom-ordering as a
  shortcut. If results look suspiciously good, ablate the PE.

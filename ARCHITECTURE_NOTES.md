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

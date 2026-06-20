"""
Fallback implementation of torch_scatter.scatter for simple reduce operations.
Supports reduce='sum' (default) and reduce='mean'.
Matches torch_scatter API: default is 'sum', codebase uses 'mean' explicitly.
"""

import torch


def scatter(src, index, dim=0, reduce="sum"):
    """
    Simple fallback for torch_scatter.scatter.
    Supports reduce='sum' and reduce='mean'.
    Default is 'sum' to match torch_scatter API.

    Args:
        src: Source tensor to scatter from
        index: Index tensor indicating which output position each input goes to
        dim: Dimension along which to scatter (only dim=0 supported)
        reduce: Reduction operation ('sum' or 'mean')

    Returns:
        Scattered tensor with specified reduction
    """
    if reduce not in ("sum", "mean"):
        raise NotImplementedError(f"Only reduce='sum' and 'mean' are implemented, got {reduce}")

    if dim != 0:
        raise NotImplementedError(f"Only dim=0 is implemented, got {dim}")

    # Get number of unique indices
    num_groups = index.max().item() + 1

    # Create output tensor
    out_shape = list(src.shape)
    out_shape[0] = num_groups
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)

    # Count elements per group
    counts = torch.zeros(num_groups, dtype=torch.long, device=src.device)

    # Accumulate sums
    out.index_add_(0, index, src)

    if reduce == "sum":
        return out

    # For mean: divide by counts
    counts.index_add_(0, index, torch.ones(len(index), dtype=torch.long, device=src.device))
    counts = counts.clamp(min=1)  # Avoid division by zero
    out = out / counts.view(-1, *([1] * (len(out_shape) - 1)))

    return out

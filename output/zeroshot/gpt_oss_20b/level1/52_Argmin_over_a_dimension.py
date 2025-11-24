import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,          # pointer to input (float32)
    out_ptr,        # pointer to output (int64)
    dim1: tl.constexpr,
    dim2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output elements
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Total number of output elements = batch * dim2
    n_out = dim1 * dim2
    mask = offsets < n_out

    # Map each output element to (batch_idx, dim2_idx)
    batch_idx = offsets // dim2
    dim2_idx = offsets % dim2

    # Base pointer for each batch
    batch_offset = batch_idx * dim1 * dim2

    # Initialize min values and indices
    min_val = tl.full([BLOCK_SIZE], 1e20, dtype=tl.float32)
    min_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)

    for i in range(dim1):
        # Compute input offsets for the i-th row
        in_offsets = batch_offset + i * dim2 + dim2_idx
        vals = tl.load(x_ptr + in_offsets, mask=mask, other=1e20)

        # Update minima
        mask_smaller = vals < min_val
        min_val = tl.where(mask_smaller, vals, min_val)
        min_idx = tl.where(mask_smaller, tl.full([BLOCK_SIZE], i, dtype=tl.int64), min_idx)

    # Store results
    tl.store(out_ptr + offsets, min_idx, mask=mask)


def triton_argmin(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Compute argmin along the specified dimension using Triton.
    """
    assert x.is_cuda, "Input must be on CUDA."
    assert x.ndim == 3, "Input must be 3‑D."
    assert dim == 1, "Only dim=1 is supported in this kernel."

    x = x.contiguous()
    batch_size, dim1, dim2 = x.shape

    out = torch.empty((batch_size, dim2), dtype=torch.int64, device=x.device)

    BLOCK_SIZE = 1024  # can be tuned

    grid = lambda meta: ((batch_size * dim2 + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    argmin_kernel[grid](x, out, dim1=dim1, dim2=dim2, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that replaces torch.argmin with a custom Triton kernel.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)
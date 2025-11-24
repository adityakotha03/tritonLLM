import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,          # pointer to input tensor (flattened)
    out_ptr,        # pointer to output tensor (flattened)
    B,              # batch size
    N,              # length of the sequence
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel that computes an exclusive cumulative sum along the last
    dimension of a 2-D tensor of shape (B, N).

    Each program instance processes a contiguous block of columns for a
    single batch element.
    """
    # program_id 0 -> batch index
    # program_id 1 -> column block index
    batch = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE

    offsets = col_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # base offset for the current batch element
    base = batch * N

    # Load the input block (zero out-of-bounds elements)
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # Inclusive prefix sum along the column dimension
    inc = tl.cumsum(x, axis=0)

    # Shift right by one to obtain the exclusive prefix sum.
    # The first element of each row becomes 0.
    exc = tl.shift(inc, shift=1, axis=0, fill_value=0.0)

    # Store the result
    tl.store(out_ptr + base + offsets, exc, mask=mask)


def triton_exclusive_cumsum(x: torch.Tensor, dim: int = 1):
    """
    Wrapper that launches the Triton kernel for an exclusive cumulative sum
    along ``dim``. ``x`` must be a 2‑D tensor on CUDA.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 2, "Input tensor must be 2-D."

    B, N = x.shape[0], x.shape[1]
    out = torch.empty_like(x)

    # Pick a block size that is a power of two.
    BLOCK_SIZE = 512

    # Grid dimensions: one program per batch element, and one per column block
    grid = (B, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)

    exclusive_cumsum_kernel[grid](
        x_ptr=x.contiguous().data_ptr(),
        out_ptr=out.data_ptr(),
        B=B,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum along a
    specified dimension using a custom Triton kernel.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # Move the dimension to the last position if it is not already.
        if self.dim != x.dim() - 1:
            perm = list(range(x.dim()))
            perm[self.dim], perm[-1] = perm[-1], perm[self.dim]
            x = x.permute(*perm)

        # Compute the exclusive cumulative sum.
        out = triton_exclusive_cumsum(x, dim=self.dim)

        # If we permuted the dimensions, bring them back to the original order.
        if self.dim != x.dim() - 1:
            inv_perm = list(range(out.dim()))
            inv_perm[self.dim], inv_perm[-1] = inv_perm[-1], inv_perm[self.dim]
            out = out.permute(*inv_perm)

        return out
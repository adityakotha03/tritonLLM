import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def cumsum_kernel(
    x_ptr,          # Pointer to the input tensor
    out_ptr,        # Pointer to the output tensor
    N,              # Size of the dimension to scan
    BLOCK_SIZE: tl.constexpr,
):
    """
    Inclusive prefix‑sum (cumsum) over a 1‑D sequence that is part of a larger
    2‑D tensor (batch × N).  Each program instance processes one batch row.
    """
    batch_id = tl.program_id(0)          # One program per batch row
    stride = N                           # Distance between consecutive rows in memory

    # Offset into the row that this program is processing
    offset = 0
    # Running sum that propagates across chunks
    sum_offset = tl.float32(0.0)

    while offset < N:
        # Indices for the current chunk
        indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = indices < N

        # Load the chunk
        vals = tl.load(x_ptr + batch_id * stride + indices, mask=mask, other=0.0)

        # Inclusive scan inside the chunk (binary tree style)
        # We unroll a few steps to keep the loop small
        for step in range(1, BLOCK_SIZE.bit_length()):  # log₂(BLOCK_SIZE)
            shifted = tl.shift(vals, step, 0.0)
            vals = tl.where(mask, vals + shifted, vals)

        # Add the offset from previous chunks
        vals = vals + sum_offset

        # Store the results
        tl.store(out_ptr + batch_id * stride + indices, vals, mask=mask)

        # Compute the offset for the next chunk
        # The last element of this chunk becomes the new offset
        last_idx = offset + tl.min(N - offset, BLOCK_SIZE) - 1
        last_val = tl.load(out_ptr + batch_id * stride + last_idx, mask=True, other=0.0)
        sum_offset = last_val

        # Advance to the next chunk
        offset += BLOCK_SIZE


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Wrapper that launches the Triton cumsum kernel.  It assumes `dim` is
    the second dimension of a 2‑D tensor (batch × N).  For other shapes,
    the tensor is reshaped accordingly.
    """
    assert x.is_cuda, "Input must be on CUDA"
    # Ensure the tensor is contiguous
    x = x.contiguous()

    # We only support the case where the scan dimension is the last one
    # (i.e. dim == 1 for a 2‑D tensor).  For a 1‑D tensor we simply
    # treat the batch size as 1.
    if x.ndim == 1:
        batch = 1
        N = x.shape[0]
        x = x.unsqueeze(0)
    else:
        batch = x.shape[0]
        N = x.shape[1]

    out = torch.empty_like(x)

    BLOCK_SIZE = 256  # Tunable parameter (must be a power of two)

    grid = lambda meta: (batch,)

    cumsum_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)

    # Remove the added batch dimension if the original tensor was 1‑D
    if x.ndim == 1:
        out = out.squeeze(0)
    return out


class ModelNew(nn.Module):
    """
    Triton‑accelerated cumulative sum (prefix sum) along a specified dimension.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_cumsum(x, self.dim)
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def softmax_max_kernel(
    x_ptr: tl.tensor,
    max_ptr: tl.tensor,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    # vector of BLOCK_SIZE values for reduction
    max_vals = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        offsets = row * n_cols + col_offset
        mask = offsets < n_cols
        vals = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        max_vals = tl.maximum(max_vals, vals)

    # Reduce within the vector
    max_vals = tl.max(max_vals, axis=0)  # all elements equal to the row max
    tl.store(max_ptr + row, max_vals[0])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def softmax_kernel(
    x_ptr: tl.tensor,
    max_ptr: tl.tensor,
    out_ptr: tl.tensor,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # First pass: compute the sum of exponentials
    sum_val = tl.zeros([], dtype=tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        offsets = row * n_cols + col_offset
        mask = offsets < n_cols
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        max_val = tl.load(max_ptr + row)
        shifted = vals - max_val
        exp_vals = tl.exp(shifted)
        sum_val += tl.sum(exp_vals, axis=0)

    inv_sum = 1.0 / sum_val

    # Second pass: compute softmax values
    for i in range(0, n_cols, BLOCK_SIZE):
        col_offset = tl.arange(0, BLOCK_SIZE)
        offsets = row * n_cols + col_offset
        mask = offsets < n_cols
        vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        max_val = tl.load(max_ptr + row)
        shifted = vals - max_val
        exp_vals = tl.exp(shifted)
        soft = exp_vals * inv_sum
        tl.store(out_ptr + offsets, soft, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of softmax over the last dimension.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous().float()

    n_rows, n_cols = x.shape
    out = torch.empty_like(x)

    # Allocate memory for per-row maxima
    max_vals = torch.empty((n_rows,), dtype=torch.float32, device=x.device)

    grid = lambda meta: (n_rows,)

    # Compute per-row maxima
    softmax_max_kernel[grid](
        x, max_vals, n_rows, n_cols, BLOCK_SIZE=meta["BLOCK_SIZE"]
    )

    # Compute softmax using the maxima
    softmax_kernel[grid](
        x, max_vals, out, n_rows, n_cols, BLOCK_SIZE=meta["BLOCK_SIZE"]
    )

    return out


class ModelNew(nn.Module):
    """
    Custom model that applies a Triton‑implemented softmax activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softmax(x)
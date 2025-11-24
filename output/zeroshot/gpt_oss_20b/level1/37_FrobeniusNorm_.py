import torch
import triton
import triton.language as tl


# Kernel to compute partial sum of squared elements
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1 << 10}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1 << 11}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1 << 12}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def _compute_partial_sum(
    x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = tl.sum(x * x, axis=0)
    tl.store(out_ptr + tl.program_id(0), s)


# Kernel to normalize elements by a scalar
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1 << 10}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1 << 11}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1 << 12}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def _normalize(
    x_ptr, out_ptr, norm, N, BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / norm
    tl.store(out_ptr + offsets, out, mask=mask)


def frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute Frobenius norm and normalize the input tensor."""
    x = x.contiguous()
    N = x.numel()
    block_size = 1 << 10  # starting block size, autotuner may change it

    # Partial sum kernel
    num_blocks = (N + block_size - 1) // block_size
    partial_sums = torch.empty(num_blocks, dtype=x.dtype, device=x.device)
    grid_partial = lambda meta: (num_blocks,)
    _compute_partial_sum[grid_partial](
        x, partial_sums, N, BLOCK_SIZE=block_size
    )

    # Reduce partial sums on GPU
    total = torch.sum(partial_sums)
    norm = torch.sqrt(total)

    # Normalization kernel
    out = torch.empty_like(x)
    grid_norm = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _normalize[grid_norm](
        x, out, norm.item(), N, BLOCK_SIZE=block_size
    )
    return out


class ModelNew(torch.nn.Module):
    """Model performing Frobenius norm normalization using Triton kernels."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frobenius_norm(x)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def frob_norm_kernel(
    x_ptr,  # Pointer to input tensor
    norm_ptr,  # Pointer to output norm (scalar)
    n_elements,  # Total number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data and compute squared values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_sq = x * x

    # Compute sum of squares in block
    sum_sq = tl.sum(x_sq, axis=0)

    # Store partial sum in shared memory for reduction
    tl.atomic_add(norm_ptr, sum_sq)


@triton.jit
def normalize_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    norm,  # Normalization scalar (Frobenius norm)
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input, normalize and store
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x / (norm + 1e-8)  # Add small epsilon to avoid division by zero
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_frobenius_norm(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."

    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()

    # Use a reasonable block size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate a temporary scalar for sum of squares
    sum_sq = torch.zeros(1, dtype=torch.float32, device=x.device)

    # First kernel: compute sum of squares
    frob_norm_kernel[(grid_size,)](x, sum_sq, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Compute norm = sqrt(sum of squares)
    norm = torch.sqrt(sum_sq).item()

    # Second kernel: normalize
    normalize_kernel[(grid_size,)](x, out, norm, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    """
    Optimized version of Frobenius norm normalization using Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_frobenius_norm(x)
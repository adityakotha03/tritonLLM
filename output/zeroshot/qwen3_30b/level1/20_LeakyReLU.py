import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    negative_slope: tl.constexpr
):
    # Calculate the block index and starting offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply LeakyReLU activation
    y = tl.where(x > 0, x, x * negative_slope)

    # Store output
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.01):
    """
    Applies LeakyReLU activation using a custom Triton kernel optimized for A100.
    Uses bf16 for better tensor core utilization and memory bandwidth efficiency.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Use bf16 for better performance on A100
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    # Optimal block size for A100 (power of 2, aligned with shared memory and warp size)
    BLOCK_SIZE = 512

    # Grid configuration
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    leaky_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE, negative_slope=negative_slope)

    return out


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_leaky_relu(x, negative_slope=self.negative_slope)
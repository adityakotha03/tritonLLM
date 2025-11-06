import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid: 1 / (1 + exp(-x))
    # Use a numerically stable version
    # We can use tl.sigmoid directly (Triton's built-in for stability and speed)
    out = tl.sigmoid(x)

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor):
    """
    Apply Sigmoid activation using a custom Triton kernel.
    Optimized for large tensors using block processing and coalesced memory access.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()

    # Block size: power of 2, tuned for A100 (e.g., 256, 512, 1024)
    # 512 is a good balance of occupancy and memory bandwidth
    BLOCK_SIZE = 512

    # Grid configuration
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],

    # Launch kernel
    sigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_sigmoid(x)
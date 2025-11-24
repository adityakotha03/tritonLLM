import torch
import torch.nn as nn
import triton
import triton.language as tl


# Element‑wise ReLU kernel written in Triton
@triton.jit
def relu_kernel(
    x_ptr,          # Pointer to input tensor
    out_ptr,        # Pointer to output tensor
    n_elements,     # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block start index for this program instance
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for handling the last partial block
    mask = offsets < n_elements

    # Load data, apply ReLU, and store results
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Apply element‑wise ReLU to a tensor using the Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n = x.numel()
    # Choose a block size that fits well into SM resources
    BLOCK_SIZE = 4096

    # Grid configuration: one program per BLOCK_SIZE elements
    grid = lambda meta: ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.relu with a Triton kernel.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
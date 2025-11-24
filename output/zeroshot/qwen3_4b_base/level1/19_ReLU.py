import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x >= 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor):
    """
    Custom ReLU kernel using Triton for improved performance on A100.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimal for Ampere, balances occupancy and memory coalescing

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_relu(x)
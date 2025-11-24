import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # We compute the tanh argument efficiently
    sqrt_2_over_pi = 0.7978845608
    x_squared = x * x
    x_cubed = x_squared * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_val = tl.tanh(tanh_arg)
    out = x * (0.5 * (1.0 + tanh_val))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gelu(x: torch.Tensor):
    """
    Custom GELU implementation using Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimal block size for GELU on Ampere (power of 2, balances register use and memory)

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    gelu_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return triton_gelu(x)
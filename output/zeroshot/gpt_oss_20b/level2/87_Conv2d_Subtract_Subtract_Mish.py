import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernel: subtract two scalars and apply Mish activation
# --------------------------------------------------------------------
@triton.jit
def subtract_mish_kernel(
    input_ptr,
    out_ptr,
    n_elements,
    sub1: tl.constexpr,
    sub2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Subtract scalars
    x = x - sub1
    x = x - sub2

    # Mish activation: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    exp_x = tl.exp(x)
    softplus = tl.log1p(exp_x)
    tanh_sp = tl.tanh(softplus)
    mish = x * tanh_sp

    # Store result
    tl.store(out_ptr + offsets, mish, mask=mask)

def subtract_mish(input: torch.Tensor, sub1: float, sub2: float) -> torch.Tensor:
    """
    Wrapper for the Triton kernel that performs:
        out = mish(input - sub1 - sub2)
    """
    assert input.is_cuda, "Input tensor must be on CUDA."

    out = torch.empty_like(input)

    n_elements = input.numel()
    BLOCK_SIZE = 1024  # Tune this for best performance

    # Grid definition
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    subtract_mish_kernel[grid](
        input,
        out,
        n_elements,
        sub1=sub1,
        sub2=sub2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# --------------------------------------------------------------------
# Optimized model using the custom Triton kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtracts two values, and applies Mish activation
    using a custom Triton kernel for the subtract + Mish stage.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

    def forward(self, x):
        # Convolution using PyTorch's optimized backend
        x = self.conv(x)

        # Subtract two scalars and apply Mish using the Triton kernel
        x = subtract_mish(x, self.subtract_value_1, self.subtract_value_2)

        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def sub_tanh_kernel(
    x_ptr,               # Pointer to input tensor (conv output)
    out_ptr,             # Pointer to output tensor (after tanh)
    sub1_value,          # Scalar to subtract before tanh
    n_elements,          # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data, subtract, apply tanh, and store
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x - sub1_value
    x = tl.math.tanh(x)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit
def sub_kernel(
    x_ptr,               # Pointer to input tensor (after tanh)
    out_ptr,             # Pointer to output tensor (after second subtract)
    sub2_value,          # Scalar to subtract
    n_elements,          # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x - sub2_value
    tl.store(out_ptr + offsets, x, mask=mask)


# --------------------------------------------------------------------------- #
# Wrapper functions
# --------------------------------------------------------------------------- #

def triton_sub_tanh(x: torch.Tensor, sub1_value: float) -> torch.Tensor:
    """Apply subtract1 + tanh using Triton."""
    assert x.is_cuda, "Input must be on CUDA."
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sub_tanh_kernel[grid](
        x,
        out,
        sub1_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_sub(x: torch.Tensor, sub2_value: float) -> torch.Tensor:
    """Apply subtract2 using Triton."""
    assert x.is_cuda, "Input must be on CUDA."
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    sub_kernel[grid](
        x,
        out,
        sub2_value,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# --------------------------------------------------------------------------- #
# New model
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtract, tanh activation, subtract, and average pooling.
    All scalar subtractions and tanh are fused into custom Triton kernels for maximum speed.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        # 1) Convolution
        x = self.conv(x)

        # 2) Subtract1 + tanh (Triton)
        x = triton_sub_tanh(x, self.subtract1_value)

        # 3) Subtract2 (Triton)
        x = triton_sub(x, self.subtract2_value)

        # 4) Average pooling (PyTorch)
        x = self.avgpool(x)

        return x
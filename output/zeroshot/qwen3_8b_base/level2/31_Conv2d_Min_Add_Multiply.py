import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_with_constant_kernel(
    x_ptr,  # Pointer to input tensor
    constant_ptr,  # Pointer to constant value
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    constant = tl.load(constant_ptr, mask=mask, other=0.0)

    # Compute min(x, constant)
    out = tl.where(x < constant, x, constant)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def add_bias_kernel(
    x_ptr,  # Pointer to input tensor
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Compute x + bias
    out = x + bias

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def scale_kernel(
    x_ptr,  # Pointer to input tensor
    scale,  # Pointer to scaling factor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale_val = tl.load(scale, mask=mask, other=0.0)

    # Compute x * scale
    out = x * scale_val

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_min_with_constant(x: torch.Tensor, constant: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Prepare output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    min_with_constant_kernel[lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x, torch.tensor(constant).cuda(), out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_add_bias(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Prepare output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    add_bias_kernel[lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_scale(x: torch.Tensor, scale: float):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    BLOCK_SIZE = 128

    # Prepare output tensor
    out = torch.empty_like(x)

    # Launch the Triton kernel
    scale_kernel[lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x, torch.tensor(scale).cuda(), out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = triton_min_with_constant(x, self.constant_value)
        x = triton_add_bias(x, self.bias)
        x = triton_scale(x, self.scaling_factor)
        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def instance_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    scale_ptr, shift_ptr,
    output_ptr,
    n_elements,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    channel_id = pid // (n_elements // num_channels)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute mean per channel
    channel_start = (offsets // (n_elements // num_channels)) * (n_elements // num_channels)
    channel_mask = (channel_start <= offsets) & (offsets < channel_start + (n_elements // num_channels))
    x_clean = tl.where(channel_mask, x, 0.0)
    mean = tl.sum(x_clean, axis=0) / (n_elements // num_channels)

    # Compute variance
    diff = tl.where(channel_mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / (n_elements // num_channels)

    # Normalize
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    normed = (x - mean) * inv_std

    # Apply scale (weight) and shift (bias)
    scale = tl.load(weight_ptr + channel_id, mask=channel_id < num_channels, other=1.0)
    shift = tl.load(bias_ptr + channel_id, mask=channel_id < num_channels, other=0.0)
    output = normed * scale + shift

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def divide_kernel(
    x_ptr,
    scalar,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x / scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_instance_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    num_channels = x.shape[1]
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    instance_norm_kernel[grid](
        x, weight, bias, None, None,
        None, None,
        out,
        n_elements,
        num_channels,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_divide(x: torch.Tensor, scalar: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    divide_kernel[grid](x, scalar, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for InstanceNorm and division.
    Conv2d is kept as-is since it's already highly optimized in PyTorch (uses cuDNN).
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm_weight = nn.Parameter(torch.ones(out_channels))
        self.instance_norm_bias = nn.Parameter(torch.zeros(out_channels))
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = triton_instance_norm(x, self.instance_norm_weight, self.instance_norm_bias)
        x = triton_divide(x, self.divide_by)
        return x
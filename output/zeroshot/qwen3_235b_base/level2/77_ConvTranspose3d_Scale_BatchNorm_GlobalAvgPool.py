import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_kernel(
    x_ptr,  # Pointer to input
    scale_factor,
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * scale_factor
    tl.store(out_ptr + offsets, output, mask=mask)


def triton_mul(x: torch.Tensor, scale_factor: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_kernel[grid](x, scale_factor, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def batch_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    out_ptr, n_channels, num_elements_per_channel, eps,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    pid_b = tl.program_id(0)  # channel block
    pid_hw = tl.program_id(1)  # spatial block

    # Channel block range
    channels_offset = pid_b * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    mask_c = channels_offset < n_channels

    # Spatial block range
    hw_offset = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    mask_hw = hw_offset < num_elements_per_channel
    offsets = channels_offset[:, None] * num_elements_per_channel + hw_offset[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channels_offset, mask=mask_c, other=1.0)
    bias = tl.load(bias_ptr + channels_offset, mask=mask_c, other=0.0)
    mean = tl.load(running_mean_ptr + channels_offset, mask=mask_c, other=0.0)
    inv_std = tl.math.rsqrt(tl.load(running_var_ptr + channels_offset, mask=mask_c, other=1.0) + eps)

    # Normalize and apply affine transform
    x_hat = (x - mean[:, None]) * inv_std[:, None]
    out = x_hat * weight[:, None] + bias[:, None]

    tl.store(out_ptr + offsets, out, mask=mask)


def triton_batch_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                      running_mean: torch.Tensor, running_var: torch.Tensor, eps: float):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_channels = x.shape[1]
    num_elements_per_channel = x.numel() // n_channels

    # Choose block sizes
    BLOCK_SIZE_C = min(triton.next_power_of_2(n_channels), 32)
    BLOCK_SIZE_HW = min(triton.next_power_of_2(num_elements_per_channel), 4096)

    # Grid: (number of channel blocks, number of spatial blocks)
    grid = (
        triton.cdiv(n_channels, BLOCK_SIZE_C),
        triton.cdiv(num_elements_per_channel, BLOCK_SIZE_HW)
    )

    batch_norm_kernel[grid](
        x, weight, bias, running_mean, running_var,
        out, n_channels, num_elements_per_channel, eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for elementwise multiplication and batch normalization.
    ConvTranspose3d and AdaptiveAvgPool3d are kept as native PyTorch ops due to complexity and sufficient optimization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_mul(x, self.scale_factor)
        x = triton_batch_norm(
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            self.batch_norm.eps
        )
        x = self.global_avg_pool(x)
        return x
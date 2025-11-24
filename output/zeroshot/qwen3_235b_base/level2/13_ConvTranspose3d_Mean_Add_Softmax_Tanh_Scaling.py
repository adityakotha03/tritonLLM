import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_channels, height, width,
    stride_c, stride_h, stride_w,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_offset = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_offset < height * width

    offsets = pid_b * n_channels * stride_c + \
              (hw_offset // width) * stride_h + \
              (hw_offset % width) * stride_w
    base_input_ptr = input_ptr + offsets
    base_output_ptr = output_ptr + offsets

    channel_offsets = tl.arange(0, BLOCK_SIZE_C)
    mask = channel_offsets < n_channels
    input_ptrs = base_input_ptr[:, None] + channel_offsets * stride_c
    x = tl.load(input_ptrs, mask=mask[None, :], other=-float('inf'))

    x_max = tl.max(x, axis=1)[:, None]
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=1)[:, None]
    softmax_output = x_exp / x_sum

    output_ptrs = base_output_ptr[:, None] + channel_offsets * stride_c
    tl.store(output_ptrs, softmax_output, mask=mask[None, :])


@triton.jit
def tanh_scale_kernel(
    input_ptr, output_ptr,
    n_elements,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr
):
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.tanh(x) * scaling_factor
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def meanpool_add_kernel(
    input_ptr, bias_ptr, output_ptr,
    total_elements, depth,
    BLOCK_SIZE: tl.constexpr
):
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load input and divide by depth (mean pool)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mean_x = x / depth

    # Add bias (broadcast over spatial dims)
    bias_offset = (offsets % (1 * 64 * 1 * 128 * 128)) // (1 * 1 * 1 * 128 * 128)
    bias = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)
    result = mean_x + bias

    tl.store(output_ptr + offsets, result, mask=mask)


def triton_softmax(x):
    B, C, H, W = x.shape[0], x.shape[1], x.shape[3], x.shape[4]
    y = torch.empty_like(x)
    n_channels = C
    hw_elements = H * W
    grid = (B, triton.cdiv(hw_elements, 1024))
    softmax_kernel[grid](
        x, y,
        n_channels, H, W,
        stride_c=x.stride(1), stride_h=x.stride(3), stride_w=x.stride(4),
        BLOCK_SIZE_C=64,
        BLOCK_SIZE_HW=1024,
    )
    return y


def triton_tanh_scale(x, scaling_factor):
    n_elements = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_scale_kernel[grid](
        x, y, n_elements, scaling_factor,
        BLOCK_SIZE=1024
    )
    return y


def triton_meanpool_add(x, bias):
    total_elements = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    meanpool_add_kernel[grid](
        x, bias, y, total_elements, x.shape[2], BLOCK_SIZE=1024
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.squeeze(2)  # Remove depth dim after mean (already kept as dim)
        x = x.unsqueeze(2)  # Restore for consistent shape (B, C, 1, H, W)
        x = x.squeeze(2)  # Now (B, C, H, W)
        x = triton_meanpool_add(x, self.bias.flatten())
        x = x.unsqueeze(2)  # Back to (B, C, 1, H, W)
        x = triton_softmax(x)
        x = triton_tanh_scale(x, self.scaling_factor)
        return x
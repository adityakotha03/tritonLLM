import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_bias_clamp_scale_div_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, height, width,
    out_channels, out_height, out_width,
    kernel_size, stride, padding, output_padding,
    scaling_factor,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Compute grouped indexing for better load balancing
    num_blocks_m = tl.cdiv(out_height * out_width, BLOCK_SIZE_M)
    num_blocks_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    num_blocks_k = tl.cdiv(in_channels * kernel_size * kernel_size, BLOCK_SIZE_K)
    total_blocks = num_blocks_m * num_blocks_n

    # Rearrange program ID for better load balancing across SMs
    group_size = min(GROUP_SIZE_M, num_blocks_n)
    group_id = pid // group_size
    first_pid_m = group_id * group_size
    group_size_m = min(num_blocks_m - first_pid_m, group_size)
    pid_m = first_pid_m + (pid % group_size)
    pid_n = pid // num_blocks_m

    if pid_m >= num_blocks_m or pid_n >= num_blocks_n:
        return

    # Offset for output tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to shared memory tiles
    x_ptrs = x_ptr + (
        (offs_m // out_width) // stride * input_stride_h +
        ((offs_m % out_width) // stride) * input_stride_w +
        (offs_k // (kernel_size * kernel_size)) * input_stride_c +
        ((offs_k % (kernel_size * kernel_size)) // kernel_size) * input_stride_h +
        ((offs_k % kernel_size)) * input_stride_w -
        padding * (input_stride_h + input_stride_w)
    )
    weight_ptrs = weight_ptr + (
        offs_n[:, None] * weight_stride_c +
        offs_k[None, :] * weight_stride_k +
        ((offs_k % (kernel_size * kernel_size)) // kernel_size) * weight_stride_r +
        (offs_k % kernel_size) * weight_stride_s
    )

    # Load bias
    bias_mask = offs_n < out_channels
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)

    # Accumulator for output
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Convolution loop
    for k in range(0, in_channels * kernel_size * kernel_size, BLOCK_SIZE_K):
        weight = tl.load(weight_ptrs, mask=(offs_k[None, :] < in_channels * kernel_size * kernel_size - k), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < out_height * out_width) & (offs_k[None, :] < in_channels * kernel_size * kernel_size - k), other=0.0)
        acc += tl.dot(x, weight)
        x_ptrs += BLOCK_SIZE_K * input_stride_c
        weight_ptrs += BLOCK_SIZE_K * weight_stride_k

    # Add bias, clamp [0,1], scale, clamp [0,1], divide by scaling factor
    acc += bias[None, :]
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)
    acc = acc * scaling_factor
    acc = tl.minimum(acc, 1.0)
    acc = acc / scaling_factor

    # Output pointers
    output_block_start = pid_m * BLOCK_SIZE_M
    output_h = (output_block_start + tl.arange(0, BLOCK_SIZE_M)) // out_width
    output_w = (output_block_start + tl.arange(0, BLOCK_SIZE_M)) % out_width
    output_ptrs = (
        output_ptr +
        output_h[:, None] * output_stride_h +
        output_w[:, None] * output_stride_w +
        offs_n[None, :] * output_stride_c
    )
    output_mask = (output_h[:, None] < out_height) & (output_w[:, None] < out_width) & (offs_n[None, :] < out_channels)
    tl.store(output_ptrs, acc, mask=output_mask)


def triton_conv_transpose2d_bias_clamp_scale_div(
    x, weight, bias, stride, padding, output_padding, scaling_factor
):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    # Compute output spatial dimensions
    out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Grid configuration
    num_blocks_m = triton.cdiv(out_height * out_width, BLOCK_SIZE_M)
    num_blocks_n = triton.cdiv(out_channels, BLOCK_SIZE_N)
    grid = (num_blocks_m * num_blocks_n,)

    # Strides
    input_strides = x.stride()
    weight_strides = weight.stride()
    output_strides = output.stride()

    # Launch kernel
    _conv_transpose2d_bias_clamp_scale_div_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, height, width,
        out_channels, out_height, out_width,
        kernel_size, stride, padding, output_padding,
        scaling_factor,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return output


class ModelNew(nn.Module):
    """
    Optimized model with fused transposed convolution, bias addition, clamping, scaling, and division using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return triton_conv_transpose2d_bias_clamp_scale_div(
            x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.scaling_factor
        )
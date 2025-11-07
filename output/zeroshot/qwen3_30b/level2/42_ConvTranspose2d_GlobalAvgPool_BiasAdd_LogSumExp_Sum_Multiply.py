import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    bias_ptr,  # Bias pointer
    n_batch, n_in_channels, n_out_channels, height, width, kernel_size,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    out_ch_offset = pid_out_ch * BLOCK_SIZE_OUT

    # Compute output shape
    out_height = height + kernel_size - 1
    out_width = width + kernel_size - 1

    # Load bias if available
    bias = tl.load(bias_ptr + out_ch_offset, mask=pid_out_ch < n_out_channels, other=0.0)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over input channels
    for in_ch in range(0, n_in_channels, 16):
        in_ch_offset = in_ch

        # Load input tile (16 channels, HxW)
        x_tile = tl.load(
            x_ptr + pid_batch * n_in_channels * height * width + in_ch_offset * height * width +
            h_offset * width + w_offset,
            mask=(tl.arange(0, BLOCK_SIZE_H)[:, None] < height - h_offset) &
                 (tl.arange(0, BLOCK_SIZE_W)[None, :] < width - w_offset) &
                 (tl.arange(0, 16)[:, None, None] < n_in_channels - in_ch_offset),
            other=0.0
        )

        # Load kernel tile (16xout_ch, kernel_size x kernel_size)
        w_tile = tl.load(
            w_ptr + in_ch_offset * n_out_channels * kernel_size * kernel_size +
            out_ch_offset * kernel_size * kernel_size +
            (tl.arange(0, 16)[:, None, None] < n_in_channels - in_ch_offset) *
            (tl.arange(0, BLOCK_SIZE_OUT)[None, :, None] < n_out_channels - out_ch_offset) *
            (tl.arange(0, kernel_size)[:, None, None, None] < kernel_size) *
            (tl.arange(0, kernel_size)[None, None, :, None] < kernel_size),
            mask=(tl.arange(0, 16)[:, None, None] < n_in_channels - in_ch_offset) &
                 (tl.arange(0, BLOCK_SIZE_OUT)[None, :, None] < n_out_channels - out_ch_offset) &
                 (tl.arange(0, kernel_size)[:, None, None, None] < kernel_size) &
                 (tl.arange(0, kernel_size)[None, None, :, None] < kernel_size),
            other=0.0
        )

        # Perform conv transpose using strided tiling (manual unrolling)
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                k_val = w_tile[:, :, kh, kw]
                x_val = x_tile[:, :, kh + h_offset, kw + w_offset]
                acc += tl.dot(k_val, x_val, out_dtype=tl.float32)

    # Add bias
    acc = acc + bias

    # Write output
    out_ptr_offset = pid_batch * n_out_channels * out_height * out_width + out_ch_offset * out_height * out_width
    tl.store(
        out_ptr + out_ptr_offset + (h_offset * out_width + w_offset),
        acc,
        mask=(tl.arange(0, BLOCK_SIZE_H)[:, None] < out_height - h_offset) &
             (tl.arange(0, BLOCK_SIZE_W)[None, :] < out_width - w_offset)
    )


@triton.jit
def logsumexp_and_sum_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_batch, n_out_channels, height, width,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W

    # Load data block
    offsets = tl.arange(0, BLOCK_SIZE_H)[:, None] + h_offset
    offsets = offsets * width + tl.arange(0, BLOCK_SIZE_W)[None, :] + w_offset
    offsets = offsets + pid_batch * n_out_channels * height * width + pid_out_ch * height * width
    x = tl.load(x_ptr + offsets, mask=(offsets < n_batch * n_out_channels * height * width), other=-float('inf'))

    # Compute max along channels
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max

    # Sum exp
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)

    # Logsumexp
    logsumexp_val = x_max + tl.log(x_sum)

    # Sum across spatial dims
    final_val = tl.sum(logsumexp_val)

    # Write output
    out_ptr_offset = pid_batch * n_out_channels
    tl.store(out_ptr + out_ptr_offset + pid_out_ch, final_val, mask=pid_out_ch < n_out_channels)


def triton_conv_transpose(x, weight, bias, kernel_size):
    """
    Custom Triton kernel for transposed convolution with fused bias.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape

    out_height = height + kernel_size - 1
    out_width = width + kernel_size - 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Grid and block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_OUT = 32

    # Grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT']),
        triton.cdiv(out_height, meta['BLOCK_SIZE_H']),
        triton.cdiv(out_width, meta['BLOCK_SIZE_W'])
    )

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, out, bias,
        batch_size, in_channels, out_channels, height, width, kernel_size,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_OUT=BLOCK_SIZE_OUT
    )

    return out


def triton_logsumexp_and_sum(x):
    """
    Fused kernel for logsumexp over channel dim and sum over spatial dims.
    """
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size, out_channels, height, width = x.shape

    # Output tensor
    out = torch.empty(batch_size, out_channels, dtype=x.dtype, device=x.device)

    # Block size
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta['BLOCK_SIZE_OUT']),
        triton.cdiv(height, meta['BLOCK_SIZE_H']),
        triton.cdiv(width, meta['BLOCK_SIZE_W'])
    )

    # Launch kernel
    triton_logsumexp_and_sum_kernel[grid](
        x, out, batch_size, out_channels, height, width,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Use Triton kernel for conv transpose + bias
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.bias, self.conv_transpose.kernel_size[0])

        # Fuse logsumexp + spatial sum into one kernel
        x = triton_logsumexp_and_sum(x)

        # Multiply by 10.0
        x = x * 10.0
        return x
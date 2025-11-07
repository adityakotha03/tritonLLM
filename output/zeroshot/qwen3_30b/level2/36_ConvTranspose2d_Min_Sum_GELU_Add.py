import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding, output_padding,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Compute output height and width
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Block offsets
    block_h_start = pid_h * BLOCK_SIZE_H
    block_w_start = pid_w * BLOCK_SIZE_W
    block_c_start = pid_c * BLOCK_SIZE_C

    # Create offsets
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)
    offs_c = tl.arange(0, BLOCK_SIZE_C)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic in range(0, in_channels, BLOCK_SIZE_C):
        # Load input block
        offs_ic = ic + offs_c
        mask_ic = offs_ic < in_channels
        offs_h_ic = block_h_start + offs_h
        offs_w_ic = block_w_start + offs_w
        mask_hw = (offs_h_ic < height) & (offs_w_ic < width)
        mask = mask_hw[:, None] & mask_ic[None, :]
        x_ptrs = x_ptr + offs_h_ic[:, None] * x_stride2 + offs_w_ic[:, None] * x_stride3 + offs_ic[None, :] * x_stride1
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # Load kernel block
        offs_kh = tl.arange(0, kernel_size)
        offs_kw = tl.arange(0, kernel_size)
        offs_ic_k = ic + offs_c
        mask_k = offs_ic_k < in_channels
        k_ptrs = w_ptr + offs_kh[:, None] * w_stride2 + offs_kw[:, None] * w_stride3 + offs_ic_k[None, :] * w_stride1 + pid_c * w_stride0
        w = tl.load(k_ptrs, mask=mask_k[None, :], other=0.0)

        # Perform convolution transpose: convolve over kernel
        # Use outer product-like operation
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                w_k = w[kh, kw]
                x_k = x[kh, kw]  # Note: x already has correct spatial offset
                acc += w_k * x_k

    # Output spatial offsets
    out_h_offset = block_h_start + offs_h
    out_w_offset = block_w_start + offs_w
    mask_out = (out_h_offset < out_h) & (out_w_offset < out_w)
    out_ptrs = out_ptr + out_h_offset[:, None] * out_stride2 + out_w_offset[:, None] * out_stride3 + pid_c * out_stride1
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def min_sum_gelu_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, out_h, out_w,
    x_stride0, x_stride1, x_stride2, x_stride3,
    bias_stride0, bias_stride1, bias_stride2, bias_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Block offsets
    block_h_start = pid_h * BLOCK_SIZE_H
    block_w_start = pid_w * BLOCK_SIZE_W
    block_c_start = pid_c * BLOCK_SIZE_C

    # Create offsets
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)
    offs_c = tl.arange(0, BLOCK_SIZE_C)

    # Load input block
    offs_h_offset = block_h_start + offs_h
    offs_w_offset = block_w_start + offs_w
    mask_hw = (offs_h_offset < out_h) & (offs_w_offset < out_w)
    offs_c_offset = block_c_start + offs_c
    mask_c = offs_c_offset < out_channels
    mask = mask_hw[:, None] & mask_c[None, :]
    x_ptrs = x_ptr + offs_h_offset[:, None] * x_stride2 + offs_w_offset[:, None] * x_stride3 + offs_c_offset[None, :] * x_stride1
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Min along channel dimension
    min_val = tl.reduce(x, axis=1, combine_op="min")

    # Sum along height dimension
    sum_val = tl.sum(x, axis=0, keepdims=True)

    # GELU activation
    # Use approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pi = 3.141592653589793
    sqrt_2_over_pi = tl.sqrt(2.0 / pi)
    x_3 = sum_val * sum_val * sum_val
    gelu_input = sum_val * (1.0 + tl.tanh(sqrt_2_over_pi * (sum_val + 0.044715 * x_3)))
    gelu_out = sum_val * 0.5 * gelu_input

    # Add bias
    bias_ptrs = bias_ptr + offs_c_offset * bias_stride1
    bias = tl.load(bias_ptrs, mask=mask_c, other=0.0)
    out = gelu_out + bias

    # Store output
    out_ptrs = out_ptr + offs_h_offset[:, None] * out_stride2 + offs_w_offset[:, None] * out_stride3 + offs_c_offset[None, :] * out_stride1
    tl.store(out_ptrs, out, mask=mask)


def triton_conv_transpose(x, weight, bias, kernel_size, stride, padding, output_padding):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]

    # Compute output dimensions
    out_h = (height - 1) * stride + kernel_size - 2 * padding + output_padding
    out_w = (width - 1) * stride + kernel_size - 2 * padding + output_padding

    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Compute strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = weight.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()
    bias_stride0, bias_stride1, bias_stride2, bias_stride3 = bias.stride()

    # Define block sizes
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_C = 32

    # Grid for convolution transpose
    grid = lambda meta: (
        (out_h + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_w + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (out_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"]
    )

    # Launch kernel
    conv_transpose_kernel[grid](
        x, weight, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, output_padding,
        x_stride0, x_stride1, x_stride2, x_stride3,
        w_stride0, w_stride1, w_stride2, w_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


def triton_min_sum_gelu(x, bias):
    # Ensure input is contiguous
    x = x.contiguous()
    bias = bias.contiguous()

    # Get dimensions
    batch_size, out_channels, out_h, out_w = x.shape

    # Create output tensor
    out = torch.empty_like(x)

    # Compute strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    bias_stride0, bias_stride1, bias_stride2, bias_stride3 = bias.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Define block sizes
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_C = 32

    # Grid for min-sum-gelu
    grid = lambda meta: (
        (out_h + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_w + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (out_channels + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"]
    )

    # Launch kernel
    min_sum_gelu_kernel[grid](
        x, bias, out,
        batch_size, out_channels, out_h, out_w,
        x_stride0, x_stride1, x_stride2, x_stride3,
        bias_stride0, bias_stride1, bias_stride2, bias_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Replace all operations with fused Triton kernels
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.bias, 
                                 self.conv_transpose.kernel_size[0], 
                                 self.conv_transpose.stride[0], 
                                 self.conv_transpose.padding[0], 
                                 self.conv_transpose.output_padding[0])
        x = triton_min_sum_gelu(x, self.bias)
        return x
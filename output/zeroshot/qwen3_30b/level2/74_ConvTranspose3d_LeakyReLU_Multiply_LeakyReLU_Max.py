import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr,  # Input data pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output data pointer
    bias_ptr,  # Bias pointer
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Thread indexing
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Block indices
    block_d_start = pid_d * BLOCK_D
    block_h_start = pid_h * BLOCK_H
    block_w_start = pid_w * BLOCK_W

    # Output indices
    out_d = block_d_start + tl.arange(0, BLOCK_D)[:, None, None]
    out_h = block_h_start + tl.arange(0, BLOCK_H)[None, :, None]
    out_w = block_w_start + tl.arange(0, BLOCK_W)[None, None, :]

    # Ensure valid indices for output
    out_d_mask = out_d < depth
    out_h_mask = out_h < height
    out_w_mask = out_w < width

    # Compute the actual output dimensions
    out_d_ = (depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_h_ = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w_ = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    # Compute input indices
    in_d = out_d // stride
    in_h = out_h // stride
    in_w = out_w // stride

    # Valid input indices
    in_d_mask = (in_d >= 0) & (in_d < depth)
    in_h_mask = (in_h >= 0) & (in_h < height)
    in_w_mask = (in_w >= 0) & (in_w < width)

    # Combined mask
    valid_mask = out_d_mask & out_h_mask & out_w_mask & in_d_mask & in_h_mask & in_w_mask

    # Channel indices
    c_in_offsets = tl.arange(0, BLOCK_C_IN)[:, None, None]
    c_out_offsets = tl.arange(0, BLOCK_C_OUT)[None, :, None]
    kernel_offsets = tl.arange(0, kernel_size)[:, None, None]

    # Compute input and output offsets
    x_offset = pid_b * in_channels * depth * height * width
    w_offset = pid_c_out * in_channels * kernel_size * kernel_size * kernel_size
    out_offset = pid_b * out_channels * out_d_ * out_h_ * out_w_

    # Load bias if used
    bias = 0.0
    if USE_BIAS:
        bias = tl.load(bias_ptr + pid_c_out * out_channels + tl.arange(0, BLOCK_C_OUT))

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and kernel size
    for c_in in range(0, in_channels, BLOCK_C_IN):
        for k in range(0, kernel_size, BLOCK_C_IN):
            # Compute input channel index
            c_in_idx = c_in + c_in_offsets
            k_idx = k + kernel_offsets

            # Compute input and output indices
            in_d_idx = in_d + k_idx[0] - padding
            in_h_idx = in_h + k_idx[1] - padding
            in_w_idx = in_w + k_idx[2] - padding

            # Valid indices
            valid_in_d = (in_d_idx >= 0) & (in_d_idx < depth)
            valid_in_h = (in_h_idx >= 0) & (in_h_idx < height)
            valid_in_w = (in_w_idx >= 0) & (in_w_idx < width)

            # Final mask
            final_mask = valid_mask & valid_in_d & valid_in_h & valid_in_w

            # Compute input pointer
            in_ptr = x_ptr + x_offset + c_in_idx * depth * height * width
            in_ptr += (in_d_idx * height * width + in_h_idx * width + in_w_idx)
            x_vals = tl.load(in_ptr, mask=final_mask, other=0.0)

            # Compute kernel pointer
            w_ptr_base = w_ptr + w_offset + c_in_idx * kernel_size * kernel_size * kernel_size
            w_ptr_base += (k_idx * kernel_size * kernel_size + k_idx * kernel_size + k_idx)
            w_vals = tl.load(w_ptr_base, mask=final_mask, other=0.0)

            # Compute output
            acc += x_vals * w_vals

    # Apply bias
    acc += bias

    # Store output
    out_ptr_base = out_ptr + out_offset
    out_ptr_base += (out_d * out_h_ * out_w_ + out_h * out_w_ + out_w)
    tl.store(out_ptr_base, acc, mask=valid_mask)


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, negative_slope * x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def max_pool_3d_kernel(
    x_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d_start = pid_d * BLOCK_D
    block_h_start = pid_h * BLOCK_H
    block_w_start = pid_w * BLOCK_W

    d = block_d_start + tl.arange(0, BLOCK_D)[:, None, None]
    h = block_h_start + tl.arange(0, BLOCK_H)[None, :, None]
    w = block_w_start + tl.arange(0, BLOCK_W)[None, None, :]

    d_mask = d < depth
    h_mask = h < height
    w_mask = w < width

    valid_mask = d_mask & h_mask & w_mask

    # Compute input indices
    in_d = d * stride
    in_h = h * stride
    in_w = w * stride

    # Adjust to kernel size
    kernel_d = tl.arange(0, kernel_size)[:, None, None]
    kernel_h = tl.arange(0, kernel_size)[None, :, None]
    kernel_w = tl.arange(0, kernel_size)[None, None, :]

    # Full kernel indices
    k_d = in_d + kernel_d
    k_h = in_h + kernel_h
    k_w = in_w + kernel_w

    # Valid kernel indices
    k_d_mask = k_d < depth
    k_h_mask = k_h < height
    k_w_mask = k_w < width

    final_mask = valid_mask & k_d_mask & k_h_mask & k_w_mask

    # Load input values
    x_offset = pid_b * channels * depth * height * width
    x_ptr_base = x_ptr + x_offset + pid_c * depth * height * width
    x_ptr_base += (k_d * height * width + k_h * width + k_w)
    x_vals = tl.load(x_ptr_base, mask=final_mask, other=-float('inf'))

    # Max reduction
    out = tl.max(x_vals, axis=0)

    # Store output
    out_offset = pid_b * channels * (depth // stride) * (height // stride) * (width // stride)
    out_ptr_base = out_ptr + out_offset + pid_c * (depth // stride) * (height // stride) * (width // stride)
    out_ptr_base += (d * (height // stride) * (width // stride) + h * (width // stride) + w)
    tl.store(out_ptr_base, out, mask=valid_mask)


def triton_conv_transpose_3d(x, weight, bias=None, stride=2, padding=1, output_padding=1):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape

    out_d = (depth - 1) * stride - 2 * padding + kernel_size + output_padding
    out_h = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_w = (width - 1) * stride - 2 * padding + kernel_size + output_padding

    out = torch.empty(batch_size, out_channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    # Use autotune to find optimal block sizes
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_C_OUT"] - 1) // meta["BLOCK_C_OUT"],
        (out_d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    # Launch kernel
    conv_transpose_3d_kernel[grid](
        x, weight, out, bias,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        BLOCK_D=64,
        BLOCK_H=64,
        BLOCK_W=64,
        BLOCK_C_OUT=16,
        BLOCK_C_IN=16,
        USE_BIAS=bias is not None,
    )

    return out


def triton_leaky_relu(x, negative_slope=0.2):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope=negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_multiply(x, y):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    multiply_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_max_pool_3d(x, kernel_size=2, stride=2):
    batch_size, channels, depth, height, width = x.shape
    out_d = (depth + stride - 1) // stride
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride

    out = torch.empty(batch_size, channels, out_d, out_h, out_w, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        batch_size,
        (channels + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (out_d + meta["BLOCK_D"] - 1) // meta["BLOCK_D"],
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    max_pool_3d_kernel[grid](
        x, out,
        batch_size=batch_size,
        channels=channels,
        depth=depth,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        BLOCK_D=64,
        BLOCK_H=64,
        BLOCK_W=64,
        BLOCK_C=16,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        # Use Triton kernels instead of PyTorch ops
        x = triton_conv_transpose_3d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )
        x = triton_leaky_relu(x, negative_slope=0.2)
        x = triton_multiply(x, self.multiplier)
        x = triton_leaky_relu(x, negative_slope=0.2)
        x = triton_max_pool_3d(x, kernel_size=2, stride=2)
        return x
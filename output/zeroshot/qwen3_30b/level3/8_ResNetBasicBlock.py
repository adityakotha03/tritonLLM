import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Kernel size (assumed to be square)
    stride,  # Stride
    pad,  # Padding
    input_stride0, input_stride1, input_stride2, input_stride3,  # Input strides
    weight_stride0, weight_stride1, weight_stride2, weight_stride3,  # Weight strides
    output_stride0, output_stride1, output_stride2, output_stride3,  # Output strides
    bias_stride0,  # Bias stride
    BLOCK_SIZE_H: tl.constexpr,  # Block size for height dimension
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width dimension
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channel dimension
):
    # Get thread and block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Calculate block start indices
    block_start_h = pid_h * BLOCK_SIZE_H
    block_start_w = pid_w * BLOCK_SIZE_W
    block_start_c = pid_c * BLOCK_SIZE_C

    # Define ranges for the block
    h_offsets = block_start_h + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = block_start_w + tl.arange(0, BLOCK_SIZE_W)
    c_offsets = block_start_c + tl.arange(0, BLOCK_SIZE_C)

    # Calculate output dimensions
    out_h = (height + 2 * pad - kernel_size) // stride + 1
    out_w = (width + 2 * pad - kernel_size) // stride + 1

    # Create masks for valid output coordinates
    h_mask = h_offsets < out_h
    w_mask = w_offsets < out_w
    c_mask = c_offsets < out_channels

    # Create combined mask
    mask = h_mask[:, None, None] & w_mask[None, :, None] & c_mask[None, None, :]

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    # Loop over input channels and kernel positions
    for k in range(in_channels):
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Calculate input indices
                input_h = block_start_h * stride + i - pad
                input_w = block_start_w * stride + j - pad

                # Create input masks
                input_h_mask = (input_h >= 0) & (input_h < height)
                input_w_mask = (input_w >= 0) & (input_w < width)

                # Load input value
                input_idx = k * input_stride1 + input_h * input_stride2 + input_w * input_stride3
                input_val = tl.load(input_ptr + input_idx, mask=input_h_mask & input_w_mask, other=0.0)

                # Load weight value
                weight_idx = k * weight_stride1 + i * weight_stride2 + j * weight_stride3
                weight_val = tl.load(weight_ptr + weight_idx, mask=(i < kernel_size) & (j < kernel_size), other=0.0)

                # Accumulate dot product
                acc += input_val[:, None, None] * weight_val[None, None, :]

    # Scale output by input channels
    acc = acc / in_channels

    # Apply bias if present
    if bias_ptr:
        bias_idx = c_offsets
        bias_val = tl.load(bias_ptr + bias_idx, mask=c_mask, other=0.0)
        acc += bias_val[None, None, :]

    # Store output
    output_idx = (pid_h * BLOCK_SIZE_H + h_offsets) * output_stride2 + (pid_w * BLOCK_SIZE_W + w_offsets) * output_stride3 + c_offsets * output_stride1
    tl.store(output_ptr + output_idx, acc, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # Input pointer
    mean_ptr,  # Mean pointer
    var_ptr,  # Variance pointer
    weight_ptr,  # Scale pointer
    bias_ptr,  # Bias pointer
    batch_size,  # Batch size
    channels,  # Channels
    height,  # Height
    width,  # Width
    x_stride0, x_stride1, x_stride2, x_stride3,  # Input strides
    mean_stride0, var_stride0, weight_stride0, bias_stride0,  # Other strides
    BLOCK_SIZE_C: tl.constexpr,  # Block size for channel dimension
):
    pid_c = tl.program_id(0)

    # Block start
    block_start_c = pid_c * BLOCK_SIZE_C
    c_offsets = block_start_c + tl.arange(0, BLOCK_SIZE_C)

    # Valid channel mask
    c_mask = c_offsets < channels

    # Load mean, var, weight, bias
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)

    # Compute normalization
    x_idx = c_offsets * x_stride1
    x_vals = tl.load(x_ptr + x_idx, mask=c_mask, other=0.0)
    normalized = (x_vals - mean) / (tl.sqrt(var + 1e-5))
    out = normalized * weight + bias

    # Store output
    out_idx = c_offsets * x_stride1
    tl.store(x_ptr + out_idx, out, mask=c_mask)


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x, weight, bias=None, stride=1, padding=0):
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Calculate output dimensions
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Initialize output
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Compute strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    weight_stride0, weight_stride1, weight_stride2, weight_stride3 = weight.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()
    bias_stride0 = bias.stride(0) if bias is not None else 0

    # Determine block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32

    # Determine grid
    grid_h = (out_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (grid_h, grid_w, grid_c)

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out, bias, batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        x_stride0, x_stride1, x_stride2, x_stride3,
        weight_stride0, weight_stride1, weight_stride2, weight_stride3,
        out_stride0, out_stride1, out_stride2, out_stride3,
        bias_stride0,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return out


def triton_batch_norm(x, running_mean, running_var, weight, bias, training=True):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, channels, height, width = x.shape

    # Normalize by channel dimension
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    mean_stride0 = running_mean.stride(0)
    var_stride0 = running_var.stride(0)
    weight_stride0 = weight.stride(0)
    bias_stride0 = bias.stride(0)

    # Set block size for channel dimension
    BLOCK_SIZE_C = 32

    # Determine grid
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid = (grid_c,)

    # Launch kernel
    batch_norm_kernel[grid](
        x, running_mean, running_var, weight, bias,
        batch_size, channels, height, width,
        x_stride0, x_stride1, x_stride2, x_stride3,
        mean_stride0, var_stride0, weight_stride0, bias_stride0,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )

    return x


def triton_add(x, y):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        identity = x

        # Conv1 + BN1 + ReLU
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, stride=self.stride, padding=1)
        x = triton_batch_norm(x, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias)
        x = self.relu(x)

        # Conv2 + BN2
        x = triton_conv2d(x, self.conv2.weight, self.conv2.bias, stride=1, padding=1)
        x = triton_batch_norm(x, self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias)

        # Downsample identity
        if self.downsample is not None:
            identity = triton_conv2d(identity, self.downsample[0].weight, self.downsample[0].bias, stride=self.stride, padding=0)
            identity = triton_batch_norm(identity, self.downsample[1].running_mean, self.downsample[1].running_var, self.downsample[1].weight, self.downsample[1].bias)

        # Add and ReLU
        x = triton_add(x, identity)
        x = self.relu(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, channels, H, W)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, 3, 3)
    bias_ptr,  # pointer to bias tensor (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Define block indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute output dimensions
    out_h_start = out_h * BLOCK_SIZE_H
    out_h_end = out_h_start + BLOCK_SIZE_H
    out_w_start = out_w * BLOCK_SIZE_W
    out_w_end = out_w_start + BLOCK_SIZE_W

    # Ensure bounds are respected
    h_mask = (out_h_start < input_height) & (out_h_end <= input_height)
    w_mask = (out_w_start < input_width) & (out_w_end <= input_width)

    # If out of bounds, skip
    if not h_mask or not w_mask:
        return

    # Load input and weight data
    # We assume input is (batch, in_channels, H, W)
    # We use shared memory to cache input patches
    # Input is accessed in a tiled fashion

    # Define tile dimensions
    tile_h = tl.arange(0, BLOCK_SIZE_H)
    tile_w = tl.arange(0, BLOCK_SIZE_W)

    # Compute input coordinates
    input_h = out_h_start + tile_h
    input_w = out_w_start + tile_w

    # Compute valid input indices with padding
    input_h_padded = input_h + padding
    input_w_padded = input_w + padding

    # Compute valid output indices
    valid_h = (input_h_padded < input_height) & (input_h_padded >= 0)
    valid_w = (input_w_padded < input_width) & (input_w_padded >= 0)

    # Load input patches (batch, in_channels, H, W)
    # Use shared memory to reduce global memory access
    # We use a 2D tile of input for each block
    input_batch = batch_idx
    input_offset = input_batch * in_channels * input_height * input_width
    input_tile = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, in_channels), dtype=tl.float32)

    # Load input tile
    input_h_idx = tl.arange(0, BLOCK_SIZE_H)
    input_w_idx = tl.arange(0, BLOCK_SIZE_W)
    input_h_idx = input_h_idx + padding
    input_w_idx = input_w_idx + padding

    # Use masking to avoid out-of-bounds
    valid_input_h = (input_h_idx < input_height)
    valid_input_w = (input_w_idx < input_width)

    # Load input data
    input_h_idx = input_h_idx.to(tl.int32)
    input_w_idx = input_w_idx.to(tl.int32)
    input_offsets = input_h_idx * input_width * in_channels + input_w_idx * in_channels
    input_offsets = input_offsets + tl.arange(0, BLOCK_SIZE_H) * input_width * in_channels + tl.arange(0, BLOCK_SIZE_W) * in_channels
    input_data = tl.load(input_ptr + input_offsets, mask=valid_input_h[:, None] & valid_input_w[None, :], other=0.0)

    # Load weights
    weight_offsets = tl.arange(0, out_channels)[:, None] * in_channels * kernel_size * kernel_size + \
                     tl.arange(0, in_channels)[None, :] * kernel_size * kernel_size + \
                     tl.arange(0, kernel_size)[None, :, None] * kernel_size + \
                     tl.arange(0, kernel_size)[None, None, :]
    weight_data = tl.load(weight_ptr + weight_offsets, mask=tl.all(weight_offsets >= 0), other=0.0)

    # Compute output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for i in range(out_channels):
        for j in range(in_channels):
            for k in range(kernel_size):
                for l in range(kernel_size):
                    h = input_h_padded - padding + k
                    w = input_w_padded - padding + l
                    if h < input_height and w < input_width:
                        input_val = input_data[h, w, j]
                        weight_val = weight_data[i, j, k, l]
                        output = output + input_val * weight_val
        output = output + tl.load(bias_ptr + i, mask=tl.ones(1), other=0.0)

    # Store output
    output_offset = batch_idx * out_channels * BLOCK_SIZE_H * BLOCK_SIZE_W + out_h * BLOCK_SIZE_H * BLOCK_SIZE_W + out_w * BLOCK_SIZE_W
    tl.store(output_ptr + output_offset, output, mask=valid_h[:, None] & valid_w[None, :])


@triton.jit
def conv2d_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Define block indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute output dimensions
    out_h_start = out_h * BLOCK_SIZE_H
    out_h_end = out_h_start + BLOCK_SIZE_H
    out_w_start = out_w * BLOCK_SIZE_W
    out_w_end = out_w_start + BLOCK_SIZE_W

    # Bounds check
    h_mask = (out_h_start < input_height) & (out_h_end <= input_height)
    w_mask = (out_w_start < input_width) & (out_w_end <= input_width)
    if not h_mask or not w_mask:
        return

    # Define tile dimensions
    tile_h = tl.arange(0, BLOCK_SIZE_H)
    tile_w = tl.arange(0, BLOCK_SIZE_W)

    # Compute input coordinates
    input_h = out_h_start + tile_h
    input_w = out_w_start + tile_w

    # Apply padding
    input_h_padded = input_h + padding
    input_w_padded = input_w + padding

    # Valid input bounds
    valid_h = (input_h_padded < input_height) & (input_h_padded >= 0)
    valid_w = (input_w_padded < input_width) & (input_w_padded >= 0)

    # Load input
    input_offsets = input_h_padded * input_width * in_channels + input_w_padded * in_channels
    input_data = tl.load(input_ptr + input_offsets, mask=valid_h[:, None] & valid_w[None, :], other=0.0)

    # Load weights
    weight_offsets = tl.arange(0, out_channels)[:, None] * in_channels * kernel_size * kernel_size + \
                     tl.arange(0, in_channels)[None, :] * kernel_size * kernel_size + \
                     tl.arange(0, kernel_size)[None, :, None] * kernel_size + \
                     tl.arange(0, kernel_size)[None, None, :]
    weight_data = tl.load(weight_ptr + weight_offsets, mask=tl.all(weight_offsets >= 0), other=0.0)

    # Compute output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for i in range(out_channels):
        for j in range(in_channels):
            for k in range(kernel_size):
                for l in range(kernel_size):
                    h = input_h_padded - padding + k
                    w = input_w_padded - padding + l
                    if h < input_height and w < input_width:
                        input_val = input_data[h, w, j]
                        weight_val = weight_data[i, j, k, l]
                        output = output + input_val * weight_val
        output = output + tl.load(bias_ptr + i, mask=tl.ones(1), other=0.0)

    # Apply ReLU activation
    output = tl.where(output > 0, output, 0.0)

    # Store output
    output_offset = batch_idx * out_channels * BLOCK_SIZE_H * BLOCK_SIZE_W + out_h * BLOCK_SIZE_H * BLOCK_SIZE_W + out_w * BLOCK_SIZE_W
    tl.store(output_ptr + output_offset, output, mask=valid_h[:, None] & valid_w[None, :])


@triton.jit
def batch_norm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    running_mean_ptr,
    running_var_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles a slice of the batch
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Define tile
    tile = tl.arange(0, BLOCK_SIZE)
    mask = tile < channels

    # Load input
    input_data = tl.load(input_ptr + batch_idx * channels * input_height * input_width + channel_idx * input_height * input_width, mask=mask, other=0.0)

    # Load gamma and beta
    gamma_val = tl.load(gamma_ptr + channel_idx, mask=mask, other=1.0)
    beta_val = tl.load(beta_ptr + channel_idx, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(input_data, axis=0) / input_height / input_width
    var = tl.sum((input_data - mean) ** 2, axis=0) / input_height / input_width

    # Normalize
    std = tl.sqrt(var + epsilon)
    normalized = (input_data - mean) / std

    # Apply scale and shift
    output_data = gamma_val * normalized + beta_val

    # Store output
    tl.store(output_ptr + batch_idx * channels * input_height * input_width + channel_idx * input_height * input_width, output_data, mask=mask)


@triton.jit
def max_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Define block indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute output dimensions
    out_h_start = out_h * BLOCK_SIZE_H
    out_h_end = out_h_start + BLOCK_SIZE_H
    out_w_start = out_w * BLOCK_SIZE_W
    out_w_end = out_w_start + BLOCK_SIZE_W

    # Bounds check
    h_mask = (out_h_start < input_height) & (out_h_end <= input_height)
    w_mask = (out_w_start < input_width) & (out_w_end <= input_width)
    if not h_mask or not w_mask:
        return

    # Define tile dimensions
    tile_h = tl.arange(0, BLOCK_SIZE_H)
    tile_w = tl.arange(0, BLOCK_SIZE_W)

    # Compute input coordinates
    input_h = out_h_start + tile_h
    input_w = out_w_start + tile_w

    # Compute input indices with padding
    input_h_padded = input_h + padding
    input_w_padded = input_w + padding

    # Valid input bounds
    valid_h = (input_h_padded < input_height) & (input_h_padded >= 0)
    valid_w = (input_w_padded < input_width) & (input_w_padded >= 0)

    # Load input
    input_offsets = input_h_padded * input_width * in_channels + input_w_padded * in_channels
    input_data = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width + input_offsets, mask=valid_h[:, None] & valid_w[None, :], other=0.0)

    # Find max in kernel
    max_val = tl.max(input_data, axis=(0, 1))

    # Store output
    output_offset = batch_idx * in_channels * BLOCK_SIZE_H * BLOCK_SIZE_W + out_h * BLOCK_SIZE_H * BLOCK_SIZE_W + out_w * BLOCK_SIZE_W
    tl.store(output_ptr + output_offset, max_val, mask=valid_h[:, None] & valid_w[None, :])


def triton_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size=3,
    stride=1,
    padding=1,
    BLOCK_SIZE_H=16,
    BLOCK_SIZE_W=16,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda and bias_tensor.is_cuda, "All tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    bias_tensor = bias_tensor.contiguous()

    # Output dimensions
    out_height = (input_height + 2 * padding - kernel_size) // stride + 1
    out_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Allocate output
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    # Define grid
    grid = lambda meta: (
        (batch_size,),
        ((input_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,),
        ((input_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,),
    )

    # Launch kernel
    conv2d_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )

    return output


def triton_conv2d_relu(
    input_tensor,
    weight_tensor,
    bias_tensor,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size=3,
    stride=1,
    padding=1,
    BLOCK_SIZE_H=16,
    BLOCK_SIZE_W=16,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda and bias_tensor.is_cuda, "All tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    bias_tensor = bias_tensor.contiguous()

    # Output dimensions
    out_height = (input_height + 2 * padding - kernel_size) // stride + 1
    out_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Allocate output
    output = torch.empty(
        (batch_size, out_channels, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    # Define grid
    grid = lambda meta: (
        (batch_size,),
        ((input_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,),
        ((input_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,),
    )

    # Launch kernel
    conv2d_relu_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )

    return output


def triton_batch_norm(
    input_tensor,
    gamma_tensor,
    beta_tensor,
    running_mean_tensor,
    running_var_tensor,
    batch_size,
    channels,
    input_height,
    input_width,
    epsilon=1e-5,
    BLOCK_SIZE=128,
):
    assert input_tensor.is_cuda and gamma_tensor.is_cuda and beta_tensor.is_cuda, "All tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    gamma_tensor = gamma_tensor.contiguous()
    beta_tensor = beta_tensor.contiguous()

    # Output tensor
    output = torch.empty_like(input_tensor)

    # Define grid
    grid = lambda meta: ((batch_size,), (channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    batch_norm_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        gamma_tensor.data_ptr(),
        beta_tensor.data_ptr(),
        running_mean_tensor.data_ptr(),
        running_var_tensor.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        input_height,
        input_width,
        epsilon,
        BLOCK_SIZE,
    )

    return output


def triton_max_pool2d(
    input_tensor,
    batch_size,
    in_channels,
    input_height,
    input_width,
    kernel_size=2,
    stride=2,
    padding=0,
    BLOCK_SIZE_H=16,
    BLOCK_SIZE_W=16,
):
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    input_tensor = input_tensor.contiguous()

    # Output dimensions
    out_height = (input_height + 2 * padding - kernel_size) // stride + 1
    out_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Allocate output
    output = torch.empty(
        (batch_size, in_channels, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    # Define grid
    grid = lambda meta: (
        (batch_size,),
        ((input_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,),
        ((input_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,),
    )

    # Launch kernel
    max_pool2d_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()
        self.stages = stages
        self.block_widths = block_widths

        layers = []
        current_channels = input_channels

        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers)

        # Final fully connected layer
        self.fc = nn.Linear(block_widths[-1], output_classes)

    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            triton_conv2d_relu(
                weight_tensor=None,  # Will be passed in forward
                bias_tensor=None,
                input_tensor=None,
                batch_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                input_height=224,
                input_width=224,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            triton_batch_norm(
                input_tensor=None,
                gamma_tensor=None,
                beta_tensor=None,
                running_mean_tensor=None,
                running_var_tensor=None,
                batch_size=1,
                channels=out_channels,
                input_height=224,
                input_width=224,
            ),
            triton_max_pool2d(
                input_tensor=None,
                batch_size=1,
                in_channels=out_channels,
                input_height=224,
                input_width=224,
                kernel_size=2,
                stride=2,
                padding=0,
            ),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x
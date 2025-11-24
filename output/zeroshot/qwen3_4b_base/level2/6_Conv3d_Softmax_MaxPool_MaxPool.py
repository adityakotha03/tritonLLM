import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, d, h, w)
    weight_ptr,  # pointer to convolution weights (out_channels, in_channels, k, k, k)
    output_ptr,  # pointer to output tensor (batch, out_channels, d', h', w')
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute output dimensions
    d_out = depth + pad_d * 2 - kernel_size
    h_out = height + pad_h * 2 - kernel_size
    w_out = width + pad_w * 2 - kernel_size

    # Block indices
    block_id_d = tl.program_id(0)
    block_id_h = tl.program_id(1)
    block_id_w = tl.program_id(2)

    # Compute block boundaries
    d_start = block_id_d * BLOCK_SIZE_D
    h_start = block_id_h * BLOCK_SIZE_H
    w_start = block_id_w * BLOCK_SIZE_W

    # Compute output indices
    d_out_idx = tl.arange(0, BLOCK_SIZE_D)
    h_out_idx = tl.arange(0, BLOCK_SIZE_H)
    w_out_idx = tl.arange(0, BLOCK_SIZE_W)

    # Create offsets for input and output
    d_offset = d_out_idx + d_start
    h_offset = h_out_idx + h_start
    w_offset = w_out_idx + w_start

    # Compute valid input indices with padding
    d_in = d_offset - pad_d
    h_in = h_offset - pad_h
    w_in = w_offset - pad_w

    # Mask for valid input indices
    d_mask = (d_in >= 0) & (d_in < depth)
    h_mask = (h_in >= 0) & (h_in < height)
    w_mask = (w_in >= 0) & (w_in < width)

    # Combine masks
    valid_mask = d_mask & h_mask & w_mask

    # Load input data (batch, in_channels, d, h, w)
    # We assume input is batched and processed per batch
    batch_idx = tl.arange(0, batch_size)
    in_channel_idx = tl.arange(0, in_channels)
    input_offsets = (batch_idx[:, None] * depth * height * width +
                     in_channel_idx[:, None] * height * width +
                     d_in[None, :] * height * width +
                     h_in[None, :] * width +
                     w_in[None, :])

    # Load input values with masking
    input_vals = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)

    # Load weights (out_channels, in_channels, k, k, k)
    weight_offsets = (tl.arange(0, out_channels)[None, :] *
                      in_channels * kernel_size * kernel_size * kernel_size +
                      in_channel_idx[:, None] * kernel_size * kernel_size * kernel_size +
                      tl.arange(0, kernel_size)[None, :, None, None] *
                      kernel_size * kernel_size +
                      tl.arange(0, kernel_size)[None, :, None, :] *
                      kernel_size +
                      tl.arange(0, kernel_size)[None, :, :, None])

    # Weight values
    weights = tl.load(weight_ptr + weight_offsets, mask=valid_mask, other=0.0)

    # Compute output for each output channel
    output_vals = tl.zeros((out_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for oc in tl.arange(0, out_channels):
        # Compute output value for each valid position
        output_channel = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        for ic in tl.arange(0, in_channels):
            # Convolution over spatial dimensions
            for kd in tl.arange(0, kernel_size):
                for kh in tl.arange(0, kernel_size):
                    for kw in tl.arange(0, kernel_size):
                        d_in_idx = d_in + kd
                        h_in_idx = h_in + kh
                        w_in_idx = w_in + kw
                        valid_d = (d_in_idx >= 0) & (d_in_idx < depth)
                        valid_h = (h_in_idx >= 0) & (h_in_idx < height)
                        valid_w = (w_in_idx >= 0) & (w_in_idx < width)
                        valid_pos = valid_d & valid_h & valid_w
                        if valid_pos.any():
                            # Load input at (d_in_idx, h_in_idx, w_in_idx)
                            input_val = tl.load(input_ptr + (batch_idx[:, None] * depth * height * width +
                                                             ic * height * width +
                                                             d_in_idx * height * width +
                                                             h_in_idx * width +
                                                             w_in_idx),
                                                mask=valid_pos, other=0.0)
                            # Load weight
                            weight_val = tl.load(weight_ptr + (oc * in_channels * kernel_size * kernel_size * kernel_size +
                                                               ic * kernel_size * kernel_size * kernel_size +
                                                               kd * kernel_size * kernel_size +
                                                               kh * kernel_size +
                                                               kw),
                                                mask=valid_pos, other=0.0)
                            output_channel += input_val * weight_val
        output_vals[oc] = output_channel

    # Store output
    output_offsets = (batch_idx[:, None] * d_out * h_out * w_out +
                      tl.arange(0, out_channels)[None, :] * h_out * w_out +
                      h_out_idx[:, None] * w_out +
                      w_out_idx)
    tl.store(output_ptr + output_offsets, output_vals, mask=valid_mask)


@triton.jit
def softmax_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, d, h, w)
    output_ptr,  # pointer to output tensor (batch, channels, d, h, w)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of (depth, height, width)
    block_id = tl.program_id(0)
    block_start_d = block_id // (height * width)
    block_start_h = (block_id % (height * width)) // width
    block_start_w = block_id % width

    d_idx = tl.arange(0, depth)
    h_idx = tl.arange(0, height)
    w_idx = tl.arange(0, width)

    # Create full grid of indices
    d_offsets = d_idx + block_start_d
    h_offsets = h_idx + block_start_h
    w_offsets = w_idx + block_start_w

    # Mask for valid indices
    d_mask = (d_offsets >= 0) & (d_offsets < depth)
    h_mask = (h_offsets >= 0) & (h_offsets < height)
    w_mask = (w_offsets >= 0) & (w_offsets < width)
    valid_mask = d_mask & h_mask & w_mask

    # Load input values
    x_vals = tl.load(x_ptr + (d_offsets[:, None] * height * width + h_offsets[:, None] * width + w_offsets), mask=valid_mask, other=0.0)

    # Compute logsumexp over channels
    # For each spatial position, compute softmax over channels
    channel_sum = tl.sum(x_vals, axis=1, keepdim=True)
    # Subtract max for numerical stability
    max_val = tl.max(x_vals, axis=1, keepdim=True)
    x_stable = x_vals - max_val
    exp_x = tl.exp(x_stable)
    softmax_vals = exp_x / tl.sum(exp_x, axis=1, keepdim=True)

    # Store output
    tl.store(output_ptr + (d_offsets[:, None] * height * width + h_offsets[:, None] * width + w_offsets),
             softmax_vals, mask=valid_mask)


@triton.jit
def max_pool3d_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, d, h, w)
    output_ptr,  # pointer to output tensor (batch, channels, d', h', w')
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    pool_kernel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    d_out = (depth - pool_kernel_size + 1)
    h_out = (height - pool_kernel_size + 1)
    w_out = (width - pool_kernel_size + 1)

    # Block indices
    block_id = tl.program_id(0)
    block_start_d = block_id // (h_out * w_out)
    block_start_h = (block_id % (h_out * w_out)) // w_out
    block_start_w = block_id % w_out

    d_idx = tl.arange(0, pool_kernel_size)
    h_idx = tl.arange(0, pool_kernel_size)
    w_idx = tl.arange(0, pool_kernel_size)

    # Compute spatial indices
    d_offsets = d_idx + block_start_d
    h_offsets = h_idx + block_start_h
    w_offsets = w_idx + block_start_w

    # Mask for valid indices
    d_mask = (d_offsets >= 0) & (d_offsets < depth)
    h_mask = (h_offsets >= 0) & (h_offsets < height)
    w_mask = (w_offsets >= 0) & (w_offsets < width)
    valid_mask = d_mask & h_mask & w_mask

    # Load input values
    x_vals = tl.load(x_ptr + (d_offsets[:, None] * height * width + h_offsets[:, None] * width + w_offsets),
                     mask=valid_mask, other=0.0)

    # Compute max over kernel
    max_val = tl.max(x_vals, axis=(1, 2, 3), keepdim=True)
    # Store output
    tl.store(output_ptr + (d_offsets[:, None] * h_out * w_out + h_offsets[:, None] * w_out + w_offsets),
             max_val, mask=valid_mask)


def triton_conv3d(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    batch_size: int,
    in_channels: int,
    out_channels: int,
    depth: int,
    height: int,
    width: int,
    kernel_size: int,
    pad_d: int = 1,
    pad_h: int = 1,
    pad_w: int = 1,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    # Output dimensions
    d_out = depth + pad_d * 2 - kernel_size
    h_out = height + pad_h * 2 - kernel_size
    w_out = width + pad_w * 2 - kernel_size

    # Allocate output
    output = torch.empty(
        (batch_size, out_channels, d_out, h_out, w_out),
        dtype=torch.float32,
        device=input_tensor.device
    )

    # Define block sizes
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8

    # Grid dimensions
    grid = lambda meta: (
        ((depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D),
        ((height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H),
        ((width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W),
    )

    # Launch kernel
    conv3d_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        depth,
        height,
        width,
        kernel_size,
        pad_d,
        pad_h,
        pad_w,
        BLOCK_SIZE_D,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    return output


def triton_softmax(
    x: torch.Tensor,
    dim: int = 1,
):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    output = torch.empty_like(x)
    # Use softmax over dim=1 (channels)
    batch_size, channels, d, h, w = x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: ((d * h * w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_kernel[
        grid
    ](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        d,
        h,
        w,
        BLOCK_SIZE,
    )
    return output


def triton_max_pool3d(
    x: torch.Tensor,
    pool_kernel_size: int,
):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, channels, d, h, w = x.shape
    d_out = (d - pool_kernel_size + 1)
    h_out = (h - pool_kernel_size + 1)
    w_out = (w - pool_kernel_size + 1)

    output = torch.empty(
        (batch_size, channels, d_out, h_out, w_out),
        dtype=torch.float32,
        device=x.device
    )

    BLOCK_SIZE = 16
    grid = lambda meta: ((d_out * h_out * w_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool3d_kernel[
        grid
    ](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        d,
        h,
        w,
        pool_kernel_size,
        BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        # Initialize weights for 3D convolution
        self.conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size).cuda()
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        # 1. Conv3d
        x = triton_conv3d(
            x, self.conv_weight,
            batch_size=x.shape[0],
            in_channels=x.shape[1],
            out_channels=self.conv_weight.shape[0],
            depth=x.shape[2],
            height=x.shape[3],
            width=x.shape[4],
            kernel_size=kernel_size,
            pad_d=1,
            pad_h=1,
            pad_w=1
        )

        # 2. Softmax over channels
        x = triton_softmax(x, dim=1)

        # 3. Max pooling
        x = triton_max_pool3d(x, self.pool_kernel_size)

        # 4. Max pooling again
        x = triton_max_pool3d(x, self.pool_kernel_size)

        return x
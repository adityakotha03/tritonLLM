import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # pointer to conv weight (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # pointer to bias (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    out_height = (input_height + 2 * pad_h - kernel_size) // stride_h + 1
    out_width = (input_width + 2 * pad_w - kernel_size) // stride_w + 1

    # Get current block and thread indices
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute the starting position in the output
    out_h_start = out_h * BLOCK_SIZE
    out_w_start = out_w * BLOCK_SIZE

    # Compute the corresponding input position
    input_h_start = out_h_start * stride_h - pad_h
    input_w_start = out_w_start * stride_w - pad_w

    # Define the range of threads in this block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Load input and weight data
    # Input: (batch, in_channels, H, W)
    # We load input in a tiled fashion across spatial dimensions
    # We use shared memory to cache input patches
    # But in this kernel, we will do direct access with masking

    # We will process one output position per block
    # For each thread in the block, we compute a small patch of input
    # We assume input is padded with zeros

    # For each output channel
    for oc in tl.arange(0, out_channels):
        # Load weight for this output channel
        weight = tl.load(weight_ptr + oc * (in_channels * kernel_size * kernel_size), mask=tl.arange(0, in_channels * kernel_size * kernel_size) < in_channels * kernel_size * kernel_size, other=0.0)
        weight = weight.reshape(in_channels, kernel_size, kernel_size)

        # Load bias
        bias_val = 0.0
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc, mask=tl.arange(0, 1) < 1, other=0.0)

        # Compute output for this output channel
        out_val = 0.0
        for ic in tl.arange(0, in_channels):
            for kh in tl.arange(0, kernel_size):
                for kw in tl.arange(0, kernel_size):
                    # Compute input position
                    ih = input_h_start + kh
                    iw = input_w_start + kw
                    # Check bounds
                    ih_mask = (ih >= 0) & (ih < input_height)
                    iw_mask = (iw >= 0) & (iw < input_width)
                    mask_valid = ih_mask & iw_mask
                    # Load input value
                    input_val = 0.0
                    if mask_valid:
                        input_val = tl.load(input_ptr + batch_idx * in_channels * input_height * input_width + ic * input_height * input_width + ih * input_width + iw, mask=mask_valid, other=0.0)
                    # Accumulate contribution
                    out_val += input_val * weight[ic, kh, kw]
        out_val += bias_val
        # Store result
        tl.store(output_ptr + batch_idx * out_channels * out_height * out_width + oc * out_height * out_width + out_h_start * out_width + out_w_start, out_val, mask=mask)


@triton.jit
def avg_pool_kernel(
    input_ptr,  # pointer to input (batch, channels, H, W)
    output_ptr,  # pointer to output (batch, channels, H_out, W_out)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    pool_kernel_size: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    out_height = (input_height + pool_kernel_size - 1) // pool_kernel_size
    out_width = (input_width + pool_kernel_size - 1) // pool_kernel_size

    # Each block handles one output position
    batch_idx = tl.program_id(0)
    out_h = tl.program_id(1)
    out_w = tl.program_id(2)

    # Compute the spatial range for this output position
    h_start = out_h * stride_h
    w_start = out_w * stride_w

    # Define thread offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Compute the sum over the pooling window
    sum_val = 0.0
    count = 0
    for ph in tl.arange(0, pool_kernel_size):
        for pw in tl.arange(0, pool_kernel_size):
            ih = h_start + ph
            iw = w_start + pw
            # Check bounds
            ih_valid = (ih >= 0) & (ih < input_height)
            iw_valid = (iw >= 0) & (iw < input_width)
            valid = ih_valid & iw_valid
            if valid:
                val = tl.load(input_ptr + batch_idx * channels * input_height * input_width + tl.arange(0, channels) * input_height * input_width + ih * input_width + iw, mask=valid, other=0.0)
                sum_val += val
                count += 1
    # Average value
    avg_val = sum_val / count if count > 0 else 0.0
    # Store result
    tl.store(output_ptr + batch_idx * channels * out_height * out_width + tl.arange(0, channels) * out_height * out_width + out_h * out_width + out_w, avg_val, mask=mask)


@triton.jit
def sigmoid_kernel(
    input_ptr,  # pointer to input (batch, channels, H, W)
    output_ptr,  # pointer to output (batch, channels, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles a block of output
    batch_idx = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < BLOCK_SIZE

    # Load input values
    input_val = 0.0
    for i in tl.arange(0, channels):
        for o in offsets:
            # Compute position
            pos = batch_idx * channels * height * width + i * height * width + h * width + w
            input_val = tl.load(input_ptr + pos, mask=mask, other=0.0)
            # Apply sigmoid: 1 / (1 + exp(-x))
            exp_val = tl.exp(-input_val)
            sigmoid_val = 1.0 / (1.0 + exp_val)
            tl.store(output_ptr + pos, sigmoid_val, mask=mask)
    # Note: This is a simplified version. In practice, we should process one element per thread.
    # We restructure below for correctness and performance.

    # Corrected: each thread handles one element
    # We use a different indexing
    for i in tl.arange(0, channels):
        for h_idx in tl.arange(0, height):
            for w_idx in tl.arange(0, width):
                pos = batch_idx * channels * height * width + i * height * width + h_idx * width + w_idx
                input_val = tl.load(input_ptr + pos, mask=tl.arange(0, 1) < 1, other=0.0)
                exp_val = tl.exp(-input_val)
                sigmoid_val = 1.0 / (1.0 + exp_val)
                tl.store(output_ptr + pos, sigmoid_val, mask=tl.arange(0, 1) < 1)


@triton.jit
def sum_kernel(
    input_ptr,  # pointer to input (batch, channels, H, W)
    output_ptr,  # pointer to output (scalar)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block handles one batch
    batch_idx = tl.program_id(0)
    total = 0.0
    for i in tl.arange(0, channels):
        for h in tl.arange(0, height):
            for w in tl.arange(0, width):
                pos = batch_idx * channels * height * width + i * height * width + h * width + w
                val = tl.load(input_ptr + pos, mask=tl.arange(0, 1) < 1, other=0.0)
                total += val
    tl.store(output_ptr + batch_idx, total, mask=tl.arange(0, 1) < 1)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pad_h: int = 1,
    pad_w: int = 1,
    stride_h: int = 1,
    stride_w: int = 1,
):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Output dimensions
    out_height = (height + 2 * pad_h - kernel_size) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_size) // stride_w + 1

    # Allocate output
    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Grid size
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        BLOCK_SIZE=128,
    )
    return output


def triton_avg_pool(
    x: torch.Tensor,
    pool_kernel_size: int,
    stride_h: int = 1,
    stride_w: int = 1,
):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, channels, height, width = x.shape

    out_height = (height + pool_kernel_size - 1) // pool_kernel_size
    out_width = (width + pool_kernel_size - 1) // pool_kernel_size

    output = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (out_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    avg_pool_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        height,
        width,
        pool_kernel_size,
        stride_h,
        stride_w,
        BLOCK_SIZE=128,
    )
    return output


def triton_sigmoid(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    output = torch.empty_like(x)
    grid = lambda meta: (
        (x.shape[0] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (x.shape[2] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (x.shape[3] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )
    sigmoid_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.shape[3],
        BLOCK_SIZE=128,
    )
    return output


def triton_sum(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    output = torch.empty(batch_size, device=x.device, dtype=x.dtype)
    grid = lambda meta: ((batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sum_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE=128,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

    def forward(self, x):
        # Replace Conv2d with custom Triton kernel
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        conv_output = triton_conv2d(x, conv_weight, conv_bias, pad_h=1, pad_w=1, stride_h=1, stride_w=1)
        # Replace AvgPool with custom kernel
        pool_output = triton_avg_pool(conv_output, pool_kernel_size=pool_kernel_size)
        # Replace Sigmoid with custom kernel
        sig_output = triton_sigmoid(pool_output)
        # Replace sum with custom kernel
        result = triton_sum(sig_output)
        return result
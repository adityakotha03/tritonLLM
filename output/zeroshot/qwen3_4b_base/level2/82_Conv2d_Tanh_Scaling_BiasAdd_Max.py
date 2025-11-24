import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # pointer to conv2d weight (out_channels, in_channels, k, k)
    bias_ptr,  # pointer to bias (out_channels)
    output_ptr,  # pointer to output (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    k: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Compute output dimensions
    out_h = (height + 2 * pad_h - k) // 1
    out_w = (width + 2 * pad_w - k) // 1

    # Current block and thread indices
    block_id = tl.program_id(0)
    block_h = block_id // (out_h // BLOCK_SIZE_H)
    block_w = block_id % (out_w // BLOCK_SIZE_W)

    # Thread indices within the block
    h = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = tl.program_id(2) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Clip to valid output range
    h = h % out_h
    w = w % out_w

    # Load input and weight
    # Input: (batch, in_channels, H, W)
    # We process one batch at a time
    batch_idx = 0
    input_h = tl.arange(0, in_channels)
    input_w = tl.arange(0, in_channels)

    # Weight: (out_channels, in_channels, k, k)
    weight_h = tl.arange(0, k)
    weight_w = tl.arange(0, k)

    # Compute valid input indices for convolution
    input_h_start = h - pad_h
    input_h_end = h + pad_h + 1
    input_w_start = w - pad_w
    input_w_end = w + pad_w + 1

    # Compute valid input indices (clamped)
    input_h_start = tl.maximum(input_h_start, 0)
    input_h_end = tl.minimum(input_h_end, height)
    input_w_start = tl.maximum(input_w_start, 0)
    input_w_end = tl.minimum(input_w_end, width)

    # Load input features (batch, in_channels, H, W)
    # We assume input is stored as (batch, in_channels, H, W)
    # We load input in a tiled fashion
    input_batch = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), batch_idx, dtype=tl.int32)
    input_offsets = input_batch[:, :, None] * (height * width) + input_h_start[:, :, None] * width + input_w_start[:, :, None]
    input_values = tl.load(input_ptr + input_offsets, mask=(input_h_start < height) & (input_h_end > 0) & (input_w_start < width) & (input_w_end > 0), other=0.0)

    # Weight loading: (out_channels, in_channels, k, k)
    # We tile over output channels and input channels
    out_ch = tl.arange(0, out_channels)
    in_ch = tl.arange(0, in_channels)
    weight_offsets = out_ch[:, None, None, None] * (in_channels * k * k) + in_ch[:, None, None, None] * (k * k) + weight_h[:, None, None] * k + weight_w[:, None]
    weights = tl.load(weight_ptr + weight_offsets, mask=(weight_h < k) & (weight_w < k), other=0.0)

    # Compute convolution sum
    # For each output channel, compute dot product over input and weight
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for i in range(out_channels):
        # Compute dot product over input and weight
        # We use a reduction over spatial and channel dimensions
        out_val = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
        for j in range(in_channels):
            for k in range(k):
                for l in range(k):
                    # Compute valid input indices
                    ih = h + k - l
                    iw = w + l - k
                    # Clamp to valid range
                    ih = tl.maximum(ih, 0)
                    iw = tl.maximum(iw, 0)
                    ih = tl.minimum(ih, height)
                    iw = tl.minimum(iw, width)
                    # Load input value
                    val = tl.load(input_ptr + (batch_idx * in_channels * height * width) + j * height * width + ih * width + iw, mask=(ih < height) & (iw < width), other=0.0)
                    # Load weight
                    w_val = tl.load(weight_ptr + (i * in_channels * k * k) + j * k * k + k * k + l, mask=(l < k), other=0.0)
                    out_val += val * w_val
        output[:, :, i] = out_val

    # Apply tanh activation
    output = tl.tanh(output)

    # Scale
    scale = tl.full((1,), 2.0, dtype=tl.float32)
    output = output * scale

    # Add bias
    bias = tl.load(bias_ptr + tl.arange(0, out_channels), mask=tl.arange(0, out_channels) < out_channels, other=0.0)
    output = output + bias[:, None, None]

    # Max pooling: (k, k) kernel
    pool_h = tl.arange(0, k)
    pool_w = tl.arange(0, k)
    pool_output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for i in range(k):
        for j in range(k):
            ph = h + i
            pw = w + j
            ph = tl.maximum(ph, 0)
            pw = tl.maximum(pw, 0)
            ph = tl.minimum(ph, out_h)
            pw = tl.minimum(pw, out_w)
            # Load pooled value
            val = tl.load(output_ptr + (block_id * out_h * out_w) + ph * out_w + pw, mask=(ph < out_h) & (pw < out_w), other=0.0)
            pool_output += val
    pool_output = pool_output / (k * k)

    # Store result
    tl.store(output_ptr + (block_id * out_h * out_w) + h * out_w + w, pool_output, mask=(h < out_h) & (w < out_w))


@triton.jit
def conv_tanh_scale_bias_pool_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    k: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Output dimensions
    out_h = (height + 2 * pad_h - k) // 1
    out_w = (width + 2 * pad_w - k) // 1

    # Block and thread indices
    block_id = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = tl.program_id(2) * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Clip to valid range
    h = h % out_h
    w = w % out_w

    # Load input (batch, in_channels, H, W)
    batch_idx = 0
    input_batch = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), batch_idx, dtype=tl.int32)
    input_offsets = input_batch[:, :, None] * (height * width) + (h - pad_h)[:, :, None] * width + (w - pad_w)[:, :, None]
    input_values = tl.load(input_ptr + input_offsets, mask=(h - pad_h) >= 0 & (h - pad_h) < height & (w - pad_w) >= 0 & (w - pad_w) < width, other=0.0)

    # Weight loading (out_channels, in_channels, k, k)
    weight_h = tl.arange(0, k)
    weight_w = tl.arange(0, k)
    weight_offsets = tl.arange(0, out_channels)[:, None, None, None] * (in_channels * k * k) + tl.arange(0, in_channels)[:, None, None] * (k * k) + weight_h[:, None, None] * k + weight_w[:, None]
    weights = tl.load(weight_ptr + weight_offsets, mask=(weight_h < k) & (weight_w < k), other=0.0)

    # Convolution sum
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for i in range(out_channels):
        for j in range(in_channels):
            for k1 in range(k):
                for k2 in range(k):
                    ih = h + k1 - k2
                    iw = w + k2 - k1
                    ih = tl.maximum(ih, 0)
                    iw = tl.maximum(iw, 0)
                    ih = tl.minimum(ih, height)
                    iw = tl.minimum(iw, width)
                    val = tl.load(input_ptr + (batch_idx * in_channels * height * width) + j * height * width + ih * width + iw, mask=(ih < height) & (iw < width), other=0.0)
                    w_val = tl.load(weight_ptr + (i * in_channels * k * k) + j * k * k + k1 * k + k2, mask=(k1 < k) & (k2 < k), other=0.0)
                    output[:, :, i] += val * w_val
    output = tl.tanh(output)
    output = output * 2.0
    bias = tl.load(bias_ptr + tl.arange(0, out_channels), mask=tl.arange(0, out_channels) < out_channels, other=0.0)
    output = output + bias[:, None, None]

    # Max pooling
    pool_h = tl.arange(0, k)
    pool_w = tl.arange(0, k)
    pool_output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for i in range(k):
        for j in range(k):
            ph = h + i
            pw = w + j
            ph = tl.maximum(ph, 0)
            pw = tl.maximum(pw, 0)
            ph = tl.minimum(ph, out_h)
            pw = tl.minimum(pw, out_w)
            val = tl.load(output_ptr + (block_id * out_h * out_w) + ph * out_w + pw, mask=(ph < out_h) & (pw < out_w), other=0.0)
            pool_output += val
    pool_output = pool_output / (k * k)

    # Store result
    tl.store(output_ptr + (block_id * out_h * out_w) + h * out_w + w, pool_output, mask=(h < out_h) & (w < out_w))


def triton_conv_tanh_scale_bias_pool(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_size: int,
    padding_h: int = 1,
    padding_w: int = 1,
    out_channels: int = 64,
    in_channels: int = 8,
    height: int = 256,
    width: int = 256,
):
    assert input.is_cuda, "Input must be on CUDA"
    assert weight.is_cuda, "Weight must be on CUDA"
    assert bias.is_cuda, "Bias must be on CUDA"

    # Ensure contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Output shape
    out_h = (height + 2 * padding_h - kernel_size) // 1
    out_w = (width + 2 * padding_w - kernel_size) // 1
    output = torch.empty(input.shape[0], out_channels, out_h, out_w, device=input.device, dtype=torch.float32)

    # Grid definition
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = lambda meta: (
        (out_h + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_w + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    conv_tanh_scale_bias_pool_kernel[
        grid
    ](
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        output.data_ptr(),
        input.shape[0],
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        padding_h,
        padding_w,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scaling_factor = scaling_factor
        self.bias_shape = bias_shape
        self.pool_kernel_size = pool_kernel_size

        # Conv2d weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1))

    def forward(self, x):
        # Use custom Triton kernel
        return triton_conv_tanh_scale_bias_pool(
            x,
            self.weight,
            self.bias,
            kernel_size=self.kernel_size,
            padding_h=1,
            padding_w=1,
            out_channels=self.out_channels,
            in_channels=self.in_channels,
            height=x.shape[2],
            width=x.shape[3],
        )
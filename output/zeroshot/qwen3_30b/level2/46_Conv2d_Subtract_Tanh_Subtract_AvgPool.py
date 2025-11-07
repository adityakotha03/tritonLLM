import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    bias_ptr,
    in_channels, out_channels, height, width,
    kernel_size,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    bias_stride_0,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # Grid: one block per output tile
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Output tile dimensions
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W

    # Block dimensions
    h_block = tl.minimum(BLOCK_H, height - h_offset)
    w_block = tl.minimum(BLOCK_W, width - w_offset)

    # Initialize output tile
    offs_h = h_offset + tl.arange(0, BLOCK_H)
    offs_w = w_offset + tl.arange(0, BLOCK_W)
    offs_c = pid_c * tl.arange(0, 1)  # Single channel per block
    mask = (offs_h < height) & (offs_w < width)

    # Load output tile
    out = tl.load(
        output_ptr + offs_h[:, None] * output_stride_2 + offs_w[None, :] * output_stride_3,
        mask=mask[:, None] & mask[None, :],
        other=0.0,
    )

    # Accumulate convolution
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Compute input position
            ih = h_offset + kh - PAD_H
            iw = w_offset + kw - PAD_W

            # Load input tile
            input_mask = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
            input_val = tl.load(
                input_ptr + ih[:, None] * input_stride_2 + iw[None, :] * input_stride_3,
                mask=input_mask[:, None] & input_mask[None, :],
                other=0.0,
            )

            # Load weight
            weight_val = tl.load(
                weight_ptr + pid_c * weight_stride_0 + kh * weight_stride_1 + kw * weight_stride_2 + offs_c * weight_stride_3,
                mask=offs_c < out_channels,
                other=0.0,
            )

            # Accumulate
            out += input_val * weight_val

    # Load bias
    bias_val = tl.load(bias_ptr + pid_c * bias_stride_0, mask=pid_c < out_channels, other=0.0)
    out += bias_val

    # Store result
    tl.store(
        output_ptr + offs_h[:, None] * output_stride_2 + offs_w[None, :] * output_stride_3,
        out,
        mask=mask[:, None] & mask[None, :],
    )


@triton.jit
def subtract_tanh_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    subtract_val: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Subtract first value
    x -= subtract_val

    # Apply tanh
    x = tl.tanh(x)

    # Subtract second value
    x -= y

    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def avgpool_kernel(
    input_ptr,
    output_ptr,
    batch_size, in_channels, height, width,
    kernel_size,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * in_channels * (height // kernel_size) * (width // kernel_size)

    # Output indices
    out_b = offsets // (in_channels * (height // kernel_size) * (width // kernel_size))
    out_c = (offsets // ((height // kernel_size) * (width // kernel_size))) % in_channels
    out_h = (offsets // (width // kernel_size)) % (height // kernel_size)
    out_w = offsets % (width // kernel_size)

    # Input indices
    in_b = out_b
    in_c = out_c
    in_h = out_h * kernel_size
    in_w = out_w * kernel_size

    # Load input values
    vals = tl.zeros((kernel_size, kernel_size), dtype=tl.float32)
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            inp_idx = in_b * input_stride_0 + in_c * input_stride_1 + (in_h + kh) * input_stride_2 + (in_w + kw) * input_stride_3
            val = tl.load(input_ptr + inp_idx, mask=(in_h + kh) < height and (in_w + kw) < width, other=0.0)
            vals += val

    # Compute average
    avg = vals / (kernel_size * kernel_size)

    # Store result
    out_idx = out_b * output_stride_0 + out_c * output_stride_1 + out_h * output_stride_2 + out_w * output_stride_3
    tl.store(output_ptr + out_idx, avg, mask=mask)


def triton_conv2d(x, weight, bias, stride=1, padding=1):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3 = weight.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = out.stride()
    bias_stride_0 = bias.stride(0)

    # Block sizes
    BLOCK_H, BLOCK_W = 16, 16

    # Grid
    grid = lambda meta: (out_h // meta["BLOCK_H"], out_w // meta["BLOCK_W"], out_channels)

    # Launch kernel
    conv2d_kernel[grid](
        x, weight, out, bias,
        in_channels, out_channels, height, width,
        kernel_size,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        bias_stride_0,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        STRIDE_H=stride, STRIDE_W=stride,
        PAD_H=padding, PAD_W=padding,
        DTYPE=tl.float32,
    )

    return out


def triton_subtract_tanh(x, subtract1_val, subtract2_val):
    assert x.is_cuda
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Use auto-tuning for optimal block size
    BLOCK_SIZE = 128
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    subtract_tanh_kernel[grid](
        x, torch.tensor([subtract2_val], device=x.device, dtype=x.dtype), out,
        n_elements, BLOCK_SIZE=BLOCK_SIZE, subtract_val=subtract1_val,
    )

    return out


def triton_avgpool(x, kernel_size):
    assert x.is_cuda
    x = x.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_h = height // kernel_size
    out_w = width // kernel_size

    out = torch.empty(batch_size, in_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = out.stride()

    # Grid
    n_elements = batch_size * in_channels * out_h * out_w
    BLOCK_SIZE = 128
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    avgpool_kernel[grid](
        x, out,
        batch_size, in_channels, height, width,
        kernel_size,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        # Step 1: Conv2D
        x = triton_conv2d(x, self.conv.weight, self.conv.bias, stride=1, padding=1)

        # Step 2: Subtract and tanh
        x = triton_subtract_tanh(x, self.subtract1_value, self.subtract2_value)

        # Step 3: AvgPool
        x = triton_avgpool(x, self.kernel_size_pool)

        return x
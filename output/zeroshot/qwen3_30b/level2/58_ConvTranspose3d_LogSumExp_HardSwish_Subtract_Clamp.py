import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    bias_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_size,
    stride,
    padding,
    o_depth,
    o_height,
    o_width,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Define block indices
    pid = tl.program_id(0)  # overall program id
    block_id = pid // (out_channels * o_depth * o_height * o_width)
    out_c = (pid // (o_depth * o_height * o_width)) % out_channels
    out_d = (pid // (o_height * o_width)) % o_depth
    out_h = (pid // o_width) % o_height
    out_w = pid % o_width

    # Calculate input indices
    input_d = out_d * stride - padding
    input_h = out_h * stride - padding
    input_w = out_w * stride - padding

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over kernel size and input channels
    for k_d in tl.arange(0, kernel_size):
        for k_h in tl.arange(0, kernel_size):
            for k_w in tl.arange(0, kernel_size):
                for ic in tl.arange(0, in_channels):
                    # Calculate input coordinates
                    in_d = input_d + k_d
                    in_h = input_h + k_h
                    in_w = input_w + k_w

                    # Check bounds for input
                    valid_d = (in_d >= 0) & (in_d < depth)
                    valid_h = (in_h >= 0) & (in_h < height)
                    valid_w = (in_w >= 0) & (in_w < width)

                    # Skip if out of bounds
                    if not (valid_d and valid_h and valid_w):
                        continue

                    # Load input and weight
                    x_idx = (block_id * in_channels * depth * height * width +
                             ic * depth * height * width +
                             in_d * height * width +
                             in_h * width +
                             in_w)
                    w_idx = (out_c * in_channels * kernel_size * kernel_size * kernel_size +
                             ic * kernel_size * kernel_size * kernel_size +
                             k_d * kernel_size * kernel_size +
                             k_h * kernel_size +
                             k_w)

                    x_val = tl.load(x_ptr + x_idx, mask=valid_d & valid_h & valid_w, other=0.0)
                    w_val = tl.load(w_ptr + w_idx)

                    acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + out_c)
    acc += bias_val

    # Store output
    out_idx = (block_id * out_channels * o_depth * o_height * o_width +
               out_c * o_depth * o_height * o_width +
               out_d * o_height * o_width +
               out_h * o_width +
               out_w)
    tl.store(out_ptr + out_idx, acc)


@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    o_depth,
    o_height,
    o_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // (out_channels * o_depth * o_height * o_width)
    out_c = (pid // (o_depth * o_height * o_width)) % out_channels
    out_d = (pid // (o_height * o_width)) % o_depth
    out_h = (pid // o_width) % o_height
    out_w = pid % o_width

    # Load all elements for this channel and spatial position
    indices = tl.arange(0, out_channels)
    offsets = (block_id * out_channels * o_depth * o_height * o_width +
               indices * o_depth * o_height * o_width +
               out_d * o_height * o_width +
               out_h * o_width +
               out_w)
    values = tl.load(x_ptr + offsets, mask=indices < out_channels, other=-float('inf'))

    # Find maximum value
    max_val = tl.max(values)
    max_val = tl.broadcast_to(max_val, values.shape)

    # Compute exp(x - max)
    exp_values = tl.exp(values - max_val)

    # Sum exp values
    sum_exp = tl.sum(exp_values)

    # Compute log(sum_exp) + max
    logsumexp_val = tl.log(sum_exp) + max_val

    # Store result
    out_idx = (block_id * out_channels * o_depth * o_height * o_width +
               out_c * o_depth * o_height * o_width +
               out_d * o_height * o_width +
               out_h * o_width +
               out_w)
    tl.store(out_ptr + out_idx, logsumexp_val)


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    o_depth,
    o_height,
    o_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // (out_channels * o_depth * o_height * o_width)
    out_c = (pid // (o_depth * o_height * o_width)) % out_channels
    out_d = (pid // (o_height * o_width)) % o_depth
    out_h = (pid // o_width) % o_height
    out_w = pid % o_width

    # Load input
    idx = (block_id * out_channels * o_depth * o_height * o_width +
           out_c * o_depth * o_height * o_width +
           out_d * o_height * o_width +
           out_h * o_width +
           out_w)
    x_val = tl.load(x_ptr + idx)

    # Compute sigmoid(x + 3)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-(x_val + 3.0)))

    # Compute x * sigmoid(x + 3) / 6
    out_val = x_val * sigmoid_val / 6.0

    # Store output
    tl.store(out_ptr + idx, out_val)


@triton.jit
def sub_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    batch_size,
    out_channels,
    o_depth,
    o_height,
    o_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // (out_channels * o_depth * o_height * o_width)
    out_c = (pid // (o_depth * o_height * o_width)) % out_channels
    out_d = (pid // (o_height * o_width)) % o_depth
    out_h = (pid // o_width) % o_height
    out_w = pid % o_width

    # Load inputs
    idx = (block_id * out_channels * o_depth * o_height * o_width +
           out_c * o_depth * o_height * o_width +
           out_d * o_height * o_width +
           out_h * o_width +
           out_w)
    x_val = tl.load(x_ptr + idx)
    y_val = tl.load(y_ptr + idx)

    # Compute subtraction
    out_val = x_val - y_val

    # Store result
    tl.store(out_ptr + idx, out_val)


@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    o_depth,
    o_height,
    o_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_id = pid // (out_channels * o_depth * o_height * o_width)
    out_c = (pid // (o_depth * o_height * o_width)) % out_channels
    out_d = (pid // (o_height * o_width)) % o_depth
    out_h = (pid // o_width) % o_height
    out_w = pid % o_width

    # Load input
    idx = (block_id * out_channels * o_depth * o_height * o_width +
           out_c * o_depth * o_height * o_width +
           out_d * o_height * o_width +
           out_h * o_width +
           out_w)
    x_val = tl.load(x_ptr + idx)

    # Clamp to [-1, 1]
    out_val = tl.clamp(x_val, -1.0, 1.0)

    # Store result
    tl.store(out_ptr + idx, out_val)


def triton_conv_transpose(x, w, bias, stride, padding, kernel_size, o_depth, o_height, o_width):
    assert x.is_cuda and w.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels = w.shape[0]

    out = torch.empty(batch_size, out_channels, o_depth, o_height, o_width, device=x.device, dtype=x.dtype)

    # Grid setup
    num_elements = batch_size * out_channels * o_depth * o_height * o_width
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    BLOCK_SIZE = 128
    TILE_SIZE = 32

    conv_transpose_kernel[grid](
        x, w, out, bias,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, stride, padding,
        o_depth, o_height, o_width,
        BLOCK_SIZE=BLOCK_SIZE, TILE_SIZE=TILE_SIZE
    )

    return out


def triton_logsumexp(x, out_channels, o_depth, o_height, o_width):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size = x.shape[0]

    out = torch.empty(batch_size, 1, o_depth, o_height, o_width, device=x.device, dtype=x.dtype)

    num_elements = batch_size * o_depth * o_height * o_width
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    BLOCK_SIZE = 128

    logsumexp_kernel[grid](
        x, out,
        batch_size, out_channels, o_depth, o_height, o_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_hardswish(x, out_channels, o_depth, o_height, o_width):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size = x.shape[0]

    out = torch.empty(batch_size, out_channels, o_depth, o_height, o_width, device=x.device, dtype=x.dtype)

    num_elements = batch_size * out_channels * o_depth * o_height * o_width
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    BLOCK_SIZE = 128

    hardswish_kernel[grid](
        x, out,
        batch_size, out_channels, o_depth, o_height, o_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_sub(x, y, out_channels, o_depth, o_height, o_width):
    assert x.is_cuda and y.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    y = y.contiguous()

    batch_size = x.shape[0]

    out = torch.empty(batch_size, out_channels, o_depth, o_height, o_width, device=x.device, dtype=x.dtype)

    num_elements = batch_size * out_channels * o_depth * o_height * o_width
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    BLOCK_SIZE = 128

    sub_kernel[grid](
        x, y, out,
        batch_size, out_channels, o_depth, o_height, o_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_clamp(x, out_channels, o_depth, o_height, o_width):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()

    batch_size = x.shape[0]

    out = torch.empty(batch_size, out_channels, o_depth, o_height, o_width, device=x.device, dtype=x.dtype)

    num_elements = batch_size * out_channels * o_depth * o_height * o_width
    grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)

    BLOCK_SIZE = 128

    clamp_kernel[grid](
        x, out,
        batch_size, out_channels, o_depth, o_height, o_width,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

        # Use A100's Tensor Cores: use fp16 or bf16 for higher throughput
        # We'll keep the computation in fp32 for precision
        self._dtype = torch.float32

    def forward(self, x):
        # Get output spatial dimensions
        o_depth = (x.shape[2] - 1) * self.conv_transpose.stride[0] - 2 * self.conv_transpose.padding[0] + self.conv_transpose.kernel_size[0]
        o_height = (x.shape[3] - 1) * self.conv_transpose.stride[1] - 2 * self.conv_transpose.padding[1] + self.conv_transpose.kernel_size[1]
        o_width = (x.shape[4] - 1) * self.conv_transpose.stride[2] - 2 * self.conv_transpose.padding[2] + self.conv_transpose.kernel_size[2]

        # Perform conv transpose with Triton
        x = triton_conv_transpose(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.kernel_size[0],
            o_depth, o_height, o_width
        )

        # LogSumExp: fusion possible? Split into kernel
        x = triton_logsumexp(x, out_channels=x.shape[1], o_depth=o_depth, o_height=o_height, o_width=o_width)

        # HardSwish: x * sigmoid(x + 3) / 6
        x = triton_hardswish(x, out_channels=x.shape[1], o_depth=o_depth, o_height=o_height, o_width=o_width)

        # Subtract bias
        x = triton_sub(x, self.bias, out_channels=x.shape[1], o_depth=o_depth, o_height=o_height, o_width=o_width)

        # Clamp
        x = triton_clamp(x, out_channels=x.shape[1], o_depth=o_depth, o_height=o_height, o_width=o_width)

        return x
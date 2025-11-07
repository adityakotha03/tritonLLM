import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    bias_ptr,  # Bias pointer
    out_ptr,  # Output pointer
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    depth,  # Depth of input
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Size of kernel
    out_depth,  # Output depth
    out_height,  # Output height
    out_width,  # Output width
    BLOCK_SIZE: tl.constexpr,
):
    # Define program ID and offsets
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * out_depth * out_height * out_width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Compute 5D index
    out_idx = offsets // (out_depth * out_height * out_width * out_channels)
    out_c = (offsets // (out_depth * out_height * out_width)) % out_channels
    out_d = (offsets // (out_height * out_width)) % out_depth
    out_h = (offsets // out_width) % out_height
    out_w = offsets % out_width

    # Compute input index
    in_d = out_d
    in_h = out_h
    in_w = out_w
    in_idx = out_idx
    in_c = tl.arange(0, in_channels)

    # Convolution loop over input channels, kernel size
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for c in range(in_channels):
        for kd in range(kernel_size):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Input indices
                    in_d_idx = in_d + kd - (kernel_size - 1) // 2
                    in_h_idx = in_h + kh - (kernel_size - 1) // 2
                    in_w_idx = in_w + kw - (kernel_size - 1) // 2

                    # Validity check
                    valid = (in_d_idx >= 0) & (in_d_idx < depth) & \
                            (in_h_idx >= 0) & (in_h_idx < height) & \
                            (in_w_idx >= 0) & (in_w_idx < width)

                    # Load input and weight
                    x_val = tl.load(x_ptr + (in_idx * in_channels + c) * (depth * height * width) +
                                    (in_d_idx * height * width) + (in_h_idx * width) + in_w_idx,
                                    mask=valid, other=0.0)
                    w_val = tl.load(w_ptr + (out_c * in_channels + c) * (kernel_size ** 3) +
                                    (kd * kernel_size + kh) * kernel_size + kw,
                                    mask=valid, other=0.0)

                    acc += x_val * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + out_c, mask=mask, other=0.0)
    acc += bias_val

    # Apply activation functions in sequence: ReLU, LeakyReLU, GELU, Sigmoid
    acc = tl.maximum(acc, 0.0)
    acc = tl.where(acc < 0.0, acc * 0.01, acc)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pi = 3.141592653589793
    sqrt_2_pi = tl.sqrt(2.0 / pi)
    x3 = acc * acc * acc
    gelu_val = 0.5 * acc * (1.0 + tl.tanh(sqrt_2_pi * (acc + 0.044715 * x3)))
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    # Combine: GELU + Sigmoid (as per model: GELU then Sigmoid)
    acc = gelu_val * sigmoid_val

    # Store result
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_conv3d_relu_leaky_gelu_sigmoid_bias(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, kernel_size, kernel_size = weight.shape
    out_depth = depth - kernel_size + 1
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    out_shape = (batch_size, out_channels, out_depth, out_height, out_width)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Determine block size and grid
    num_elements = batch_size * out_channels * out_depth * out_height * out_width
    BLOCK_SIZE = 1024  # Tunable, power of 2 for optimal performance
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    conv3d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        kernel_size, out_depth, out_height, out_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Use Triton-optimized 3D convolution with fused activations
        return triton_conv3d_relu_leaky_gelu_sigmoid_bias(x, self.conv.weight, self.bias)
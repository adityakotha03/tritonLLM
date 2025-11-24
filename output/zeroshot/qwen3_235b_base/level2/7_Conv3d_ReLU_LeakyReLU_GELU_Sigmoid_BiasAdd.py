import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_act_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, bias_size,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU
    x = tl.maximum(x, 0.0)
    # Apply LeakyReLU: x if x > 0 else 0.01 * x
    leaky_x = x * 0.01
    x = tl.where(x >= 0, x, leaky_x)
    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants for GELU
    PI = 3.14159265
    SQRT_2_OVER_PI = tl.sqrt(2.0 / PI)
    GELU_COEF = 0.044715
    x3 = x * x * x
    inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * x * (1.0 + tanh_inner)
    x = gelu_out
    # Apply Sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-x))
    x = sigmoid_out
    # Load bias: broadcast bias across spatial dimensions
    bias_offset = (offsets % bias_size) % bias_size
    bias = tl.load(bias_ptr + bias_offset)
    x = x + bias
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_fused_conv_act_bias(conv_op, bias, x):
    # Perform 3D convolution using PyTorch (no Triton kernel for Conv3d yet)
    x = conv_op(x)
    # Now apply fused activation + bias with Triton
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    bias_size = bias.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_act_bias_kernel[grid](
        x, bias, out,
        n_elements, bias_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused activation functions and bias addition using Triton.
    Conv3d is kept as-is due to complexity, but all pointwise operations are fused.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return triton_fused_conv_act_bias(self.conv, self.bias, x)
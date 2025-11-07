import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr, STRIDE_C: tl.constexpr,
):
    # Block indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    block_h_start = pid_h * BLOCK_H
    block_w_start = pid_w * BLOCK_W
    block_c_start = pid_c * BLOCK_C

    # Initialize output
    out = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic in range(in_channels):
        # Load kernel weights for current channel
        w_offsets = (ic * out_channels + pid_c) * kernel_size * kernel_size
        w_ptr_c = w_ptr + w_offsets
        w = tl.load(w_ptr_c + tl.arange(0, kernel_size) * STRIDE_C + tl.arange(0, kernel_size)[:, None] * STRIDE_W, mask=(tl.arange(0, kernel_size)[:, None] < kernel_size) & (tl.arange(0, kernel_size)[None, :] < kernel_size), other=0.0)

        # Load input data for current channel
        x_offsets = (pid_batch * in_channels + ic) * height * width
        x_ptr_c = x_ptr + x_offsets
        x = tl.load(
            x_ptr_c + (block_h_start + tl.arange(0, BLOCK_H)[:, None]) * STRIDE_H + (block_w_start + tl.arange(0, BLOCK_W)[None, :]) * STRIDE_W,
            mask=(block_h_start + tl.arange(0, BLOCK_H)[:, None] < height) & (block_w_start + tl.arange(0, BLOCK_W)[None, :] < width),
            other=0.0
        )

        # Perform convolution: outer product of x and w
        out += tl.dot(x, w)

    # Write output
    out_ptr_batch = out_ptr + pid_batch * out_channels * height * width
    out_ptr_c = out_ptr_batch + block_c_start * height * width
    out_ptr_h = out_ptr_c + block_h_start * STRIDE_H
    out_ptr_w = out_ptr_h + block_w_start * STRIDE_W
    tl.store(
        out_ptr_w + tl.arange(0, BLOCK_H)[:, None] * STRIDE_H + tl.arange(0, BLOCK_W)[None, :] * STRIDE_W,
        out,
        mask=(block_h_start + tl.arange(0, BLOCK_H)[:, None] < height) & (block_w_start + tl.arange(0, BLOCK_W)[None, :] < width)
    )


@triton.jit
def mul_relu_gelu_kernel(
    x_ptr, m_ptr, out_ptr,
    batch_size, out_channels, height, width,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr, STRIDE_C: tl.constexpr,
):
    # Block indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    block_h_start = pid_h * BLOCK_H
    block_w_start = pid_w * BLOCK_W
    block_c_start = pid_c * BLOCK_C

    # Load multiplier
    m = tl.load(m_ptr + block_c_start * STRIDE_C, mask=block_c_start < out_channels, other=0.0)

    # Load input data
    x_ptr_batch = x_ptr + pid_batch * out_channels * height * width
    x_ptr_c = x_ptr_batch + block_c_start * height * width
    x_ptr_h = x_ptr_c + block_h_start * STRIDE_H
    x_ptr_w = x_ptr_h + block_w_start * STRIDE_W
    x = tl.load(
        x_ptr_w + tl.arange(0, BLOCK_H)[:, None] * STRIDE_H + tl.arange(0, BLOCK_W)[None, :] * STRIDE_W,
        mask=(block_h_start + tl.arange(0, BLOCK_H)[:, None] < height) & (block_w_start + tl.arange(0, BLOCK_W)[None, :] < width),
        other=0.0
    )

    # Perform multiplication
    x = x * m

    # Apply LeakyReLU (alpha = 0.01)
    x = tl.where(x > 0, x, 0.01 * x)

    # Apply GELU (approximation)
    # Using: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = 0.7978845608028654
    x_sq = x * x
    x_cube = x_sq * x
    inner = sqrt_2_pi * (x + 0.044715 * x_cube)
    tanh_val = tl.tanh(inner)
    x = 0.5 * x * (1 + tanh_val)

    # Store output
    out_ptr_batch = out_ptr + pid_batch * out_channels * height * width
    out_ptr_c = out_ptr_batch + block_c_start * height * width
    out_ptr_h = out_ptr_c + block_h_start * STRIDE_H
    out_ptr_w = out_ptr_h + block_w_start * STRIDE_W
    tl.store(
        out_ptr_w + tl.arange(0, BLOCK_H)[:, None] * STRIDE_H + tl.arange(0, BLOCK_W)[None, :] * STRIDE_W,
        x,
        mask=(block_h_start + tl.arange(0, BLOCK_H)[:, None] < height) & (block_w_start + tl.arange(0, BLOCK_W)[None, :] < width)
    )


def triton_conv2d(x, w, stride=1, padding=1):
    """Triton-based Conv2d with fused kernel."""
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape

    # Assume kernel_size is odd, padding and stride
    pad_h = pad_w = padding

    # Output shape
    out_h = (height + 2 * pad_h - kernel_size) // stride + 1
    out_w = (width + 2 * pad_w - kernel_size) // stride + 1

    # Initialize output
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Tune block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16

    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (out_h + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_w + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
        STRIDE_H=width, STRIDE_W=1, STRIDE_C=kernel_size * kernel_size,
    )
    return out


def triton_mul_relu_gelu(x, m):
    """Triton-based multiplication, LeakyReLU, and GELU fused kernel."""
    assert x.is_cuda and m.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    m = m.contiguous()

    batch_size, out_channels, height, width = x.shape

    # Tune block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16

    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_C"] - 1) // meta["BLOCK_C"],
        (height + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (width + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    # Initialize output
    out = torch.empty_like(x)

    # Launch kernel
    mul_relu_gelu_kernel[grid](
        x, m, out,
        batch_size, out_channels, height, width,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
        STRIDE_H=width, STRIDE_W=1, STRIDE_C=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        # No need to define LeakyReLU, it's in the Triton kernel

    def forward(self, x):
        # Custom Triton kernels
        x = triton_conv2d(x, self.conv.weight)
        x = triton_mul_relu_gelu(x, self.multiplier)
        return x
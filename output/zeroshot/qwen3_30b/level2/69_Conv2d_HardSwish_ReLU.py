import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Size of kernel (assumed square)
    stride,  # Stride (assumed 1)
    padding,  # Padding (assumed 1 for kernel_size=3)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Block indices
    pid = tl.program_id(0)  # Global block ID
    pid_batch = pid // (out_channels * (height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W))
    pid_out_channel = (pid % (out_channels * (height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W))) // ((height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W))
    pid_h = (pid % (out_channels * (height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W))) % ((height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W)) // (width - 2 * padding)
    pid_w = (pid % (out_channels * (height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W))) % ((height - 2 * padding) * (width - 2 * padding) // (BLOCK_H * BLOCK_W)) % (width - 2 * padding)

    # Thread indices
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    offs_c = pid_out_channel * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_c_in = tl.arange(0, in_channels)

    # Load input and weight
    x_ptrs = x_ptr + (pid_batch * in_channels * height * width + offs_c_in[:, None, None] * height * width + offs_h[None, :, None] * width + offs_w[None, None, :])
    x = tl.load(x_ptrs, mask=(offs_h[None, :, None] < height - 2 * padding) & (offs_w[None, None, :] < width - 2 * padding), other=0.0)

    w_ptrs = w_ptr + (pid_out_channel * in_channels * kernel_size * kernel_size + offs_c_in[:, None, None] * kernel_size * kernel_size + (tl.arange(0, kernel_size)[:, None] * kernel_size + tl.arange(0, kernel_size)[None, :]))
    w = tl.load(w_ptrs, mask=(tl.arange(0, kernel_size)[:, None] < kernel_size) & (tl.arange(0, kernel_size)[None, :] < kernel_size), other=0.0)

    # Convolution: outer product over spatial and channel dims
    # x: [in_c, H, W], w: [in_c, K, K] -> out: [out_c, H, W]
    # Use a single accumulator for blockwise reduction
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for c in range(0, in_channels, BLOCK_C):
        offs_c_in = tl.arange(0, min(BLOCK_C, in_channels - c))
        x_c = tl.load(x_ptrs + c * height * width, mask=(offs_h[None, :, None] < height - 2 * padding) & (offs_w[None, None, :] < width - 2 * padding), other=0.0)
        w_c = tl.load(w_ptrs + c * kernel_size * kernel_size, mask=(tl.arange(0, kernel_size)[:, None] < kernel_size) & (tl.arange(0, kernel_size)[None, :] < kernel_size), other=0.0)
        acc += tl.dot(x_c, w_c)

    # Store output
    out_ptrs = out_ptr + (pid_batch * out_channels * (height - 2 * padding) * (width - 2 * padding) + pid_out_channel * (height - 2 * padding) * (width - 2 * padding) + pid_h * BLOCK_H * (width - 2 * padding) + pid_w * BLOCK_W)
    tl.store(out_ptrs, acc, mask=(offs_h[:, None] < height - 2 * padding) & (offs_w[None, :] < width - 2 * padding))


@triton.jit
def hardswish_relu_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # HardSwish: x * ReLU6(x + 3) / 6
    # ReLU6: clamp(x, 0, 6)
    x_clamp = tl.clamp(x + 3.0, 0.0, 6.0)
    x_hardswish = x * x_clamp / 6.0

    # ReLU: max(0, x)
    out = tl.maximum(x_hardswish, 0.0)

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d_hardswish_relu(x: torch.Tensor, weight: torch.Tensor, stride: int = 1, padding: int = 1):
    """
    Perform conv2d followed by hardswish and ReLU in a single fused Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Input dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = weight.shape

    # Output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Determine block sizes
    BLOCK_SIZE = 128
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 8

    # Grid size
    grid_size = (batch_size * out_channels * out_height * out_width + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    conv2d_kernel[grid_size](
        x,
        weight,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        BLOCK_C=BLOCK_C,
    )

    # Apply hardswish + ReLU fused kernel
    n_elements = out.numel()
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    hardswish_relu_kernel[grid](
        out,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False)
        # Ensure weight is on GPU and doesn't require gradients
        self.register_buffer('weight', self.conv.weight.detach())

    def forward(self, x):
        # Use fused Triton kernel for conv2d + hardswish + relu
        return triton_conv2d_hardswish_relu(x, self.weight)
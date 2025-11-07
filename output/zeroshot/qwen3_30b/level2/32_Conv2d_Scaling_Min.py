import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X_ptr,  # Input pointer
    W_ptr,  # Weight pointer
    Out_ptr,  # Output pointer
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Size of convolution kernel
    stride,  # Stride of convolution
    padding,  # Padding of convolution
    BLOCK_H: tl.constexpr,  # Block height
    BLOCK_W: tl.constexpr,  # Block width
    BLOCK_C_OUT: tl.constexpr,  # Output channel block
    BLOCK_C_IN: tl.constexpr,  # Input channel block
):
    # Thread indices
    pid_b = tl.program_id(0)  # Batch index
    pid_c_out = tl.program_id(1)  # Output channel index
    pid_h = tl.program_id(2)  # Output height index
    pid_w = tl.program_id(3)  # Output width index

    # Compute output coordinates
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_h = tl.clip(out_h, 0, height - 1)
    out_w = tl.clip(out_w, 0, width - 1)

    # Compute input coordinates
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding
    in_h = tl.clip(in_h, 0, height - 1)
    in_w = tl.clip(in_w, 0, width - 1)

    # Block offsets for output
    off_h = pid_h * BLOCK_H
    off_w = pid_w * BLOCK_W
    off_c_out = pid_c_out * BLOCK_C_OUT
    off_c_in = tl.arange(0, BLOCK_C_IN)

    # Create offsets for input and weight
    input_offsets = (pid_b * in_channels * height * width +
                     off_c_in[:, None, None] * height * width +
                     in_h[None, None, :] * width +
                     in_w[None, :, None])
    weight_offsets = (off_c_out[:, None, None] * in_channels * kernel_size * kernel_size +
                      off_c_in[None, :, None] * kernel_size * kernel_size +
                      tl.arange(0, kernel_size)[:, None] * kernel_size +
                      tl.arange(0, kernel_size)[None, :])

    # Load input data
    X = tl.load(X_ptr + input_offsets, mask=(in_h[None, None, :] < height) & (in_w[None, :, None] < width), other=0.0)
    W = tl.load(W_ptr + weight_offsets, mask=(in_h[None, None, :] < height) & (in_w[None, :, None] < width), other=0.0)

    # Compute convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            acc += tl.load(W_ptr + weight_offsets + i * kernel_size + j, mask=(in_h[None, None, :] < height) & (in_w[None, :, None] < width), other=0.0) * X
    acc = tl.sum(acc, axis=2)  # Sum over kernel spatial dims

    # Store output
    out_offsets = (pid_b * out_channels * height * width +
                   off_c_out[:, None, None] * height * width +
                   out_h[None, None, :] * width +
                   out_w[None, :, None])
    tl.store(Out_ptr + out_offsets, acc, mask=(out_h[None, None, :] < height) & (out_w[None, :, None] < width))


@triton.jit
def scale_and_min_kernel(
    X_ptr,  # Input pointer
    Out_ptr,  # Output pointer
    batch_size,  # Number of batches
    out_channels,  # Number of output channels
    height,  # Height of input
    width,  # Width of input
    scale_factor: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Thread indices
    pid_b = tl.program_id(0)  # Batch index
    pid_c = tl.program_id(1)  # Channel index
    pid_h = tl.program_id(2)  # Height index
    pid_w = tl.program_id(3)  # Width index

    # Compute output coordinates
    out_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_h = tl.clip(out_h, 0, height - 1)
    out_w = tl.clip(out_w, 0, width - 1)

    # Block offsets
    off_h = pid_h * BLOCK_H
    off_w = pid_w * BLOCK_W
    off_c = pid_c * BLOCK_C

    # Create offsets for input
    input_offsets = (pid_b * out_channels * height * width +
                     off_c[:, None, None] * height * width +
                     out_h[None, None, :] * width +
                     out_w[None, :, None])
    output_offsets = (pid_b * height * width +
                      out_h[None, None, :] * width +
                      out_w[None, :, None])

    # Load input
    X = tl.load(X_ptr + input_offsets, mask=(out_h[None, None, :] < height) & (out_w[None, :, None] < width), other=0.0)

    # Scale
    X = X * scale_factor

    # Min along channel dimension
    X_min = tl.reduce(X, axis=0, combine_fn=tl.min)
    X_min = tl.broadcast_to(X_min, (out_channels, BLOCK_H, BLOCK_W))

    # Store output
    tl.store(Out_ptr + output_offsets, X_min, mask=(out_h[None, None, :] < height) & (out_w[None, :, None]))


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0):
    # Ensure inputs are contiguous on GPU
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Get shapes
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape
    assert kernel_size == 3, "Kernel size must be 3 for this implementation"

    # Calculate output shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_C_OUT = 32
    BLOCK_C_IN = 32

    # Grid dimensions
    grid = (
        batch_size,
        (out_channels + BLOCK_C_OUT - 1) // BLOCK_C_OUT,
        (out_height + BLOCK_H - 1) // BLOCK_H,
        (out_width + BLOCK_W - 1) // BLOCK_W,
    )

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_C_OUT=BLOCK_C_OUT, BLOCK_C_IN=BLOCK_C_IN
    )

    return out


def triton_scale_and_min(x: torch.Tensor, scale_factor: float):
    # Ensure input is contiguous on GPU
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Get shapes
    batch_size, out_channels, height, width = x.shape

    # Prepare output tensor
    out = torch.empty(batch_size, height, width, device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_H = 32
    BLOCK_W = 32
    BLOCK_C = 32

    # Grid dimensions
    grid = (
        batch_size,
        (out_channels + BLOCK_C - 1) // BLOCK_C,
        (height + BLOCK_H - 1) // BLOCK_H,
        (width + BLOCK_W - 1) // BLOCK_W,
    )

    # Launch kernel
    scale_and_min_kernel[grid](
        x, out,
        batch_size, out_channels, height, width,
        scale_factor,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.scale_factor = scale_factor

    def forward(self, x):
        # Perform convolution with Triton kernel
        x = triton_conv2d(x, self.conv_weight, stride=1, padding=1)

        # Scale and min with Triton kernel
        x = triton_scale_and_min(x, self.scale_factor)

        return x
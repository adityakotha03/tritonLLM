import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    output_ptr,  # pointer to output tensor (batch, out_channels, H, W)
    input_stride_h,  # stride in H direction
    input_stride_w,  # stride in W direction
    input_stride_c,  # stride in channel direction
    input_stride_b,  # stride in batch direction
    kernel_ptr,  # pointer to kernel (out_channels, in_channels, kernel_size, kernel_size)
    kernel_stride_c,  # stride in out_channels
    kernel_stride_i,  # stride in in_channels
    kernel_stride_k,  # stride in kernel_size
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block and thread indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)

    # Define the block size for each dimension
    block_h = tl.arange(0, BLOCK_SIZE)
    block_w = tl.arange(0, BLOCK_SIZE)
    block_c = tl.arange(0, BLOCK_SIZE)

    # Compute the output position
    h_start = out_h_idx * BLOCK_SIZE
    w_start = out_w_idx * BLOCK_SIZE
    c_start = out_channel_idx * BLOCK_SIZE

    # Compute the valid range of output indices
    h_end = min(h_start + BLOCK_SIZE, height)
    w_end = min(w_start + BLOCK_SIZE, width)
    c_end = min(c_start + BLOCK_SIZE, out_channels)

    # Load the kernel (out_channels, in_channels, k, k)
    kernel = tl.load(kernel_ptr + (out_channel_idx * kernel_stride_c + block_c) * kernel_stride_i + block_c * kernel_stride_k, mask=block_c < kernel_size, other=0.0)

    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float16)

    # Compute convolution for each input channel
    for i in range(in_channels):
        # Load input features for current channel
        input_h = tl.arange(0, height)
        input_w = tl.arange(0, width)
        input_c = tl.arange(0, in_channels)

        # Compute input indices for each kernel position
        input_h_offset = h_start + block_h
        input_w_offset = w_start + block_w
        input_c_offset = i

        # Load input data with masking
        input_val = tl.load(
            input_ptr + (batch_idx * input_stride_b + 0) * input_stride_c + input_c_offset * input_stride_h + input_h_offset * input_stride_w + input_w_offset,
            mask=(input_h_offset < height) & (input_w_offset < width),
            other=0.0
        )

        # Convolve with kernel
        for k in range(kernel_size):
            for k2 in range(kernel_size):
                kernel_val = tl.load(
                    kernel_ptr + (out_channel_idx * kernel_stride_c + block_c) * kernel_stride_i + i * kernel_stride_k + k * kernel_stride_k + k2,
                    mask=(k < kernel_size) & (k2 < kernel_size),
                    other=0.0
                )
                output += input_val * kernel_val

    # Store output
    tl.store(
        output_ptr + (batch_idx * input_stride_b + 0) * input_stride_c + out_channel_idx * input_stride_h + h_start * input_stride_w + w_start,
        output,
        mask=(h_start < height) & (w_start < width)
    )


@triton.jit
def mish_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute Mish activation: x * tanh(log(1 + exp(x)))
    log_exp_x = tl.log(1.0 + tl.exp(x))
    tanh_log_exp_x = (2.0 * tl.sigmoid(log_exp_x) - 1.0)
    out = x * tanh_log_exp_x
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x: torch.Tensor, kernel: torch.Tensor):
    """
    Custom Triton kernel for 2D convolution.
    """
    assert x.is_cuda and kernel.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    kernel = kernel.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, k, k = kernel.shape

    # Output tensor
    out = torch.empty((batch_size, out_channels, height, width), dtype=torch.float16, device=x.device)

    # Define block size
    BLOCK_SIZE = 16  # Optimized for memory access and warp utilization

    # Grid dimensions
    grid = lambda meta: (
        (batch_size, out_channels, height // BLOCK_SIZE + 1, width // BLOCK_SIZE + 1),
    )

    # Launch kernel
    conv2d_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        x.stride(2),  # H
        x.stride(3),  # W
        x.stride(1),  # C
        x.stride(0),  # B
        kernel.data_ptr(),
        kernel.stride(0),  # out_channels
        kernel.stride(1),  # in_channels
        kernel.stride(2),  # kernel_size
        kernel.stride(3),  # kernel_size
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        k,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_mish(x: torch.Tensor):
    """
    Custom Triton kernel for Mish activation.
    """
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_SIZE = 128
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    mish_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        x.numel(),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Define convolution kernel (out_channels, in_channels, k, k)
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float16))

    def forward(self, x):
        # Apply convolution with custom Triton kernel
        x = triton_conv2d(x, self.kernel)
        # Apply Mish activation with custom Triton kernel
        x = triton_mish(x)
        x = triton_mish(x)
        return x
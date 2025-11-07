import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh_kernel(
    x_ptr,  # Input tensor pointer (batch, in_channels, height, width)
    w_ptr,  # Conv weight pointer (out_channels, in_channels, kernel_size, kernel_size)
    out_ptr,  # Output tensor pointer (batch, out_channels, height, width)
    bias_ptr,  # Bias pointer (out_channels,) if present
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    # Shared memory for the input tile and weight tile
    # Using shared memory to reduce global memory access
    # Shared memory layout: [out_channels, BLOCK_SIZE, BLOCK_SIZE]
    # Block dimensions: tile_height x tile_width
    tile_height = BLOCK_SIZE
    tile_width = BLOCK_SIZE

    # Compute global thread indices
    pid = tl.program_id(0)  # Block index
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE

    # Define offsets for the current block
    off_h = block_start // tile_width
    off_w = block_start % tile_width

    # Thread indices within the block
    tx = tl.arange(0, BLOCK_SIZE)
    ty = tl.arange(0, BLOCK_SIZE)

    # Create masks for boundary checking
    h_mask = off_h + ty < height
    w_mask = off_w + tx < width

    # Calculate output tile position
    out_h = off_h
    out_w = off_w

    # Compute the output element for this thread
    # Calculate the convolution output for the current output element
    # We need to sum over the input channels and kernel size
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over input channels
    for c in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Compute input position
                in_h = out_h + kh - padding
                in_w = out_w + kw - padding

                # Check if input position is valid
                valid = (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)

                # Load input value
                in_ptr = x_ptr + c * height * width + in_h * width + in_w
                x_val = tl.load(in_ptr, mask=valid, other=0.0)

                # Load weight value
                w_ptr_base = w_ptr + c * out_channels * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = tl.load(w_ptr_base + out_ptr, other=0.0)  # We assume out_ptr is used for indexing

                # Accumulate
                accumulator += x_val * w_val

    # Apply bias
    if bias_ptr:
        bias = tl.load(bias_ptr + out_ptr, mask=tl.arange(0, out_channels) < out_channels)
        accumulator += bias

    # Apply first tanh
    accumulator = tl.tanh(accumulator)

    # Apply min along channel dimension (dim=1)
    # Each block handles one output element per channel
    # Min operation over channels for each spatial position
    min_val = tl.reduce(accumulator, axis=0, op=tl.min)

    # Apply second tanh
    min_val = tl.tanh(min_val)

    # Store the result
    out_ptr_base = out_ptr + out_h * width + out_w
    tl.store(out_ptr_base + tx, min_val, mask=tl.arange(0, BLOCK_SIZE) < out_channels)


def triton_conv_min_tanh(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Optimized convolution + min + tanh + tanh using Triton.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."

    # Ensure contiguous tensors
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]

    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, height, width, dtype=x.dtype, device=x.device)

    # Define block size
    BLOCK_SIZE = 128  # Good balance for A100 with 164 KB shared memory

    # Grid size: number of blocks
    grid = lambda meta: (triton.cdiv(height * width, meta["BLOCK_SIZE"]),)

    # Launch the kernel
    conv_min_tanh_kernel[grid](
        x,
        w,
        out,
        bias,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2

    def forward(self, x):
        # Use Triton-optimized convolution + min + tanh + tanh
        # Fuse all operations into one kernel for minimal memory bandwidth
        return triton_conv_min_tanh(
            x, 
            self.conv.weight, 
            self.conv.bias, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
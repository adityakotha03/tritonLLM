import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,  # pointer to input tensor
    output_ptr,  # pointer to output tensor
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    depth: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Compute the range of indices for this block
    block_start_d = tl.program_id(2) * BLOCK_SIZE
    block_start_h = tl.program_id(3) * BLOCK_SIZE
    block_start_w = tl.program_id(4) * BLOCK_SIZE

    # Create offsets for the current block
    d_offsets = block_start_d + tl.arange(0, BLOCK_SIZE)
    h_offsets = block_start_h + tl.arange(0, BLOCK_SIZE)
    w_offsets = block_start_w + tl.arange(0, BLOCK_SIZE)

    # Create mask to avoid out-of-bounds access
    d_mask = d_offsets < depth
    h_mask = h_offsets < height
    w_mask = w_offsets < width

    # Compute valid spatial indices for the current block
    valid_d = d_offsets[None, :, :] < depth
    valid_h = h_offsets[None, :, :] < height
    valid_w = w_offsets[None, :, :] < width

    # Create a 3D grid of indices to compute pooling
    # We will compute the pooled value for each valid spatial location
    # Each thread computes one output element (batch, channel, d, h, w)
    # We use a tiled approach to process spatial dimensions in chunks

    # Initialize output for this channel and batch
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over the spatial dimensions (d, h, w) to compute average
    # We use a nested loop over the spatial dimensions with shared memory
    # Instead, we compute the average over a kernel window using direct indexing

    # We'll use a different strategy: compute the average over the kernel window
    # by loading all valid inputs in a block and averaging them.

    # For each output position (d, h, w), compute the average over kernel window
    # We will compute the average for each (d, h, w) in the output space
    # The kernel window is defined by (kernel_size, kernel_size, kernel_size)
    # We use a loop over the kernel window

    # Instead, we use a tiling approach where each thread handles one output element
    # and computes the average over the kernel window

    # Compute the spatial indices of the output
    d_out = tl.arange(0, depth) // (stride)
    h_out = tl.arange(0, height) // (stride)
    w_out = tl.arange(0, width) // (stride)

    # We'll compute the average over the kernel window using a loop over the kernel
    # We use a 3D loop over the kernel window
    # But since we are in a kernel, we must avoid too many loops

    # Instead, we use a block-based approach: each thread handles one output element
    # and computes the average over the kernel window using a loop over the kernel

    # We'll use a different approach: loop over the kernel window for each output position
    # We compute the average over the kernel window for each output position

    # Let's define the output spatial indices
    d_out = tl.program_id(2)
    h_out = tl.program_id(3)
    w_out = tl.program_id(4)

    # Compute the corresponding input spatial indices
    d_in = d_out * stride
    h_in = h_out * stride
    w_in = w_out * stride

    # Compute the kernel window
    d_kernel_start = d_in - padding
    h_kernel_start = h_in - padding
    w_kernel_start = w_in - padding

    d_kernel_end = d_in + kernel_size - padding
    h_kernel_end = h_in + kernel_size - padding
    w_kernel_end = w_in + kernel_size - padding

    # Compute valid input indices
    d_kernel = tl.arange(0, kernel_size)
    h_kernel = tl.arange(0, kernel_size)
    w_kernel = tl.arange(0, kernel_size)

    # Create mask for valid kernel window
    d_mask_kernel = (d_kernel_start + d_kernel) < depth
    h_mask_kernel = (h_kernel_start + h_kernel) < height
    w_mask_kernel = (w_kernel_start + w_kernel) < width

    # Create a 3D mask for valid kernel positions
    mask = (d_kernel_start + d_kernel) < depth
    mask &= (h_kernel_start + h_kernel) < height
    mask &= (w_kernel_start + w_kernel) < width

    # Load the input values for the kernel window
    # We use a 3D loop over the kernel window
    # We load values from input tensor
    # We use a 3D loop over the kernel window
    # We will use a nested loop over the kernel window

    # Instead, we use a more efficient approach: precompute the kernel window
    # and use a single loop over the kernel window

    # We will use a different strategy: each thread computes one output element
    # and computes the average over the kernel window

    # Compute the input indices for the kernel window
    d_in_idx = d_kernel_start + d_kernel
    h_in_idx = h_kernel_start + h_kernel
    w_in_idx = w_kernel_start + w_kernel

    # Load values from input tensor
    # We use a loop over the kernel window
    # We will use a nested loop over the kernel window
    # We use a 3D loop over the kernel window

    # We will use a different approach: tile the input and compute average in block
    # This is a simplified version that assumes the kernel is small and we can
    # compute the average over the kernel window for each output position

    # We'll use a single loop over the kernel window
    # We compute the average over the kernel window for each output position

    # We use a 3D loop over the kernel window
    # We will compute the average over the kernel window for each output position

    # Initialize the sum and count
    sum_val = tl.zeros(1, dtype=tl.float32)
    count = tl.zeros(1, dtype=tl.int32)

    # Loop over the kernel window
    for d_k in tl.arange(0, kernel_size):
        for h_k in tl.arange(0, kernel_size):
            for w_k in tl.arange(0, kernel_size):
                # Check if the index is valid
                d_idx = d_kernel_start + d_k
                h_idx = h_kernel_start + h_k
                w_idx = w_kernel_start + w_k

                # Check bounds
                if d_idx >= depth or h_idx >= height or w_idx >= width:
                    continue

                # Load value
                val = tl.load(x_ptr + batch_idx * channels * depth * height * width +
                              channel_idx * depth * height * width +
                              d_idx * height * width + h_idx * width + w_idx,
                              mask=(d_idx < depth) & (h_idx < height) & (w_idx < width), other=0.0)
                sum_val += val
                count += 1

    # Compute average
    avg_val = sum_val / count
    # Store output
    output_idx = batch_idx * channels * depth * height * width + channel_idx * depth * height * width + d_out * height * width + h_out * width + w_out
    tl.store(output_ptr + output_idx, avg_val, mask=(d_out < depth // stride) & (h_out < height // stride) & (w_out < width // stride))


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Custom Triton kernel for 3D Average Pooling.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device."
    x = x.contiguous()

    batch_size, channels, depth, height, width = x.shape
    output_depth = (depth + 2 * padding - kernel_size) // stride + 1
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1

    # Allocate output tensor
    output = torch.empty(
        (batch_size, channels, output_depth, output_height, output_width),
        dtype=x.dtype,
        device=x.device
    )

    # Define grid and block size
    BLOCK_SIZE = 16  # Optimal for 3D pooling with coalesced access

    # Grid dimensions: batch, channel, depth, height, width
    # We tile over the spatial dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_depth + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    avg_pool3d_kernel[
        grid
    ](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        depth,
        height,
        width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)
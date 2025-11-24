import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def avg_pool2d_kernel(
    x_ptr,  # pointer to input tensor (batch, channels, H, W)
    output_ptr,  # pointer to output tensor (batch, channels, H_out, W_out)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output dimensions
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    # Current block indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Compute the output position in the output tensor
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    # Check if this block is within bounds
    if out_h >= output_height or out_w >= output_width:
        return

    # Compute the corresponding input region
    h_start = out_h * stride - padding
    h_end = h_start + kernel_size
    w_start = out_w * stride - padding
    w_end = w_start + kernel_size

    # Clamp input bounds to avoid out-of-bounds access
    h_start = tl.max(h_start, 0)
    h_end = tl.min(h_end, input_height)
    w_start = tl.max(w_start, 0)
    w_end = tl.min(w_end, input_width)

    # If the region is empty, skip
    if h_start >= h_end or w_start >= w_end:
        return

    # Initialize accumulator for the average
    acc = 0.0
    count = 0

    # Compute the number of elements in the pooling window
    window_size = (h_end - h_start) * (w_end - w_start)
    if window_size == 0:
        return

    # Load input values in a tiled fashion
    # We process each element in the window using a nested loop
    # Use a block size of BLOCK_SIZE to process contiguous regions
    offsets_h = tl.arange(0, BLOCK_SIZE)
    offsets_w = tl.arange(0, BLOCK_SIZE)

    # Create mask for valid indices
    mask_h = (offsets_h < h_end - h_start)
    mask_w = (offsets_w < w_end - w_start)

    # Compute the full input indices
    # We use a loop over the valid region
    # For each valid (h, w), load the value
    # Use a nested loop to avoid complex indexing
    for h_offset in offsets_h:
        for w_offset in offsets_w:
            h = h_start + h_offset
            w = w_start + w_offset
            if h < input_height and w < input_width:
                idx = batch_idx * channels * input_height * input_width + \
                      channel_idx * input_height * input_width + \
                      h * input_width + w
                val = tl.load(x_ptr + idx, mask=tl.ones_like(h_offset), other=0.0)
                acc += val
                count += 1

    # Compute average
    avg_val = acc / tl.float32(count)

    # Store result
    out_idx = batch_idx * channels * output_height * output_width + \
              channel_idx * output_height * output_width + \
              out_h * output_width + out_w
    tl.store(output_ptr + out_idx, avg_val)


def triton_avg_pool2d(x: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0):
    """
    Custom Triton kernel for 2D Average Pooling.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    batch_size, channels, height, width = x.shape
    if stride is None:
        stride = kernel_size

    # Validate inputs
    assert kernel_size > 0, "Kernel size must be positive"
    assert stride > 0, "Stride must be positive"
    assert padding >= 0, "Padding must be non-negative"

    # Output dimensions
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1

    # Allocate output tensor
    output = torch.empty((batch_size, channels, output_height, output_width), dtype=x.dtype, device=x.device)

    # Define block sizes
    BLOCK_SIZE = 16  # Optimal for small kernels and coalesced access

    # Grid dimensions
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_height + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (output_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    avg_pool2d_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        batch_size,
        channels,
        height,
        width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    y_ptr,
    batch_size,
    channels,
    input_length,
    output_length,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)

    batch_idx = pid_b
    channel_idx = pid_c
    start_l = pid_l * BLOCK_SIZE_L

    offsets_l = start_l + tl.arange(0, BLOCK_SIZE_L)
    mask_l = offsets_l < output_length

    # Compute input start index for each output position
    input_start_idx = offsets_l * stride - padding
    input_end_idx = input_start_idx + kernel_size

    # Clamp input indices to valid range [0, input_length)
    clamped_start_idx = tl.maximum(input_start_idx, 0)
    clamped_end_idx = tl.minimum(input_end_idx, input_length)

    # Valid length per output position
    valid_length = clamped_end_idx - clamped_start_idx

    # Initialize sum accumulator
    sum_val = tl.zeros((BLOCK_SIZE_L,), dtype=tl.float32)

    # Iterate over kernel positions
    for k in range(kernel_size):
        current_idx = input_start_idx + k
        current_mask = (current_idx >= 0) & (current_idx < input_length)
        x_offsets = batch_idx * channels * input_length + \
                    channel_idx * input_length + current_idx
        x_vals = tl.load(x_ptr + x_offsets, mask=current_mask, other=0.0)
        sum_val += x_vals

    # Normalize by kernel size (standard AvgPool behavior, including padding in divisor)
    avg_val = sum_val / kernel_size

    # Store output
    y_offsets = batch_idx * channels * output_length + \
                channel_idx * output_length + offsets_l
    tl.store(y_ptr + y_offsets, avg_val, mask=mask_l)


class ModelNew(nn.Module):
    """
    Optimized 1D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA."

        batch_size, channels, input_length = x.shape

        # Compute output length
        output_length = (input_length + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_shape = (batch_size, channels, output_length)
        y = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Handle empty output
        if output_length == 0:
            return y

        # Define block sizes
        BLOCK_SIZE_M = 1
        BLOCK_SIZE_C = 1
        BLOCK_SIZE_L = 128

        # Grid: (batch_size, channels, number of output blocks)
        grid = (batch_size, channels, triton.cdiv(output_length, BLOCK_SIZE_L))

        avg_pool1d_kernel[grid](
            x,
            y,
            batch_size,
            channels,
            input_length,
            output_length,
            self.kernel_size,
            self.stride,
            self.padding,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_L=BLOCK_SIZE_L,
        )

        return y
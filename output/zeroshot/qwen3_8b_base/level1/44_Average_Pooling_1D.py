import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    n_channels: tl.constexpr,
    input_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index along the input length
    block_idx = tl.program_id(0)
    # Compute the start and end indices for this block
    start = block_idx * BLOCK_SIZE
    end = start + BLOCK_SIZE
    # Compute the actual start and end considering padding
    actual_start = tl.max(tl.arange(0, BLOCK_SIZE) - padding, 0)
    actual_end = tl.min(tl.arange(0, BLOCK_SIZE) + padding + kernel_size, input_length)
    # Compute the offset in the input tensor
    offset = tl.arange(0, BLOCK_SIZE)
    # Compute the input indices
    input_indices = (tl.arange(0, n_channels)[:, None] * input_length + actual_start + offset)
    # Load input values
    input_vals = tl.load(input_ptr + input_indices, mask=actual_start < input_length, other=0.0)
    # Compute the average
    avg = tl.sum(input_vals) / (kernel_size * n_channels)
    # Compute the output indices
    output_indices = (tl.arange(0, n_channels)[:, None] * input_length + (start + kernel_size // 2))
    # Store the result
    tl.store(output_ptr + output_indices, avg, mask=start + kernel_size // 2 < input_length)


def triton_avg_pool(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    Applies 1D Average Pooling using a custom Triton kernel.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    # Compute output length
    output_length = (input.shape[2] + 2 * padding - kernel_size) // stride + 1
    # Prepare output tensor
    output = torch.empty((input.shape[0], input.shape[1], output_length), device=input.device, dtype=input.dtype)
    # Number of elements in the tensor
    n_channels = input.shape[1]
    input_length = input.shape[2]
    # Choose block size (power of 2)
    BLOCK_SIZE = 128
    # Determine the number of blocks needed
    num_blocks = (input_length + 2 * padding - kernel_size) // stride + 1
    grid = lambda meta: (num_blocks,)
    # Launch the Triton kernel
    avg_pool_kernel[grid](input, output, kernel_size, stride, padding, n_channels, input_length, BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)
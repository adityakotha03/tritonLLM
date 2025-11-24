import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    pid = tl.program_id(0)
    # Compute the block's starting position in the input
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < (height + 2 * padding)
    # Compute the input offset for each position
    input_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the output offset
    output_offset = tl.arange(0, BLOCK_SIZE)
    # Compute the input and output positions
    input_pos = input_offsets + (block_start - padding)
    output_pos = output_offset
    # Load input values
    input_vals = tl.load(input_ptr + input_pos, mask=mask, other=0.0)
    # Compute the average
    avg = tl.sum(input_vals) / tl.numel(input_vals)
    # Store the result
    tl.store(output_ptr + output_offset, avg, mask=mask)


def triton_avg_pool(input: torch.Tensor, kernel_size: int, stride: int, padding: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    # Compute the output dimensions
    output_height = (input.shape[2] + 2 * padding - kernel_size) // stride + 1
    output_width = (input.shape[3] + 2 * padding - kernel_size) // stride + 1
    # Create output tensor
    output = torch.empty((input.shape[0], input.shape[1], output_height, output_width), device=input.device)
    # Number of elements in the tensor
    n_elements = output.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    avg_pool_kernel[grid](input, output, input.shape[0], input.shape[1], input.shape[2], input.shape[3],
                          kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        return triton_avg_pool(x, self.kernel_size, self.stride, self.padding)
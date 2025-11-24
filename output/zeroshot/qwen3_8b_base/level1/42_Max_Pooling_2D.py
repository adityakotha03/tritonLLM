import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def max_pool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the output dimensions
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Compute the starting position in the input
    input_h_start = block_idx // (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    input_w_start = block_idx % (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Compute the output position
    out_h_start = input_h_start // stride
    out_w_start = input_w_start // stride

    # Compute the range of offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (kernel_size * kernel_size)

    # Load input data
    input_block = tl.load(
        input_ptr + (out_h_start * stride + offsets // kernel_size) * (width + 2 * padding) + (out_w_start * stride + offsets % kernel_size),
        mask=mask,
        other=-float('inf')
    )

    # Compute max value
    max_val = tl.max(input_block, axis=0)

    # Store the result
    tl.store(output_ptr + (out_h_start * out_w + out_w_start), max_val, mask=tl.full_like(offsets, True, dtype=tl.int32))


def triton_max_pool2d(input: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Triton implementation of 2D Max Pooling.
    """
    assert input.is_cuda, "Input tensor must be on CUDA."
    input = input.contiguous()
    batch_size, channels, height, width = input.shape
    out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    output = torch.empty((batch_size, channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Compute the grid size
    num_elements = out_h * out_w
    BLOCK_SIZE = 128  # Tunable parameter for block size
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    grid = lambda meta: (num_blocks,)
    max_pool2d_kernel[grid](input, output, batch_size, channels, height, width, kernel_size, stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_pool2d(x, self.kernel_size, self.stride, self.padding, self.dilation)
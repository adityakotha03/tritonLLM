import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x.shape[2]

    # Compute the actual input indices
    input_indices = offsets - padding
    input_indices = tl.where(input_indices < 0, 0, input_indices)
    input_indices = tl.where(input_indices >= x.shape[2], x.shape[2] - 1, input_indices)

    # Load the input values
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

    # Compute the max value in the window
    max_val = tl.max(x)

    # Store the result
    tl.store(out_ptr + pid, max_val, mask=pid < out.shape[0])


def triton_maxpool1d(x: torch.Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    Applies 1D Max Pooling using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Compute output shape
    input_size = x.shape[2]
    output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out = torch.empty((x.shape[0], x.shape[1], output_size), device=x.device, dtype=x.dtype)

    # Determine block size
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Define grid
    grid = lambda meta: (out.shape[2],)

    # Launch the kernel
    maxpool1d_kernel[grid](x, out, stride, kernel_size, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_maxpool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
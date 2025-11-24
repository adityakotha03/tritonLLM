import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr,  # Pointer to first input (x)
    w_ptr,  # Pointer to weight matrix (weight.T)
    out_ptr,  # Pointer to output
    batch_size,  # Number of batches
    input_size,  # Input size
    hidden_size,  # Hidden size
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the block start and end indices
    block_start = block_idx
    block_end = block_start + BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the row index
    row = block_start + offsets
    # Compute the column index
    col = tl.arange(0, hidden_size)
    # Compute the matrix indices
    a_offsets = row[:, None] * input_size + col[None, :]
    w_offsets = col[None, :] * input_size + row[:, None]
    # Load the a and w matrices
    a = tl.load(a_ptr + a_offsets, mask=(row < batch_size) & (col < input_size), other=0.0)
    w = tl.load(w_ptr + w_offsets, mask=(row < batch_size) & (col < hidden_size), other=0.0)
    # Compute the matrix multiplication
    acc = tl.dot(a, w)
    # Store the result
    tl.store(out_ptr + row[:, None] * hidden_size + col[None, :], acc, mask=(row < batch_size) & (col < hidden_size))


def triton_gemm(x: torch.Tensor, weight: torch.Tensor, batch_size: int, input_size: int, hidden_size: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    out = torch.empty(batch_size, hidden_size, device=x.device, dtype=x.dtype)

    # Determine the number of blocks needed
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    gemm_kernel[grid](x, weight, out, batch_size, input_size, hidden_size, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def scale_and_sum_kernel(
    x_ptr,  # Pointer to input
    out_ptr,  # Pointer to output
    batch_size,  # Number of batches
    hidden_size,  # Hidden size
    scaling_factor,  # Scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_idx = pid * BLOCK_SIZE
    # Compute the block start and end indices
    block_start = block_idx
    block_end = block_start + BLOCK_SIZE
    # Create a range of offsets for the current block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the row index
    row = block_start + offsets
    # Compute the column index
    col = tl.arange(0, hidden_size)
    # Compute the matrix indices
    x_offsets = row[:, None] * hidden_size + col[None, :]
    # Load the input
    x = tl.load(x_ptr + x_offsets, mask=(row < batch_size) & (col < hidden_size), other=0.0)
    # Compute the sum
    sum_val = tl.sum(x, axis=1)
    # Scale the sum
    scaled_sum = sum_val * scaling_factor
    # Store the result
    tl.store(out_ptr + row[:, None], scaled_sum, mask=row < batch_size)


def triton_scale_and_sum(x: torch.Tensor, scaling_factor: float, batch_size: int, hidden_size: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty(batch_size, 1, device=x.device, dtype=x.dtype)

    # Determine the number of blocks needed
    num_blocks = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks,)

    # Launch the Triton kernel
    scale_and_sum_kernel[grid](x, out, batch_size, hidden_size, scaling_factor, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    Optimized with custom Triton kernels.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Matrix multiplication (GEMM) using Triton kernel
        x = triton_gemm(x, self.weight, x.size(0), x.size(1), self.weight.size(0))
        # Division by 2
        x = x / 2
        # Summation over the hidden dimension
        x = triton_scale_and_sum(x, self.scaling_factor, x.size(0), x.size(1))
        return x
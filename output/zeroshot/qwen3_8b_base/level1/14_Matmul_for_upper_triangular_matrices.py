import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    N,      # Size of square matrices
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Compute the block's starting row and column
    block_start_row = pid * BLOCK_SIZE
    block_start_col = pid * BLOCK_SIZE
    # Compute the block's offset in the matrix
    offsets = block_start_row * N + block_start_col + tl.arange(0, BLOCK_SIZE) * N + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N * N
    # Load A and B matrices
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute the matrix multiplication
    C = tl.dot(A, B)
    # Store the result
    tl.store(C_ptr + offsets, C, mask=mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, N):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()
    # Prepare output tensor
    C = torch.empty_like(A)
    # Number of elements in the tensor
    # Determine the number of blocks needed
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, N, BLOCK_SIZE=1024)
    return C


@triton.jit
def triu_kernel(
    C_ptr,  # Pointer to matrix C
    N,      # Size of square matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    # Compute the block's starting row and column
    block_start_row = pid * BLOCK_SIZE
    block_start_col = pid * BLOCK_SIZE
    # Compute the block's offset in the matrix
    offsets = block_start_row * N + block_start_col + tl.arange(0, BLOCK_SIZE) * N + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N * N
    # Load matrix C
    C = tl.load(C_ptr + offsets, mask=mask, other=0.0)
    # Compute the upper triangular mask
    row_idx = tl.arange(0, BLOCK_SIZE)
    col_idx = tl.arange(0, BLOCK_SIZE)
    mask = row_idx < col_idx
    # Apply the mask
    C = tl.where(mask, C, 0.0)
    # Store the result
    tl.store(C_ptr + offsets, C, mask=mask)


def triton_triu(C: torch.Tensor, N):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert C.is_cuda, "Tensor must be on CUDA."
    C = C.contiguous()
    # Prepare output tensor (same as input)
    # Determine the number of blocks needed
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # Launch the Triton kernel
    triu_kernel[grid](C, N, BLOCK_SIZE=1024)
    return C


class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A * B) for upper triangular matrices
    using custom Triton kernels.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices using custom Triton kernels.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        C = triton_matmul(A, B, A.size(0))
        C = triton_triu(C, A.size(0))
        return C
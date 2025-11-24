import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_scalar_kernel(
    A_ptr,  # Pointer to input matrix A
    s_ptr,  # Pointer to scalar s
    C_ptr,  # Pointer to output matrix C
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes a block of size BLOCK_SIZE
    block_start_m = tl.program_id(0) * BLOCK_SIZE
    block_start_n = tl.program_id(1) * BLOCK_SIZE

    # Create row and column offsets
    row_offsets = block_start_m + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_start_n + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    row_mask = row_offsets < M
    col_mask = col_offsets < N

    # Load the scalar value
    s = tl.load(s_ptr, mask=tl.ones_like(s_ptr), other=0.0)

    # Load matrix A in tile fashion
    A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    A = tl.load(A_ptr + row_offsets[:, None] * N + col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0)

    # Perform scalar multiplication
    C = A * s

    # Store result
    tl.store(C_ptr + row_offsets[:, None] * N + col_offsets[None, :], C, mask=row_mask[:, None] & col_mask[None, :])


def triton_matmul_scalar(A: torch.Tensor, s: float):
    """
    Custom Triton kernel for matrix-scalar multiplication.
    """
    assert A.is_cuda, "Input tensor must be on CUDA."
    A = A.contiguous()

    M, N = A.shape
    s_tensor = torch.tensor([s], device=A.device, dtype=A.dtype)

    # Define block size (power of 2, optimized for Ampere)
    BLOCK_SIZE = 128

    # Grid dimensions: number of blocks in row and column directions
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    matmul_scalar_kernel[grid](A, s_tensor, A, M, N, BLOCK_SIZE=BLOCK_SIZE)

    return A


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using custom Triton kernel.
        """
        return triton_matmul_scalar(A, s)
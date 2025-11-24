import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    n_row: tl.constexpr, n_col: tl.constexpr, n_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a tile of size BLOCK_SIZE x BLOCK_SIZE
    pid = tl.program_id(0)
    num_blocks = (n_row + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_row = pid % num_blocks
    block_col = pid // num_blocks

    # Compute the block's row and column indices
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE

    # Load A and B blocks
    A_block = tl.load(A + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * n_col + col_start + tl.arange(0, BLOCK_SIZE), mask=(row_start + tl.arange(0, BLOCK_SIZE))[:, None] < n_row, other=0.0)
    B_block = tl.load(B + col_start + tl.arange(0, BLOCK_SIZE)[None, :] * n_row + row_start + tl.arange(0, BLOCK_SIZE), mask=(col_start + tl.arange(0, BLOCK_SIZE))[None, :] < n_col, other=0.0)

    # Compute the dot product
    C_block = tl.dot(A_block, B_block)

    # Write back to C
    tl.store(C + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * n_col + col_start + tl.arange(0, BLOCK_SIZE), C_block, mask=(row_start + tl.arange(0, BLOCK_SIZE))[:, None] < n_row)


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom matrix multiplication using Triton.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor
    C = torch.empty(A.shape, dtype=A.dtype, device=A.device)

    # Constants
    n_row = A.shape[0]
    n_col = B.shape[1]
    n_k = A.shape[1]

    # Choose block size (adjust for performance)
    BLOCK_SIZE = 128

    # Determine number of blocks
    num_blocks = (n_row + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch kernel
    grid = (num_blocks,)
    matmul_kernel[grid](A, B, C, n_row, n_col, n_k, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    M,      # Number of rows in A
    N,      # Number of columns in B
    K,      # Number of columns in A / rows in B
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Each program processes a block of the output matrix C
    pid = tl.program_id(0)
    # Compute the block's row and column indices
    block_row = pid // GROUP_SIZE
    block_col = pid % GROUP_SIZE
    # Compute the offset in C
    row_start = block_row * BLOCK_SIZE
    col_start = block_col * BLOCK_SIZE
    # Compute the offsets in A and B
    A_offsets = row_start + tl.arange(0, BLOCK_SIZE)[:, None] * K + col_start + tl.arange(0, BLOCK_SIZE)
    B_offsets = col_start + tl.arange(0, BLOCK_SIZE)[None, :] * K + row_start + tl.arange(0, BLOCK_SIZE)
    # Load A and B
    A = tl.load(A_ptr + A_offsets, mask=(A_offsets < M * K) & (A_offsets >= 0), other=0.0)
    B = tl.load(B_ptr + B_offsets, mask=(B_offsets < K * N) & (B_offsets >= 0), other=0.0)
    # Compute the dot product
    C = tl.dot(A, B)
    # Store the result
    tl.store(C_ptr + row_start + tl.arange(0, BLOCK_SIZE)[:, None] * N + col_start + tl.arange(0, BLOCK_SIZE), C, mask=(row_start + tl.arange(0, BLOCK_SIZE) < M) & (col_start + tl.arange(0, BLOCK_SIZE) < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
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
    C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

    # Parameters
    M, K = A.shape
    _, N = B.shape
    BLOCK_SIZE = 128
    GROUP_SIZE = 8

    # Determine the number of blocks needed
    num_blocks = (M * N + (BLOCK_SIZE * GROUP_SIZE - 1)) // (BLOCK_SIZE * GROUP_SIZE)
    grid = lambda meta: (num_blocks,)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE, GROUP_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
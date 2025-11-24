import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A (K, M)
    B_ptr,  # Pointer to matrix B (K, N)
    C_ptr,  # Pointer to output matrix C (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of output
    pid = tl.program_id(0)
    # Compute the block offset in the output matrix
    block_row = pid // N
    block_col = pid % N
    # Compute the offset in the output matrix
    row_offsets = block_row * N + tl.arange(0, BLOCK_SIZE)
    col_offsets = block_col * N + tl.arange(0, BLOCK_SIZE)
    # Compute the offset in matrix A and B
    a_offsets = tl.arange(0, BLOCK_SIZE) * M + row_offsets
    b_offsets = tl.arange(0, BLOCK_SIZE) * N + col_offsets
    # Load matrix A and B
    a = tl.load(A_ptr + a_offsets, mask=a_offsets < K * M, other=0.0)
    b = tl.load(B_ptr + b_offsets, mask=b_offsets < K * N, other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(C_ptr + row_offsets + col_offsets, c, mask=row_offsets < M * N)


def triton_matmul(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
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
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # Define block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((M * N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Transpose A to (M, K) and perform matrix multiplication with B (K, N)
        # Result is (M, N)
        return triton_matmul(A.T, B, M=A.shape[1], N=B.shape[1], K=A.shape[0])
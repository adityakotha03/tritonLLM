import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A
    B_ptr,  # Pointer to matrix B
    C_ptr,  # Pointer to output matrix C
    M,  # Number of rows in A and C
    N,  # Number of columns in B and C
    K,  # Number of columns in A and rows in B
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Compute the block dimensions
    block_m = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Compute the block offsets
    block_idx = pid % (block_m * block_n)
    block_m_idx = block_idx // block_n
    block_n_idx = block_idx % block_n
    # Compute the block start and end indices
    block_start_m = block_m_idx * BLOCK_SIZE
    block_start_n = block_n_idx * BLOCK_SIZE
    block_end_m = min(block_start_m + BLOCK_SIZE, M)
    block_end_n = min(block_start_n + BLOCK_SIZE, N)
    # Compute the thread indices within the block
    thread_idx = tl.program_id(1)
    # Compute the thread's row and column indices
    row_idx = thread_idx // GROUP_SIZE
    col_idx = thread_idx % GROUP_SIZE
    # Compute the offset for the current thread
    offset = row_idx * K + col_idx
    # Load the A matrix values
    a = tl.load(A_ptr + offset, mask=offset < K, other=0.0)
    # Compute the dot product
    acc = tl.dot(a, tl.load(B_ptr + offset, mask=offset < K, other=0.0))
    # Store the result
    tl.store(C_ptr + row_idx * N + col_idx, acc)


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
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    # Define block size and group size
    BLOCK_SIZE = 128
    GROUP_SIZE = 64

    # Determine the number of blocks needed
    num_blocks = (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    matmul_kernel[ (num_blocks, 1) ](A, B, C, M, N, K, BLOCK_SIZE, GROUP_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose A and B for the kernel
        A_T = A.t()
        B_T = B.t()
        return triton_matmul(A_T, B_T, A_T.shape[1], B_T.shape[0], A_T.shape[0])
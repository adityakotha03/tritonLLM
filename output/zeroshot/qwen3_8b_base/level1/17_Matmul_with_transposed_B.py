import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to matrix A (M, K)
    B_ptr,  # Pointer to matrix B (K, N)
    C_ptr,  # Pointer to output matrix C (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the block offset
    block_m = pid // (N // BLOCK_SIZE)
    block_n = pid % (N // BLOCK_SIZE)
    # Compute the block start indices
    block_start_m = block_m * BLOCK_SIZE
    block_start_n = block_n * BLOCK_SIZE
    # Compute the offset for each thread in the block
    offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the block's row and column indices
    m_offsets = block_start_m + offsets
    n_offsets = block_start_n + offsets
    # Load A and B
    a = tl.load(A_ptr + m_offsets[:, None] * K + n_offsets[None, :], mask=(m_offsets < M) & (n_offsets < K), other=0.0)
    b = tl.load(B_ptr + n_offsets[:, None] * K + m_offsets[None, :], mask=(n_offsets < N) & (m_offsets < K), other=0.0)
    # Compute the dot product
    c = tl.dot(a, b)
    # Store the result
    tl.store(C_ptr + m_offsets[:, None] * N + n_offsets[None, :], c, mask=(m_offsets < M) & (n_offsets < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor, M: int, N: int, K: int):
    """
    This function wraps the Triton kernel call for matrix multiplication.
    """
    # Ensure the inputs are on the GPU
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    # Ensure the inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    # Prepare output tensor
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    # Determine the block size
    BLOCK_SIZE = 128
    # Determine the number of blocks needed
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Launch the Triton kernel
    grid = (num_blocks,)
    matmul_kernel[grid](A, B, C, M, N, K, BLOCK_SIZE=BLOCK_SIZE)
    return C


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using a custom Triton kernel.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Transpose B to (N, K) for the kernel
        B_T = B.T
        # Ensure the input tensors are on the GPU
        A = A.cuda()
        B_T = B_T.cuda()
        # Perform matrix multiplication using the Triton kernel
        return triton_matmul(A, B_T, A.shape[0], B_T.shape[1], A.shape[1])
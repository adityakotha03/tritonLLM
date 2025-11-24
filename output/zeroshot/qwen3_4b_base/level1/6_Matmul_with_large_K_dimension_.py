import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input A of shape (M, K)
    b_ptr,  # Pointer to input B of shape (K, N)
    c_ptr,  # Pointer to output C of shape (M, N)
    m: tl.constexpr,  # Number of rows in A
    n: tl.constexpr,  # Number of columns in B
    k: tl.constexpr,  # Number of columns in A / rows in B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for row block (M dimension)
    pid_m = tl.program_id(0)
    # Program ID for column block (N dimension)
    pid_n = tl.program_id(1)

    # Compute the starting row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Create the range of indices for the current block
    rm = tl.arange(0, BLOCK_SIZE_M)
    rn = tl.arange(0, BLOCK_SIZE_N)
    rk = tl.arange(0, BLOCK_SIZE_K)

    # Load row and column indices
    mask_m = rm < m
    mask_n = rn < n
    mask_k = rk < k

    # Load A: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a = tl.load(a_ptr + (row_start + rm)[:, None] * k + rk[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    
    # Load B: (K, N) -> (BLOCK_SIZE_K, BLOCK_SIZE_N)
    b = tl.load(b_ptr + (rk[:, None]) * n + (col_start + rn)[None, :], mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Compute the dot product across K
    acc = tl.dot(a, b)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Write result to output
    c = tl.store(c_ptr + (row_start + rm)[:, None] * n + (col_start + rn)[None, :], acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication with optimized block tiling.
    Uses fused block-wise computation to reduce global memory traffic and leverage Tensor Cores.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure input shapes are correct
    assert a.shape[1] == b.shape[0], "Incompatible dimensions: A.shape[1] must equal B.shape[0]"

    M, K = a.shape
    K, N = b.shape
    assert K == a.shape[1] and K == b.shape[0], "Shape mismatch: A and B must have compatible inner dimensions"

    # Define block sizes (powers of 2, optimized for Ampere Tensor Cores)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Compute grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Launch kernel
    matmul_kernel[grid_m, grid_n](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=torch.empty(M, N, dtype=a.dtype, device=a.device).data_ptr(),
        m=M,
        n=N,
        k=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return torch.empty(M, N, dtype=a.dtype, device=a.device)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using a custom Triton kernel.
        Optimized with block tiling and Tensor Core-friendly data types.
        """
        return triton_matmul(A, B)
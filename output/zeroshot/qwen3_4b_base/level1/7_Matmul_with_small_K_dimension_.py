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
    m: tl.constexpr,  # M dimension
    n: tl.constexpr,  # N dimension
    k: tl.constexpr,  # K dimension
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute the block of output (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting row and column for this block
    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Create offsets for the current block
    offsets_m = tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)

    # Load A and B in tile fashion
    # A: (M, K) -> tile along M and K
    # B: (K, N) -> tile along K and N
    # We compute C[i, j] = sum_k A[i, k] * B[k, j]

    # Load A: (BLOCK_SIZE_M, k)
    a = tl.zeros((BLOCK_SIZE_M, k), dtype=tl.float16)
    a_mask = offsets_m < m
    a = tl.load(a_ptr + offsets_m[:, None] * k + tl.arange(0, k)[None, :], mask=a_mask[:, None], other=0.0)

    # Load B: (k, BLOCK_SIZE_N)
    b = tl.zeros((k, BLOCK_SIZE_N), dtype=tl.float16)
    b_mask = offsets_n < n
    b = tl.load(b_ptr + tl.arange(0, k)[None, :] * n + offsets_n[None, :], mask=b_mask[None, :], other=0.0)

    # Compute the dot product over K
    # We use a fused reduction over k
    # This is done via a loop over k with shared memory for intermediate accumulation
    # But in Triton, we can use a simple loop over k with tiling
    # We'll use a fused kernel with a loop over k (inner dimension)
    # Since k is small (64), we can safely loop over k

    # Accumulate result
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k_idx in range(k):
        # Load a_row and b_col for this k
        a_row = tl.load(a_ptr + (row_start + offsets_m)[:, None] * k + k_idx, mask=(offsets_m < m)[:, None], other=0.0)
        b_col = tl.load(b_ptr + k_idx * n + (col_start + offsets_n), mask=(offsets_n < n)[None, :], other=0.0)
        # Multiply and accumulate
        c = c + (a_row * b_col)

    # Store the result
    c = c.to(tl.float16)
    tl.store(c_ptr + (row_start + offsets_m)[:, None] * n + (col_start + offsets_n)[None, :], c, mask=(offsets_m < m)[:, None] & (offsets_n < n)[None, :])


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel for matrix multiplication.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    # Use FP16 for better performance on Tensor Cores
    # We use a fused kernel with tiling over M and N
    # We use BLOCK_SIZE_M and BLOCK_SIZE_N as powers of 2 for optimal performance
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid dimensions: number of blocks in M and N directions
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    # Allocate output tensor
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)

    # Launch kernel
    matmul_kernel[grid](a, b, c, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N)

    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input A of shape (K, M)
    b_ptr,  # Pointer to input B of shape (N, K)
    c_ptr,  # Pointer to output C of shape (M, N)
    m,      # Number of rows in A (M)
    n,      # Number of columns in B (N)
    k,      # Number of columns in A and rows in B (K)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the block of indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load the A and B matrices in tile format
    # A: (K, M) -> we access A[offs_m, :] as (K, M)
    # B: (N, K) -> we access B[offs_n, :] as (N, K)
    # We use a loop over K to compute the dot product
    # We use shared memory to cache the tiles of A and B
    # A tile: (K, BLOCK_SIZE_M)
    # B tile: (BLOCK_SIZE_N, K)
    # We compute C[offs_m, offs_n] = sum_k A[offs_m, k] * B[offs_n, k]

    # Load A tile: (K, BLOCK_SIZE_M)
    a = tl.load(a_ptr + offs_m[:, None] * k + tl.arange(0, k)[None, :], mask=(offs_m[:, None] < m) & (tl.arange(0, k)[None, :] < k), other=0.0)
    # Load B tile: (BLOCK_SIZE_N, K)
    b = tl.load(b_ptr + offs_n[None, :] * k + tl.arange(0, k)[None, :], mask=(offs_n[None, :] < n) & (tl.arange(0, k)[None, :] < k), other=0.0)

    # Compute the dot product across K
    # We use a fused reduction over K
    # We compute c[offs_m, offs_n] = sum_k a[offs_m, k] * b[offs_n, k]
    # But note: a is (BLOCK_SIZE_M, K), b is (BLOCK_SIZE_N, K)
    # So we need to reshape and do a proper inner product
    # Instead, we do a proper tiling with shared memory

    # Let's restructure: we compute a tile of A and B in shared memory
    # We use shared memory for A and B tiles
    # We do a loop over k, but we can fuse the computation

    # Instead, we do a more efficient version: use shared memory to cache tiles
    # We will compute the full matrix multiplication in a fused kernel
    # We use a different tiling pattern: (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Reset: we use a different approach: loop over k, and use shared memory
    # We will compute the dot product over k using shared memory
    # But note: we are already in a block, so we can compute the inner product directly

    # We recompute the tiles with proper indexing
    # We define the tile of A: (BLOCK_SIZE_M, K)
    # We define the tile of B: (K, BLOCK_SIZE_N)

    # Load A tile: (BLOCK_SIZE_M, K)
    a_tile = tl.load(a_ptr + offs_m[:, None] * k + tl.arange(0, k)[None, :], mask=(offs_m[:, None] < m) & (tl.arange(0, k)[None, :] < k), other=0.0)
    # Load B tile: (K, BLOCK_SIZE_N)
    b_tile = tl.load(b_ptr + tl.arange(0, k)[None, :] * n + offs_n[None, :], mask=(tl.arange(0, k)[None, :] < k) & (offs_n[None, :] < n), other=0.0)

    # Now compute the dot product over k
    # a_tile: (BLOCK_SIZE_M, K), b_tile: (K, BLOCK_SIZE_N)
    # c = a_tile @ b_tile
    # We do a fused reduction over k
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in tl.range(k):
        a_k = a_tile[:, k_idx]
        b_k = b_tile[k_idx, :]
        c += tl.dot(a_k, b_k)

    # Write result to output
    c = tl.where(offs_m[:, None] < m, c, 0.0)
    c = tl.where(offs_n[None, :] < n, c, 0.0)
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def matmul_kernel_fused(
    a_ptr,  # (K, M)
    b_ptr,  # (N, K)
    c_ptr,  # (M, N)
    m,      # M
    n,      # N
    k,      # K
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the block of indices
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Define the tile dimensions
    # We use shared memory to cache tiles of A and B
    # A: (K, M) -> we want (BLOCK_SIZE_M, K)
    # B: (N, K) -> we want (K, BLOCK_SIZE_N)

    # Load A tile: (BLOCK_SIZE_M, K)
    a = tl.load(a_ptr + offs_m[:, None] * k + tl.arange(0, k)[None, :], mask=(offs_m[:, None] < m) & (tl.arange(0, k)[None, :] < k), other=0.0)
    # Load B tile: (K, BLOCK_SIZE_N)
    b = tl.load(b_ptr + tl.arange(0, k)[None, :] * n + offs_n[None, :], mask=(tl.arange(0, k)[None, :] < k) & (offs_n[None, :] < n), other=0.0)

    # Compute the dot product over k
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_idx in tl.arange(0, k):
        a_k = a[:, k_idx]
        b_k = b[k_idx, :]
        c += tl.dot(a_k, b_k)

    # Write result to output
    tl.store(c_ptr + offs_m[:, None] * n + offs_n[None, :], c, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using a custom Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure shapes are correct
    assert a.shape[0] == b.shape[1], "Shape mismatch: A must have shape (K, M) and B must have shape (N, K)"
    assert a.shape[1] == b.shape[0], "Shape mismatch: A and B must be compatible"

    m, k, n = a.shape[1], a.shape[0], b.shape[1]

    # Output tensor
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Launch kernel
    matmul_kernel_fused[grid_m, grid_n](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=c.data_ptr(),
        m=m,
        n=n,
        k=k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication: C = A.T @ B.T
        """
        # A: (K, M), B: (N, K)
        # We compute A.T @ B.T = (M, K) @ (K, N) = (M, N)
        # So we compute C = A.T @ B.T
        # A.T is (M, K), B.T is (K, N)
        # So we compute C = A.T @ B.T

        # Instead of using torch.matmul, we use our custom Triton kernel
        # We compute A.T @ B.T directly
        A_T = A.T
        B_T = B.T

        return triton_matmul(A_T, B_T)
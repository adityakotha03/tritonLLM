import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,  # Pointer to input A of shape (K, M)
    b_ptr,  # Pointer to input B of shape (K, N)
    c_ptr,  # Pointer to output C of shape (M, N)
    m,      # Number of rows in A.T (i.e., M)
    n,      # Number of columns in B (i.e., N)
    k,      # Number of columns in A (i.e., K)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program instance handles a block of rows (M) and columns (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Define the range of rows and columns this block is responsible for
    row_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load A.T (which is (M, K)) and B (which is (K, N)) in tiles
    # A.T: (M, K) -> we access row i, col j as A[i, j]
    # B: (K, N) -> we access row j, col k as B[j, k]
    # C[i, k] = sum_j A[i, j] * B[j, k]

    # Load A.T (M, K) in tile: row_offsets[i], col j
    a = tl.zeros((BLOCK_SIZE_M, k), dtype=tl.float16)
    b = tl.zeros((k, BLOCK_SIZE_N), dtype=tl.float16)

    # Load A.T in row-major: for each row in row_offsets, load k columns
    a_ptr_offset = a_ptr + row_offsets[:, None] * k + tl.arange(0, k)[None, :]
    a = tl.load(a_ptr_offset, mask=row_offsets[:, None] < m, other=0.0)

    # Load B in column-major: for each column in col_offsets, load k rows
    b_ptr_offset = b_ptr + tl.arange(0, k)[:, None] * n + col_offsets[None, :]
    b = tl.load(b_ptr_offset, mask=col_offsets[None, :] < n, other=0.0)

    # Perform matrix multiplication: C[i, j] = sum_k A[i, k] * B[k, j]
    # Use fused computation to avoid intermediate storage
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k_idx in range(0, k, BLOCK_SIZE_M):
        k_start = k_idx
        k_end = min(k_idx + BLOCK_SIZE_M, k)
        # Compute partial dot product
        a_k = a[:, k_start:k_end]
        b_k = b[k_start:k_end, :]
        # Compute dot product for each (i, j)
        temp = tl.dot(a_k, b_k)
        c += temp

    # Store result
    c_ptr_offset = c_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None] * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]
    tl.store(c_ptr_offset, c, mask=(row_offsets[:, None] < m) & (col_offsets[None, :] < n))


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Custom Triton kernel to compute A.T @ B.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure tensors are in correct shape: A is (K, M), B is (K, N)
    # Output C is (M, N)
    m, k = a.shape[1], a.shape[0]
    n = b.shape[1]

    # Output tensor
    c = torch.empty((m, n), dtype=a.dtype, device=a.device)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = lambda meta: (grid_m, grid_n)

    # Launch kernel
    matmul_kernel[grid](a, b, c, m, n, k, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N)
    return c


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Replace torch.matmul(A.T, B) with custom Triton kernel
        return triton_matmul(A, B)
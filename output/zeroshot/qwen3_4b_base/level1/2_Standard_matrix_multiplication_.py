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
    k: tl.constexpr,  # Number of columns in A and rows in B
    n: tl.constexpr,  # Number of columns in B
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the starting row and column indices for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Load the row of A and column of B into shared memory
    # We use shared memory to reduce global memory accesses and enable better cache reuse
    # A: (M, K), B: (K, N)
    # We tile the matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # We compute this in blocks of (BLOCK_SIZE_M, BLOCK_SIZE_N) for output, and (BLOCK_SIZE_K) for inner dimension

    # Shared memory for A and B
    a_shared = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16)
    b_shared = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)

    # Load A into shared memory
    a_mask = (offs_m < m) & (tl.arange(0, BLOCK_SIZE_M) < k)
    a_offsets = offs_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[None, :]
    a_idx = a_offsets * k + tl.arange(0, BLOCK_SIZE_M)
    a_load = tl.load(a_ptr + a_idx, mask=a_mask, other=0.0)
    a_shared = a_load

    # Load B into shared memory
    b_mask = (offs_n < n) & (tl.arange(0, BLOCK_SIZE_N) < k)
    b_offsets = tl.arange(0, BLOCK_SIZE_K)[:, None] + offs_n[None, :]
    b_idx = b_offsets * n + tl.arange(0, BLOCK_SIZE_N)
    b_load = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
    b_shared = b_load

    # Perform the matrix multiplication in shared memory
    # We compute C[pid_m, pid_n] = sum_k a_shared[i,k] * b_shared[k,j]
    # We use a loop over k to compute the dot product
    # We use a single kernel to handle the full matmul via tiling
    # Use a fused kernel with shared memory to reduce global memory traffic

    # Compute the output
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    for k_idx in range(0, k, BLOCK_SIZE_K):
        # Load the current block of K
        k_start = k_idx
        k_end = min(k_idx + BLOCK_SIZE_K, k)
        k_range = tl.arange(0, k_end - k_start)

        # Load A block
        a_k = tl.load(a_ptr + (offs_m[:, None] * k + k_range[None, :]), mask=(offs_m[:, None] < m) & (k_range < k), other=0.0)
        # Load B block
        b_k = tl.load(b_ptr + (k_range[:, None] * n + offs_n[None, :]), mask=(k_range < k) & (offs_n < n), other=0.0)

        # Compute dot product
        temp = tl.dot(a_k, b_k)
        c = c + temp

    # Store result
    c_out = c
    c_mask = (offs_m < m) & (offs_n < n)
    tl.store(c_ptr + (offs_m[:, None] * n + offs_n[None, :]), c_out, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Performs matrix multiplication using a custom Triton kernel.
    Optimized with tiling and shared memory for high performance on A100.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Ensure inputs are in float16 for TF32/FP16 Tensor Core usage
    a = a.half()
    b = b.half()

    # Get dimensions
    m, k = a.shape
    k_, n = b.shape

    # Validate dimensions
    assert k == k_, "Incompatible dimensions: A's K must equal B's K."

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    # Grid dimensions
    grid_m = (m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

    # Launch kernel
    matmul_kernel[grid_m, grid_n](
        a_ptr=a.data_ptr(),
        b_ptr=b.data_ptr(),
        c_ptr=torch.empty(m, n, dtype=torch.float16, device=a.device).data_ptr(),
        m=m,
        k=k,
        n=n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Convert back to float32 if needed (e.g., for downstream operations)
    return torch.empty(m, n, dtype=torch.float32, device=a.device)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
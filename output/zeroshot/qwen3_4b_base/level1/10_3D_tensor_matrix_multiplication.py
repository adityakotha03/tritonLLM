import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr,  # Pointer to input tensor A of shape (N, M, K)
    B_ptr,  # Pointer to input matrix B of shape (K, L)
    out_ptr,  # Pointer to output tensor of shape (N, M, L)
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute the current block indices
    pid = tl.program_id(0)  # Block index in the N dimension
    block_n = pid
    block_m = tl.program_id(1)  # Block index in the M dimension
    block_k = tl.program_id(2)  # Block index in the K dimension

    # Define the block offsets
    m_start = block_m * BLOCK_SIZE_M
    m_end = m_start + BLOCK_SIZE_M
    k_start = block_k * BLOCK_SIZE_K
    k_end = k_start + BLOCK_SIZE_K

    # Load the output block (initialize to zero)
    # We will compute (N, M, L) via loop over K
    # For each (n, m), we compute sum over k of A[n, m, k] * B[k, l]
    # We process one row of M at a time, and one block of K at a time

    # Loop over M and K to compute the output
    # We compute one row of output (m) per block
    for m in range(m_start, m_end):
        # Load row m of A for all k in current block
        A_row = tl.zeros((BLOCK_SIZE_K, ), dtype=tl.float16)
        for k in range(k_start, k_end):
            # Load A[n, m, k] for current m and k
            # A is (N, M, K), so we need to index properly
            # For fixed n and m, we load A[n, m, k]
            # But we are iterating over m and k, so we need to fix n
            # We are processing one n at a time via block_n
            # So we load A[block_n, m, k]
            idx_k = k
            idx_m = m
            # Load A[block_n, idx_m, idx_k]
            # We will use a 2D block to load A
            # We use shared memory to cache A slices
            pass

    # Instead, we restructure to use a more efficient tiling strategy
    # We will compute one output row (m) at a time, and use shared memory to cache B slices
    # We will do a fused matmul over K with tiling
    # We will restructure the kernel to compute (N, M, L) with block tiling

    # Corrected version: Compute (N, M, L) via fused tiling over K
    # We loop over M and K, and accumulate over K

    # We'll use a different approach: for each (n, m), compute sum over k of A[n, m, k] * B[k, l]
    # We'll use shared memory to cache B slices
    # We'll use 2D tiling: (M, K) and (K, L)

    # Reset block indices
    m_start = block_m * BLOCK_SIZE_M
    m_end = m_start + BLOCK_SIZE_M
    k_start = block_k * BLOCK_SIZE_K
    k_end = k_start + BLOCK_SIZE_K

    # Shared memory for B block (K, L)
    B_shared = tl.zeros((BLOCK_SIZE_K, L), dtype=tl.float16)

    # Load B block into shared memory
    for k in range(k_start, k_end):
        # Load B[k, :] into shared memory
        # B is (K, L), so we load row k
        k_idx = k
        # Load B[k, :] from global memory
        B_row = tl.load(B_ptr + k_idx * L, mask=(k_idx < K), other=0.0)
        # Store into shared memory
        B_shared[:, :] = B_row  # This is not correct

    # We need to fix the indexing

    # Let's use a different, correct tiling: compute one (M, L) block at a time
    # We'll loop over M and K, and accumulate over K

    # Correct tiling: for each (n, m), we compute sum_k A[n, m, k] * B[k, l]
    # We use shared memory to cache B[k, l] for k in current block

    # We'll do a fused kernel that computes one (M, L) block per block
    # We'll loop over m and l

    # Instead, let's do a clean, correct implementation using tiling over K
    # We will compute one (M, L) block per block, using shared memory for B

    # We are processing one block of M and one block of K
    # We will compute output for a fixed n

    # We need to fix the kernel structure

    # Final correct version: for each (n, m), compute sum_k A[n, m, k] * B[k, l]
    # We use tiling over K with shared memory for B

    # We process one block of M and one block of K
    # We compute output for a fixed n

    # Load B block into shared memory
    B_shared = tl.zeros((BLOCK_SIZE_K, L), dtype=tl.float16)
    for k in range(k_start, k_end):
        k_idx = k
        B_row = tl.load(B_ptr + k_idx * L, mask=(k_idx < K), other=0.0)
        B_shared[k - k_start, :] = B_row

    # Accumulate over k
    out = tl.zeros((M, L), dtype=tl.float16)
    for m in range(m_start, m_end):
        # Load A[n, m, :] for current m
        # A is (N, M, K), so we load A[block_n, m, k]
        A_vals = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float16)
        for k in range(k_start, k_end):
            k_idx = k
            a_val = tl.load(A_ptr + block_n * M * K + m * K + k_idx, mask=(k_idx < K), other=0.0)
            A_vals[k - k_start] = a_val
        # Multiply with B_shared and accumulate
        for l in range(L):
            sum_val = tl.dot(A_vals, B_shared[:, l])
            out[m - m_start, l] = sum_val

    # Store output
    # We need to store to correct location
    # out is (M, L), and we are computing for fixed n
    # So we store to out_ptr + block_n * M * L + m_start * L + l
    # But we are not storing in a contiguous way

    # Instead, we restructure to compute one (M, L) block per block
    # We'll store the output in a shared buffer and then write back

    # We will write the output in a single block
    # We compute one (M, L) block for fixed n
    # We store it in the output tensor

    # We'll write to output using a loop over m and l
    # But we need to handle memory layout

    # We will instead use a simpler, correct, and efficient kernel

    # Final clean implementation: compute (N, M, L) via tiling over K
    # We use shared memory for B
    # We compute one (M, L) block per block

    # Reset
    m_start = block_m * BLOCK_SIZE_M
    m_end = m_start + BLOCK_SIZE_M
    k_start = block_k * BLOCK_SIZE_K
    k_end = k_start + BLOCK_SIZE_K

    # Shared memory for B
    B_shared = tl.zeros((BLOCK_SIZE_K, L), dtype=tl.float16)

    # Load B into shared memory
    for k in range(k_start, k_end):
        k_idx = k
        B_row = tl.load(B_ptr + k_idx * L, mask=(k_idx < K), other=0.0)
        B_shared[k - k_start, :] = B_row

    # Accumulate output for current block
    out = tl.zeros((m_end - m_start, L), dtype=tl.float16)
    for m in range(m_start, m_end):
        # Load A[n, m, k] for current m
        A_vals = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float16)
        for k in range(k_start, k_end):
            k_idx = k
            a_val = tl.load(A_ptr + block_n * M * K + m * K + k_idx, mask=(k_idx < K), other=0.0)
            A_vals[k - k_start] = a_val
        # Multiply with B_shared
        for l in range(L):
            sum_val = tl.dot(A_vals, B_shared[:, l])
            out[m - m_start, l] = sum_val

    # Store output to global memory
    # We store in output: block_n * M * L + m_start * L + l
    # We loop over m and l
    for m in range(m_start, m_end):
        for l in range(L):
            idx = block_n * M * L + m * L + l
            tl.store(out_ptr + idx, out[m - m_start, l])


@triton.jit
def matmul_kernel_v2(
    A_ptr,
    B_ptr,
    out_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Compute block indices
    n = tl.program_id(0)
    m_start = tl.program_id(1) * BLOCK_SIZE_M
    m_end = m_start + BLOCK_SIZE_M
    k_start = tl.program_id(2) * BLOCK_SIZE_K
    k_end = k_start + BLOCK_SIZE_K

    # Shared memory for B block
    B_shared = tl.zeros((BLOCK_SIZE_K, L), dtype=tl.float16)

    # Load B block into shared memory
    for k in range(k_start, k_end):
        k_idx = k
        B_row = tl.load(B_ptr + k_idx * L, mask=(k_idx < K), other=0.0)
        B_shared[k - k_start, :] = B_row

    # Accumulate output
    out = tl.zeros((m_end - m_start, L), dtype=tl.float16)
    for m in range(m_start, m_end):
        # Load A[n, m, k] for current m
        A_vals = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float16)
        for k in range(k_start, k_end):
            k_idx = k
            a_val = tl.load(A_ptr + n * M * K + m * K + k_idx, mask=(k_idx < K), other=0.0)
            A_vals[k - k_start] = a_val
        # Multiply with B_shared
        for l in range(L):
            sum_val = tl.dot(A_vals, B_shared[:, l])
            out[m - m_start, l] = sum_val

    # Store output
    for m in range(m_start, m_end):
        for l in range(L):
            idx = n * M * L + m * L + l
            tl.store(out_ptr + idx, out[m - m_start, l])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Performs 3D tensor-matrix multiplication using a custom Triton kernel.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Ensure input shapes are correct
    assert A.shape[2] == B.shape[0], "Incompatible dimensions: A.shape[2] must equal B.shape[0]"

    N, M, K = A.shape
    K_out = B.shape[1]
    L = K_out

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 256

    # Compute grid
    grid = lambda meta: (
        (N,),  # n
        ((M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"]),  # m
        ((K + meta["BLOCK_SIZE_K"] - 1) // meta["BLOCK_SIZE_K"]),  # k
    )

    # Launch kernel
    matmul_kernel_v2[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B.data_ptr(),
        out_ptr=torch.empty(N, M, L, dtype=torch.float16, device=A.device).data_ptr(),
        N=N,
        M=M,
        K=K,
        L=L,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return torch.empty(N, M, L, dtype=torch.float16, device=A.device)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B):
        return triton_matmul(A, B)
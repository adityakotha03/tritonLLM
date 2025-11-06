import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, n, k,
    stride_A, stride_B, stride_C,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_CACHE: tl.constexpr
):
    # 3D grid: (batch, m_block, n_block)
    batch_id = tl.program_id(0)
    m_block_id = tl.program_id(1)
    n_block_id = tl.program_id(2)

    # Compute offsets
    m_start = m_block_id * BLOCK_M
    n_start = n_block_id * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    # Create masks for boundaries
    m_mask = m_offsets < m
    n_mask = n_offsets < n
    mask = m_mask[:, None] & n_mask[None, :]

    # Base pointers for A and B in this block
    A_block_ptr = A_ptr + batch_id * stride_A + m_block_id * BLOCK_M * stride_A
    B_block_ptr = B_ptr + batch_id * stride_B + n_block_id * BLOCK_N * stride_B

    # Shared memory for A and B tiles
    shared_A = tl.shared_memory(shape=(BLOCK_M, BLOCK_K), dtype=tl.bf16)
    shared_B = tl.shared_memory(shape=(BLOCK_K, BLOCK_N), dtype=tl.bf16)

    # Accumulator for C (BLOCK_M x BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over k tiles
    for k_idx in range(0, k, BLOCK_K):
        # Load A tile (BLOCK_M x BLOCK_K) from global memory to shared memory
        a_ptr = A_block_ptr + k_idx * stride_A
        a = tl.load(
            a_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_A + tl.arange(0, BLOCK_K)[None, :],
            mask=tl.arange(0, BLOCK_M)[:, None] * stride_A + tl.arange(0, BLOCK_K)[None, :] < m * k,
            cache=tl.LoadCache.L1 if USE_CACHE else tl.LoadCache.NONE,
            eviction=tl.EvictionPolicy.L1 if USE_CACHE else tl.EvictionPolicy.NONE
        )
        tl.store(shared_A, a, mask=tl.arange(0, BLOCK_M)[:, None] < m, cache=tl.StoreCache.L1)

        # Load B tile (BLOCK_K x BLOCK_N) from global memory to shared memory
        b_ptr = B_block_ptr + k_idx * stride_B
        b = tl.load(
            b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_B + tl.arange(0, BLOCK_N)[None, :],
            mask=tl.arange(0, BLOCK_K)[:, None] * stride_B + tl.arange(0, BLOCK_N)[None, :] < k * n,
            cache=tl.LoadCache.L1 if USE_CACHE else tl.LoadCache.NONE,
            eviction=tl.EvictionPolicy.L1 if USE_CACHE else tl.EvictionPolicy.NONE
        )
        tl.store(shared_B, b, mask=tl.arange(0, BLOCK_K)[:, None] < k, cache=tl.StoreCache.L1)

        # Compute dot product using shared memory tiles
        a = tl.load(shared_A, cache=tl.LoadCache.L1)
        b = tl.load(shared_B, cache=tl.LoadCache.L1)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    # Store result to global memory
    C_block_ptr = C_ptr + batch_id * stride_C + m_start * stride_C + n_start
    tl.store(
        C_block_ptr + tl.arange(0, BLOCK_M)[:, None] * stride_C + tl.arange(0, BLOCK_N)[None, :],
        acc,
        mask=mask
    )


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Tensors must be float32"
    assert A.shape[0] == B.shape[0], "Batch dimensions must match"
    
    batch_size, m, k = A.shape
    _, _, n = B.shape

    # Convert to bf16 for tensor core utilization
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    
    # Output tensor
    C = torch.empty(batch_size, m, n, dtype=torch.float32, device='cuda')

    # Calculate strides
    stride_A = A.stride(1)
    stride_B = B.stride(1)
    stride_C = C.stride(1)

    # Configure autotuning
    configs = [
        triton.autotune.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.autotune.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.autotune.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8),
        triton.autotune.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8),
    ]

    # Calculate grid
    grid = lambda meta: (batch_size, (m + meta['BLOCK_M'] - 1) // meta['BLOCK_M'], (n + meta['BLOCK_N'] - 1) // meta['BLOCK_N'])

    # Launch kernel
    bmm_kernel[grid](
        A_bf16, B_bf16, C,
        batch_size, m, n, k,
        stride_A, stride_B, stride_C,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        USE_CACHE=True
    )
    
    return C


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_bmm(A, B)
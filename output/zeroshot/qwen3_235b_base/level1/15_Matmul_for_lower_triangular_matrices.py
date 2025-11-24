import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _triton_matmul_lower_tri_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for blocks of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate in registers
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # We only compute the lower triangular part
    # So if the current output row < col, we skip
    # But we still need to launch all blocks for simplicity; we mask out invalid ones
    valid_mask = (offs_m[:, None] >= offs_n[None, :])

    # We iterate over K in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load tiles of A and B
        a = tl.load(a_ptrs, mask=(offs_m[:, None] >= k + offs_k[None, :]) & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] >= k + offs_k[:, None]) & (offs_n[None, :] < N), other=0.0)

        # Dot product
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Only write lower triangular part
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=valid_mask & (offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_lower_tri(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda, "Inputs must be on GPU"
    assert A.shape == B.shape, "A and B must have same shape"
    assert A.dim() == 2, "Only 2D matrices supported"

    M, K = A.shape
    K, N = B.shape
    assert K == N, "Incompatible dimensions"

    # Output tensor
    C = torch.empty_strided((M, N), (N, 1), device=A.device, dtype=A.dtype)

    # Lower triangular mask
    tri_mask = torch.tril(torch.ones(M, N, device=A.device)).bool()
    
    # Use Tensor Core with mixed precision for better performance
    A = A.to(torch.float16)
    B = B.to(torch.float16)

    # Launch grid
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    num_stages = 4
    num_warps = 4

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    _triton_matmul_lower_tri_kernel[grid](
        A, B, C,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    # Mask out upper triangle
    C = C.masked_fill(~tri_mask, 0.0)
    return C.to(A.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return triton_matmul_lower_tri(A, B)
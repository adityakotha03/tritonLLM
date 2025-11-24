import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for matmul with small K (K=64) using fp16 tensor cores
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,  # [M, K]
    B_ptr,  # [K, N]
    C_ptr,  # [M, N]
    M, N, K,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # compute the starting index for this block
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Allocate accumulator in registers
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over K in tiles
    for k_start in range(0, K, BLOCK_K):
        # Load tiles of A and B
        a_offsets = (
            (m_start[:, None] + tl.arange(0, BLOCK_M)[None, :]) * stride_am
            + k_start * stride_an
        )
        b_offsets = (
            k_start * stride_bm
            + (n_start[None, :] + tl.arange(0, BLOCK_N)[None, :]) * stride_bn
        )

        a = tl.load(A_ptr + a_offsets, mask=(m_start[:, None] + tl.arange(0, BLOCK_M)[None, :]) < M, other=0.0).to(tl.float16)
        b = tl.load(B_ptr + b_offsets, mask=(n_start[None, :] + tl.arange(0, BLOCK_N)[None, :]) < N, other=0.0).to(tl.float16)

        # Matrix multiply accumulation (tensor core friendly)
        acc += tl.dot(a, b, allow_tf32=False)

    # Write results
    c_offsets = (
        (m_start[:, None] + tl.arange(0, BLOCK_M)[None, :]) * stride_cm
        + (n_start[None, :] + tl.arange(0, BLOCK_N)[None, :]) * stride_cn
    )
    tl.store(C_ptr + c_offsets, acc.to(tl.float16), mask=(m_start[:, None] + tl.arange(0, BLOCK_M)[None, :]) < M & (n_start[None, :] + tl.arange(0, BLOCK_N)[None, :]) < N)

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for Triton matmul kernel.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype == torch.float16 or A.dtype == torch.bfloat16, "A must be fp16 or bf16."
    assert B.dtype == torch.float16 or B.dtype == torch.bfloat16, "B must be fp16 or bf16."
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match."

    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return C

class ModelNew(nn.Module):
    """
    Optimized model that performs a single matrix multiplication using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are fp16 for tensor core acceleration
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)
        return triton_matmul(A, B)
# Best Kernel for 6_Matmul_with_large_K_dimension_
# Generated: 20251116_150532
# Speedup: 2.80x
# Runtime: 1.3800 ms
# Round: 1
# Idea: Split-K parallel reduction to saturate SMs and improve latency hiding - Strategy: Split the giant K into split_k chunks (e.g., 16–64), launch CTAs over (M_tile, N_tile, K_chunk), each CTA computes a partial C tile over its K range, then reduce across K-chunks. Use either: - One-pass atomicAdd on C in FP32 at the end of each CTA, or - Two-pass: write partials to a scratch buffer [M, N, split_k] then launch a lightweight reduction kernel. For this problem size (256x256), scratch with split_k=32 is ~8.4 MB (manageable). - Example tiling: BLOCK_M=128, BLOCK_N=128 → base grid is 2x2=4 CTAs; with split_k=32 → 128 CTAs, enough to keep 80–108 SMs busy. Tune split_k to maintain high occupancy without excessive contention. - Why on A100: Without split-K, only 4 CTAs exist; that underutilizes an A100. Split-K creates sufficient parallel work to hide memory latency and better utilize the 64K registers/SM and 163 KB SMEM. Ampere’s FP32 atomics are fast enough for this scale, and L2 can service concurrent CTAs efficiently. - Targets: Parallelism & occupancy, latency hiding (via more CTAs), better SM utilization.

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_splitk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    K_CHUNK,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over this split's K-chunk with fixed iteration count
    for kk in range(0, K_CHUNK, BLOCK_K):
        k0 = pid_s * K_CHUNK + kk
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def _cdiv(a, b):
    return (a + b - 1) // b


def triton_matmul_splitk(A: torch.Tensor, B: torch.Tensor, split_k: int = 32,
                         BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 is supported"
    assert A.shape[1] == B.shape[0], "Incompatible shapes for matmul"

    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    A = A.contiguous()
    B = B.contiguous()

    C = torch.zeros((M, N), device=A.device, dtype=torch.float32)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    K_CHUNK = _cdiv(K, split_k)

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), split_k)

    matmul_splitk_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        K_CHUNK,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )
    return C


class ModelNew(nn.Module):
    """
    Optimized model using Triton split-K GEMM with atomic reduction.
    """
    def __init__(self, split_k: int = 32, block_m: int = 128, block_n: int = 128, block_k: int = 64):
        super().__init__()
        self.split_k = split_k
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        orig_device = A.device
        # Move to CUDA and ensure float32 for atomicAdd support and best precision
        A_cuda = A.to(device='cuda', dtype=torch.float32, copy=False).contiguous()
        B_cuda = B.to(device='cuda', dtype=torch.float32, copy=False).contiguous()

        C_cuda = triton_matmul_splitk(
            A_cuda, B_cuda,
            split_k=self.split_k,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k
        )
        # Return on original device
        if orig_device.type == 'cuda':
            return C_cuda
        return C_cuda.to(orig_device)
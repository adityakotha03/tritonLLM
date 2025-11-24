import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton matrix‑multiply kernel ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the current tile
    i = pid_m * BLOCK_M
    j = pid_n * BLOCK_N
    i_offsets = i + tl.arange(0, BLOCK_M)[:, None]          # (BLOCK_M, 1)
    j_offsets = j + tl.arange(0, BLOCK_N)[None, :]          # (1, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets_a = k + tl.arange(0, BLOCK_K)[None, :]    # (1, BLOCK_K)
        k_offsets_b = k + tl.arange(0, BLOCK_K)[:, None]    # (BLOCK_K, 1)

        # Load tile of A
        A_tile = tl.load(
            A_ptr + i_offsets * stride_am + k_offsets_a * stride_ak,
            mask=(i_offsets < M) & (k_offsets_a < K),
            other=0.0,
        ).to(tl.float16)

        # Load tile of B
        B_tile = tl.load(
            B_ptr + k_offsets_b * stride_bk + j_offsets * stride_bn,
            mask=(k_offsets_b < K) & (j_offsets < N),
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(A_tile, B_tile)

    C_ptr_offsets = i_offsets * stride_cm + j_offsets * stride_cn
    tl.store(
        C_ptr + C_ptr_offsets,
        acc,
        mask=(i_offsets < M) & (j_offsets < N),
    )


# ---------- Wrapper for the Triton kernel ----------
def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        M=M,
        N=N,
        K=K,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
    )
    return C


# ---------- Optimized model ----------
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
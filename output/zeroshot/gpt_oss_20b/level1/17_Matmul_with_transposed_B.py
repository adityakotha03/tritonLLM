import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_b_n,
    stride_b_k,
    stride_c_m,
    stride_c_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    row_start = pid_x * BLOCK_SIZE_M
    col_start = pid_y * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(
            A_ptr
            + (row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_a_m
            + (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_a_k,
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) < M,
            other=0.0,
        )
        B = tl.load(
            B_ptr
            + (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_b_n
            + (k + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_b_k,
            mask=(col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N,
            other=0.0,
        )
        acc += tl.dot(A, B)

    tl.store(
        C_ptr
        + (row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) * stride_c_m
        + (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) * stride_c_n,
        acc,
        mask=((row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]) < M)
        & ((col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N),
    )


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Custom Triton implementation of A @ B.T where
    A: (M, K) and B: (N, K)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on GPU"
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, "Inner dimensions must match"

    # Transpose B to get (K, N) shape for multiplication
    B_T = B.t().contiguous()

    out = torch.empty((M, N), dtype=A.dtype, device=A.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    _matmul[grid](
        A_ptr=A.data_ptr(),
        B_ptr=B_T.data_ptr(),
        C_ptr=out.data_ptr(),
        M=M,
        N=N,
        K=K,
        stride_a_m=A.stride(0),
        stride_a_k=A.stride(1),
        stride_b_n=B_T.stride(0),
        stride_b_k=B_T.stride(1),
        stride_c_m=out.stride(0),
        stride_c_n=out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel for matrix multiplication.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
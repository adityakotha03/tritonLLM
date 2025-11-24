import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # Compute the starting block coordinates
    block_row = pid // ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    block_col = pid % ((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)

    # Offsets for the current block
    row_offset = block_row * BLOCK_SIZE_M
    col_offset = block_col * BLOCK_SIZE_N

    # Shared accumulators
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load tile of A
        a_row = row_offset + tl.arange(0, BLOCK_SIZE_M)
        a_col = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + a_row[:, None] * stride_am + a_col[None, :] * stride_an
        a = tl.load(a_ptrs, mask=(a_row[:, None] < M) & (a_col[None, :] < K), other=0.0)

        # Load tile of B
        b_row = k + tl.arange(0, BLOCK_SIZE_K)
        b_col = col_offset + tl.arange(0, BLOCK_SIZE_N)
        b_ptrs = B_ptr + b_row[:, None] * stride_bm + b_col[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(b_row[:, None] < K) & (b_col[None, :] < N), other=0.0)

        # Matrix multiply
        acc += tl.dot(a, b)

    # Write the result
    c_row = row_offset + tl.arange(0, BLOCK_SIZE_M)
    c_col = col_offset + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + c_row[:, None] * stride_cm + c_col[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(c_row[:, None] < M) & (c_col[None, :] < N))


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of matrix multiplication for contiguous float tensors.
    Supports float32 and float16 inputs. Uses tensor‑core friendly dot product.
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA."
    assert A.dtype in (torch.float32, torch.float16), "Unsupported dtype for A"
    assert B.dtype in (torch.float32, torch.float16), "Unsupported dtype for B"
    assert A.shape[1] == B.shape[0], "Inner dimensions must match"

    M, K = A.shape
    K_, N = B.shape
    dtype = torch.float32  # compute in fp32 for stability

    A_cast = A.to(dtype)
    B_cast = B.to(dtype)

    C = torch.empty((M, N), dtype=dtype, device=A.device)

    stride_am = A_cast.stride(0)
    stride_an = A_cast.stride(1)
    stride_bm = B_cast.stride(0)
    stride_bn = B_cast.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    grid = lambda meta: ((M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"]) * \
                        ((N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"])

    _matmul_kernel[grid](
        A_cast,
        B_cast,
        C,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bm,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=128,
    )

    # Cast back if inputs were fp16
    if A.dtype == torch.float16 or B.dtype == torch.float16:
        return C.to(A.dtype)
    return C


class ModelNew(nn.Module):
    """
    Optimized model that replaces torch.matmul with a Triton kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
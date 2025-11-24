import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------- Triton kernels -----------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        A = tl.load(
            A_ptr + (block_m + tl.arange(0, BLOCK_M))[:, None] * stride_am
            + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
            mask=(block_m + tl.arange(0, BLOCK_M))[:, None] < M
            & (k + tl.arange(0, BLOCK_K))[None, :] < K,
            other=0.0,
        )
        B = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk
            + (block_n + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
            mask=(k + tl.arange(0, BLOCK_K))[:, None] < K
            & (block_n + tl.arange(0, BLOCK_N))[None, :] < N,
            other=0.0,
        )
        acc += tl.dot(A, B)

    C = acc.to(tl.float16)
    tl.store(
        C_ptr + (block_m + tl.arange(0, BLOCK_M))[:, None] * stride_cm
        + (block_n + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
        C,
        mask=(block_m + tl.arange(0, BLOCK_M))[:, None] < M
        & (block_n + tl.arange(0, BLOCK_N))[None, :] < N,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def avgpool_kernel(
    x_ptr,
    out_ptr,
    N,
    pool_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # each thread sums over its pool window
    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(0, pool_k):
        sum += tl.load(x_ptr + offsets * pool_k + k, mask=mask, other=0.0)
    avg = sum / pool_k
    tl.store(out_ptr + offsets, avg, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # approximate GELU
    gelu = 0.5 * x * (1 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * tl.math.pow(x, 3))))
    tl.store(out_ptr + offsets, gelu, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def scale_kernel(
    x_ptr,
    out_ptr,
    N,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


# ----------------- Helper functions -----------------

def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
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
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return C


def triton_avgpool(x: torch.Tensor, pool_k: int) -> torch.Tensor:
    N = x.shape[0]
    out = torch.empty((N // pool_k), dtype=x.dtype, device=x.device)
    grid = lambda meta: ((N // pool_k + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    avgpool_kernel[grid](x, out, N, pool_k, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x, out, N, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out


def triton_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    N = x.shape[0]
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    scale_kernel[grid](x, out, N, scale=scale, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out


# ----------------- Optimized model -----------------

class ModelNew(nn.Module):
    """
    Optimized implementation of the pattern
    "Matmul_AvgPool_GELU_Scale_Max" using custom Triton kernels.
    """

    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
        )
        self.bias = nn.Parameter(
            torch.randn(out_features, device="cuda", dtype=torch.float16)
        )
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        """
        x: (batch_size, in_features) float16 tensor on CUDA
        returns: (batch_size,) float16 tensor
        """
        # Matmul
        matmul_out = triton_matmul(x, self.weight.t())  # (B, out_features)
        matmul_out = matmul_out + self.bias  # add bias

        # AvgPool1d over feature dimension
        # reshape to (B * out_features) for convenience
        B, N = matmul_out.shape
        flat = matmul_out.reshape(-1)
        pooled = triton_avgpool(flat, self.pool_kernel_size)
        pooled = pooled.reshape(B, -1)  # (B, out_features // pool)

        # GELU
        gelu_out = triton_gelu(pooled)

        # Scale
        scaled = triton_scale(gelu_out, self.scale_factor)

        # Max reduction over the feature dimension
        max_vals, _ = torch.max(scaled, dim=1)
        return max_vals
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel: MatMul + Mish + Mish (fused)
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_mish2_kernel(
    A_ptr,
    B_ptr,
    bias_ptr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Allocate accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load tiles of A and B
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float16)

        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & mask_n[None, :],
            other=0.0,
        ).to(tl.float16)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + bias

    # Apply Mish twice
    def mish(x):
        return x * tl.math.tanh(tl.math.softplus(x))
    acc = mish(acc)
    acc = mish(acc)

    # Store result
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.float16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ------------------------------------------------------------------
# Wrapper functions
# ------------------------------------------------------------------
def matmul_mish2(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """
    A: (B, M, K)  ->  fp16
    B: (K, N)     ->  fp16
    bias: (N,)    ->  fp16
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert bias.dtype == torch.float16

    B_t = B.t()
    M, K = A.shape[1], A.shape[2]
    N = B_t.shape[1]

    C = torch.empty((A.shape[0], M, N), dtype=torch.float16, device=A.device)

    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128), A.shape[0])
    matmul_mish2_kernel[grid](
        A,
        B_t,
        bias,
        C,
        M,
        N,
        K,
        A.stride(1),
        A.stride(2),
        B_t.stride(0),
        B_t.stride(1),
        C.stride(1),
        C.stride(2),
    )
    return C


# ------------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernel for matmul + Mish + Mish.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Linear layer parameters stored as float16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float16, device="cuda"))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16, device="cuda"))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)  -> float16
        """
        assert x.is_cuda
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        # Perform fused matmul + mish + mish
        y = matmul_mish2(x, self.weight, self.bias)
        return y


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def get_inputs():
    return [torch.rand(1024, 8192, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [8192, 8192]
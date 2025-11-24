import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
#  Triton kernel:  matmul + bias addition + ReLU (fused)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_bias_relu_kernel(
    A_ptr,  # (M, K)
    B_ptr,  # (K, N)
    bias_ptr,  # (N,)
    out_ptr,  # (M, N)
    M, N, K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Shared accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        block_k = min(BLOCK_K, K - k)

        # Load tiles from A and B
        A_block = tl.load(
            A_ptr + (start_m + tl.arange(0, BLOCK_M))[:, None] * stride_am
                     + (k + tl.arange(0, block_k))[None, :] * stride_ak,
            mask=(start_m + tl.arange(0, BLOCK_M)[:, None] < M)
                & (k + tl.arange(0, block_k)[None, :] < K),
            other=0.0,
        )

        B_block = tl.load(
            B_ptr + (k + tl.arange(0, block_k))[:, None] * stride_bk
                     + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
            mask=(k + tl.arange(0, block_k)[:, None] < K)
                & (start_n + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(A_block, B_block)

    # Add bias and apply ReLU
    bias = tl.load(bias_ptr + start_n + tl.arange(0, BLOCK_N),
                   mask=start_n + tl.arange(0, BLOCK_N) < N,
                   other=0.0)

    acc += bias[None, :]

    acc = tl.maximum(acc, 0.0)

    # Store results
    tl.store(
        out_ptr + (start_m + tl.arange(0, BLOCK_M))[:, None] * stride_outm
                 + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_outn,
        acc,
        mask=(start_m + tl.arange(0, BLOCK_M)[:, None] < M)
            & (start_n + tl.arange(0, BLOCK_N)[None, :] < N),
    )


def gemm_bias_relu(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor | None = None,
):
    """
    A: (M, K)  float32 or bfloat16
    B: (K, N)  float32 or bfloat16
    bias: (N,)  float32
    Returns: (M, N) float32
    """
    assert A.is_cuda and B.is_cuda and bias.is_cuda
    assert A.shape[1] == B.shape[0]

    M, K = A.shape
    K, N = B.shape

    if out is None:
        out = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Compute strides
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_outm, stride_outn = out.stride()

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32  # default, autotuner will adjust

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _gemm_bias_relu_kernel[grid](
        A,
        B,
        bias,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_outm,
        stride_outn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


# --------------------------------------------------------------------------- #
#  Optimized model using the fused Triton kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized model: GEMM + bias + ReLU fused in a single Triton kernel.
    """

    def __init__(self, in_features: int, out_features: int, bias_shape: tuple[int]):
        super().__init__()
        # Weight matrix (in_features x out_features), store as bfloat16 for Tensor Core usage
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, dtype=torch.bfloat16, device="cuda")
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Bias vector
        self.bias = nn.Parameter(torch.empty(bias_shape, dtype=torch.float32, device="cuda"))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_features) float32
        """
        assert x.is_cuda
        # Ensure weight is bfloat16
        A = x.to(torch.bfloat16)
        B = self.weight  # (in_features, out_features)
        out = gemm_bias_relu(A, B, self.bias)
        return out

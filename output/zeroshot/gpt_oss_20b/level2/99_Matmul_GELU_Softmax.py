import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# --------------------------------------------------------------------------- #
#          Triton kernels: fused matmul + GELU, and softmax                #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Fused matrix multiplication + GELU
# --------------------------------------------------------------------------- #
# Parameters:
#   - BLOCK_SIZE_M: tile size for the batch dimension
#   - BLOCK_SIZE_N: tile size for the output feature dimension
#   - BLOCK_SIZE_K: tile size for the shared dimension
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128}, num_warps=6),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_matmul_gelu(
    # Pointers
    input_ptr,       # [M, K]
    weight_ptr,      # [K, N]  (weight transposed)
    bias_ptr,        # [N]
    output_ptr,      # [M, N]
    # Sizes
    M, N, K,
    # Compile time constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Grid dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Mask for out-of-bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialise accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Main loop over K tiles
    for offs_k in range(0, K, BLOCK_SIZE_K):
        # Load A tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a = tl.load(
            input_ptr + (offs_m[:, None] * K + (offs_k + tl.arange(0, BLOCK_SIZE_K))),
            mask=mask_m[:, None] & (offs_k + tl.arange(0, BLOCK_SIZE_K) < K),
            other=0.0,
            dtype=tl.bfloat16,
        ).to(tl.float32)

        # Load B tile [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b = tl.load(
            weight_ptr + ((offs_k + tl.arange(0, BLOCK_SIZE_K)) * N + offs_n[None, :]),
            mask=(offs_k + tl.arange(0, BLOCK_SIZE_K) < K) & mask_n[None, :],
            other=0.0,
            dtype=tl.bfloat16,
        ).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0, dtype=tl.float32)
    acc += bias[None, :]

    # GELU approximation (fast)
    acc = acc * 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)))

    # Store result in BF16
    tl.store(
        output_ptr + (offs_m[:, None] * N + offs_n[None, :]),
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )

# --------------------------------------------------------------------------- #
# Softmax along dim=1 (row-wise)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=12),
    ],
    key=["N"],
)
@triton.jit
def softmax_kernel(
    input_ptr,   # [M, N] (BF16)
    output_ptr,  # [M, N] (BF16)
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # each program handles one row
    row = pid

    # Offsets for this row
    offs = tl.arange(0, BLOCK_SIZE)

    # Compute max for numerical stability
    max_val = tl.float32(-1e9)
    for i in range(0, N, BLOCK_SIZE):
        cur_offs = i + offs
        mask = cur_offs < N
        x = tl.load(input_ptr + row * N + cur_offs, mask=mask, other=0.0, dtype=tl.bfloat16).to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x, axis=0))

    # Compute exponentials and sum
    sum_exp = tl.float32(0.0)
    for i in range(0, N, BLOCK_SIZE):
        cur_offs = i + offs
        mask = cur_offs < N
        x = tl.load(input_ptr + row * N + cur_offs, mask=mask, other=0.0, dtype=tl.bfloat16).to(tl.float32)
        e = tl.exp(x - max_val)
        tl.store(output_ptr + row * N + cur_offs, e.to(tl.bfloat16), mask=mask)
        sum_exp = tl.sum(e, axis=0, mask=mask) + sum_exp

    # Normalize
    inv_sum = 1.0 / sum_exp
    for i in range(0, N, BLOCK_SIZE):
        cur_offs = i + offs
        mask = cur_offs < N
        e = tl.load(output_ptr + row * N + cur_offs, mask=mask, other=0.0, dtype=tl.bfloat16).to(tl.float32)
        tl.store(output_ptr + row * N + cur_offs, (e * inv_sum).to(tl.bfloat16), mask=mask)

# --------------------------------------------------------------------------- #
# Wrapper functions for PyTorch
# --------------------------------------------------------------------------- #
def triton_fused_linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    x: [M, K] BF16
    weight: [K, N] BF16 (already transposed)
    bias: [N] BF16
    returns y: [M, N] BF16
    """
    M, K = x.shape
    K2, N = weight.shape
    assert K == K2

    # Allocate output
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    # Grid dimensions
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    fused_matmul_gelu[grid](
        x,
        weight,
        bias,
        out,
        M, N, K,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out

def triton_softmax(x: torch.Tensor):
    """
    x: [M, N] BF16
    returns softmax over dim=1
    """
    M, N = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (M,)
    softmax_kernel[grid](x, out, M, N, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out

# --------------------------------------------------------------------------- #
# Model definition using Triton kernels
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Use bf16 weights for better performance
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda"))
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device="cuda"))

    def forward(self, x: torch.Tensor):
        # Ensure BF16 dtype
        x = x.to(torch.bfloat16)
        # Fused linear + GELU
        y = triton_fused_linear_gelu(x, self.weight.t(), self.bias)
        # Softmax
        y = triton_softmax(y)
        return y
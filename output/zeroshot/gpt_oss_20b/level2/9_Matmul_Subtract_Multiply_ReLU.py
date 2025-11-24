import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel that fuses: linear (matmul + bias), subtract, multiply, ReLU
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 512}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_fused_kernel(
    X_ptr,          # [M, K] input
    W_ptr,          # [N, K] weight (transpose)
    B_ptr,          # [N] bias
    Y_ptr,          # output [M, N]
    subtract_val,   # scalar to subtract
    multiply_val,   # scalar to multiply
    M, N, K,       # dimensions
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Load tiles of X and W
        X_tile = tl.load(X_ptr + (offs_m[:, None] * K + offs_k[None, :]), mask=offs_m[:, None] < M, other=0.0)
        W_tile = tl.load(W_ptr + (offs_n[:, None] * K + offs_k[None, :]), mask=offs_n[:, None] < N, other=0.0)

        acc += tl.dot(X_tile, W_tile, allow_tf32=False)

    # Add bias
    if offs_n[None, 0] < N:
        bias = tl.load(B_ptr + offs_n[None, 0])
        acc += bias

    # Subtract, multiply, ReLU
    acc = acc - subtract_val
    acc = acc * multiply_val
    acc = tl.maximum(acc, 0.0)

    # Store result
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    tl.store(Y_ptr + (offs_m[:, None] * N + offs_n[None, :]), acc, mask=mask_m & mask_n)


def linear_fused(x, weight, bias, subtract_val, multiply_val):
    """
    Wrapper for the fused kernel. Assumes x is [M, K], weight is [N, K] (pre-transposed),
    bias is [N].
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = bias.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Transpose weight to shape [N, K] for the kernel
    weight_t = weight.t()

    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    linear_fused_kernel[grid](
        x, weight_t, bias, out,
        subtract_val, multiply_val,
        M, N, K,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation,
    all fused into a single Triton kernel.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Use fused kernel: input x, weight, bias
        out = linear_fused(
            x,
            self.linear.weight,
            self.linear.bias,
            self.subtract_value,
            self.multiply_value,
        )
        return out
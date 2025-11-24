import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=4),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def matmul_bias_sum_fused_kernel(
    X_ptr,    # (M, K) input matrix
    W_ptr,    # (N, K) weight matrix (transposed)
    b_ptr,    # (N,) bias vector
    Y_ptr,    # (M,) output scalar per row (sum over features)
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute Y_i = sum_j ( X_i,j * W_k,j + b_k )
    Then sum over k to get a single scalar per i.
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    row_end = min(row_start + BLOCK_M, M)

    # Accumulator for each row
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_end = min(k + BLOCK_K, K)

        # Load tile of X: shape (BLOCK_M, BLOCK_K)
        X_tile = tl.load(
            X_ptr + row_start[:, None] * K + tl.arange(0, BLOCK_K)[None, :],
            mask=(row_start[:, None] < M) & (tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )

        # Load tile of W: shape (BLOCK_K, N)
        W_tile = tl.load(
            W_ptr + tl.arange(0, BLOCK_K)[:, None] * N + None,
            mask=tl.arange(0, BLOCK_K)[:, None] < K,
            other=0.0,
        )  # broadcast N

        # Multiply and accumulate: (BLOCK_M, BLOCK_K) x (BLOCK_K, N) -> (BLOCK_M, N)
        prod = tl.dot(X_tile, W_tile)  # (BLOCK_M, N)

        # Add bias
        prod += tl.load(b_ptr + None, mask=None, other=0.0)

        # Sum over N to get scalar per row
        acc += tl.sum(prod, dim=1)

    # Store result
    tl.store(Y_ptr + row_start, acc, mask=(row_start < M))


def linear_sum_fused(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Perform linear transformation followed by sum over features.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M = x.size(0)
    K = x.size(1)
    N = weight.size(0)

    out = torch.empty((M,), dtype=torch.float32, device=x.device)

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    grid = lambda meta: ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"], )

    matmul_bias_sum_fused_kernel[grid](
        x,
        weight.t(),   # transpose to (N, K)
        bias,
        out,
        M, N, K,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using a custom Triton kernel for the linear
    transformation followed by summation. All subsequent operations
    (max, mean, logsumexp) reduce a single scalar and thus become
    no-ops.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute linear + sum in one fused kernel
        x = linear_sum_fused(x, self.linear.weight, self.linear.bias)  # shape (batch,)
        # Remaining operations are identity on a single scalar per sample
        x = x.unsqueeze(1)  # shape (batch, 1)
        return x
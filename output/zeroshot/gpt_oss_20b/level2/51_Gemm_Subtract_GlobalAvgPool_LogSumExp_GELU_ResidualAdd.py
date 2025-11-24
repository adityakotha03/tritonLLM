import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# ---------- Linear + Subtract ---------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Compute the program id in the grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Start row/col index for this block
    row = pid_m * BLOCK_M
    col = pid_n * BLOCK_N

    # Allocate accumulators for the C matrix
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Load A and B tiles
        a = tl.load(
            a_ptr + (row + tl.arange(0, BLOCK_M))[:, None] * K + (k + tl.arange(0, BLOCK_K)),
            mask=(row + tl.arange(0, BLOCK_M)[:, None] < M)
            & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + (col + tl.arange(0, BLOCK_N)),
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K)
            & (col + tl.arange(0, BLOCK_N)[None, :] < N),
            other=0.0,
        )

        # Accumulate product
        acc += tl.dot(a, b)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + col + tl.arange(0, BLOCK_N))
        acc += bias[None, :]

    # Store the result
    tl.store(
        c_ptr + (row + tl.arange(0, BLOCK_M))[:, None] * N + (col + tl.arange(0, BLOCK_N)),
        acc,
        mask=(row + tl.arange(0, BLOCK_M)[:, None] < M)
        & (col + tl.arange(0, BLOCK_N)[None, :] < N),
    )


def triton_linear(a: torch.Tensor, w: torch.Tensor, b: torch.Tensor = None):
    """
    Perform a linear transformation using a Triton matmul kernel.
    a: [batch, in_features]
    w: [in_features, out_features]
    b: [out_features] or None
    """
    a = a.contiguous()
    w = w.contiguous()
    batch, in_feat = a.shape
    out_feat = w.shape[1]

    # Output tensor
    out = torch.empty((batch, out_feat), device=a.device, dtype=a.dtype)

    # Grid shape
    grid = lambda meta: (
        (batch + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (out_feat + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=w.data_ptr(),
        c_ptr=out.data_ptr(),
        bias_ptr=b.data_ptr() if b is not None else None,
        M=batch,
        N=out_feat,
        K=in_feat,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out


@triton.jit
def subtract_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x - y, mask=mask)


def triton_subtract(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    subtract_kernel[grid](x.data_ptr(), y.data_ptr(), out.data_ptr(), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------- Global Reduction (mean, logsumexp, gelu) ---------- #
@triton.jit
def reduce_kernel(
    in_ptr,
    out_ptr,  # scalar output
    batch: tl.constexpr,
    feat: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs:
      * sum over batch -> sum_x[feat]
      * mean = sum_x / batch
      * logsumexp(mean) = log(sum(exp(mean)))
      * gelu(logsumexp)
    Result is a single scalar written to out_ptr.
    """
    # Accumulate sums per feature
    sum_x = tl.zeros((feat,), dtype=tl.float32)
    for i in range(0, batch, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < batch
        batch_slice = tl.load(in_ptr + offsets[:, None] * feat + tl.arange(0, feat), mask=mask[:, None], other=0.0)
        sum_x += tl.sum(batch_slice, axis=0)

    # Mean
    mean = sum_x / tl.float32(batch)

    # LogSumExp
    max_val = tl.max(mean)
    exp_sum = tl.sum(tl.exp(mean - max_val))
    logsumexp = max_val + tl.math.log(exp_sum)

    # GELU approximation (torch.nn.functional.gelu)
    coeff = 0.044715
    sqrt2_over_pi = 0.7978845608028654
    tanh_term = tl.math.tanh(
        sqrt2_over_pi * (logsumexp + coeff * tl.math.pow(logsumexp, 3))
    )
    gelu_out = 0.5 * logsumexp * (1.0 + tanh_term)

    tl.store(out_ptr, gelu_out, mask=1)


def triton_reduce(x: torch.Tensor):
    batch, feat = x.shape
    out = torch.empty((1,), device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 256
    grid = lambda meta: (1,)
    reduce_kernel[grid](x.data_ptr(), out.data_ptr(), batch, feat, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------- Final Model ---------- #
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device="cuda"))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, device="cuda"))
        else:
            self.bias = None
        self.subtract = nn.Parameter(torch.randn(out_features, device="cuda"))

    def forward(self, x):
        # Keep original input for residual addition
        original_x = x.clone().detach()

        # Linear transformation (matmul + bias)
        linear_out = triton_linear(x, self.weight, self.bias)

        # Subtract learned vector
        sub_out = triton_subtract(linear_out, self.subtract)

        # Global reduction to scalar
        scalar = triton_reduce(sub_out)

        # Broadcast scalar and add to original input
        out = original_x + scalar
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    input_ptr,           # Input tensor (batch, in_features)
    weight_ptr,          # Weight matrix (in_features, out_features)
    bias_ptr,            # Bias vector (out_features)
    output_ptr,          # Output tensor (batch, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute the block of output to process
    batch_idx = tl.program_id(0)
    # Each thread block processes one batch element
    # We compute a block of M x N matrix multiplication
    # M = in_features, N = out_features
    # We use a tiling strategy to reduce global memory accesses

    # Compute the row and column indices for the block
    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    col_offsets = tl.arange(0, BLOCK_SIZE_N)

    # Load weights and input in tiles
    # Input: (batch, in_features) -> tile (batch, BLOCK_SIZE_M)
    # Weights: (in_features, out_features) -> tile (BLOCK_SIZE_M, BLOCK_SIZE_N)

    # Load input for this batch
    input_batch = tl.load(input_ptr + batch_idx * in_features + row_offsets, mask=row_offsets < in_features, other=0.0)

    # Load weights in tiles
    weights = tl.load(weight_ptr + row_offsets[:, None] * out_features + col_offsets[None, :], mask=(row_offsets < in_features) & (col_offsets < out_features), other=0.0)

    # Compute dot product
    acc = tl.dot(input_batch, weights)

    # Add bias
    bias = tl.load(bias_ptr + col_offsets, mask=col_offsets < out_features, other=0.0)
    acc = acc + bias

    # Store output
    output_idx = batch_idx * out_features + col_offsets
    tl.store(output_ptr + output_idx, acc, mask=col_offsets < out_features)


@triton.jit
def subtract_kernel(
    x_ptr,                # Input tensor (batch, out_features)
    subtract_ptr,         # Subtract vector (out_features)
    out_ptr,              # Output tensor (batch, out_features)
    batch_size: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_features
    x = tl.load(x_ptr + batch_idx * out_features + offsets, mask=mask, other=0.0)
    subtract_val = tl.load(subtract_ptr + offsets, mask=mask, other=0.0)
    out = x - subtract_val
    tl.store(out_ptr + batch_idx * out_features + offsets, out, mask=mask)


@triton.jit
def global_avg_pool_kernel(
    x_ptr,                # Input tensor (batch, out_features)
    out_ptr,              # Output tensor (batch, 1)
    batch_size: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_features
    x = tl.load(x_ptr + batch_idx * out_features + offsets, mask=mask, other=0.0)
    mean_val = tl.sum(x, axis=0) / out_features
    tl.store(out_ptr + batch_idx, mean_val, mask=mask)


@triton.jit
def logsumexp_kernel(
    x_ptr,                # Input tensor (batch, 1)
    out_ptr,              # Output tensor (batch, 1)
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 1
    x = tl.load(x_ptr + batch_idx, mask=mask, other=0.0)
    # LogSumExp: log(sum(exp(x)))
    # We compute exp(x) and sum, then take log
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x)
    out = tl.log(sum_exp)
    tl.store(out_ptr + batch_idx, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,                # Input tensor (batch, 1)
    out_ptr,              # Output tensor (batch, 1)
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 1
    x = tl.load(x_ptr + batch_idx, mask=mask, other=0.0)
    # GELU: x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # We use a stable approximation
    sqrt_2_over_pi = 0.7978845608
    x3 = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_val = tl.tanh(tanh_arg)
    out = x * (1 + tanh_val)
    tl.store(out_ptr + batch_idx, out, mask=mask)


@triton.jit
def residual_add_kernel(
    x_ptr,                # Input tensor (batch, 1)
    original_x_ptr,       # Original input tensor (batch, in_features)
    out_ptr,              # Output tensor (batch, in_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < in_features
    # Load original x and current x
    original_x = tl.load(original_x_ptr + batch_idx * in_features + offsets, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + batch_idx, mask=mask, other=0.0)
    out = original_x + x_val
    tl.store(out_ptr + batch_idx * in_features + offsets, out, mask=mask)


def triton_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_features = x.shape
    out_features = weight.shape[1]

    # Use tiling with BLOCK_SIZE_M=128, BLOCK_SIZE_N=128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = lambda meta: (batch_size,)
    gemm_kernel[grid](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr(),
        torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device).data_ptr(),
        batch_size, in_features, out_features,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return torch.empty(batch_size, out_features, dtype=x.dtype, device=x.device)


def triton_subtract(x: torch.Tensor, subtract: torch.Tensor):
    assert x.is_cuda and subtract.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    subtract = subtract.contiguous()

    batch_size, out_features = x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: (batch_size,)
    subtract_kernel[grid](
        x.data_ptr(),
        subtract.data_ptr(),
        torch.empty_like(x).data_ptr(),
        batch_size, out_features, BLOCK_SIZE
    )
    return torch.empty_like(x)


def triton_global_avg_pool(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size, out_features = x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: (batch_size,)
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
    global_avg_pool_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        batch_size, out_features, BLOCK_SIZE
    )
    return out


def triton_logsumexp(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size = x.shape[0]
    BLOCK_SIZE = 128

    grid = lambda meta: (batch_size,)
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
    logsumexp_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        batch_size, BLOCK_SIZE
    )
    return out


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    batch_size = x.shape[0]
    BLOCK_SIZE = 128

    grid = lambda meta: (batch_size,)
    out = torch.empty(batch_size, 1, dtype=x.dtype, device=x.device)
    gelu_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        batch_size, BLOCK_SIZE
    )
    return out


def triton_residual_add(x: torch.Tensor, original_x: torch.Tensor):
    assert x.is_cuda and original_x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    original_x = original_x.contiguous()

    batch_size, in_features = original_x.shape
    BLOCK_SIZE = 128

    grid = lambda meta: (batch_size,)
    out = torch.empty_like(original_x)
    residual_add_kernel[grid](
        x.data_ptr(),
        original_x.data_ptr(),
        out.data_ptr(),
        batch_size, in_features, BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm
        x = triton_gemm(x, self.gemm.weight, self.gemm.bias)
        # Subtract
        x = triton_subtract(x, self.subtract)
        # GlobalAvgPool
        x = triton_global_avg_pool(x)
        # LogSumExp
        x = triton_logsumexp(x)
        # GELU
        x = triton_gelu(x)
        # ResidualAdd
        x = triton_residual_add(x, original_x)
        return x
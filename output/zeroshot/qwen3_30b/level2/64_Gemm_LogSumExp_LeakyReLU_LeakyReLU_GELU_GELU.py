import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BIAS_PTR,
    OUTPUT_SCALE,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        # Load A and B
        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k_offset),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K - k_offset) & (offs_n[None, :] < N),
            other=0.0
        )

        # Compute partial gemm
        accumulator += tl.dot(a, b)

    # Handle bias if present
    if USE_BIAS:
        bias = tl.load(BIAS_PTR + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    # Apply activation (GELU is approximated)
    if ACTIVATION == 1:  # LeakyReLU
        accumulator = tl.where(accumulator > 0, accumulator, 0.01 * accumulator)
    elif ACTIVATION == 2:  # GELU
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (accumulator + 0.044715 * accumulator * accumulator * accumulator)))
        accumulator = accumulator * cdf

    # Scale output
    accumulator *= OUTPUT_SCALE

    # Store output
    tl.store(
        c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        accumulator,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Load input
    x = tl.load(
        x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=-float('inf')
    )

    # Compute max along dim=1 (across columns)
    max_val = tl.max(x, axis=1)

    # Compute log(sum(exp(x - max_val))) + max_val
    x_shifted = x - max_val[:, None]
    exp_sum = tl.sum(tl.exp(x_shifted), axis=1)
    result = tl.log(exp_sum) + max_val

    # Store result
    tl.store(
        out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn),
        result[:, None],
        mask=offs_m[:, None] < M
    )


@triton.jit
def fused_activation_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_outm, stride_outn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)

    # Load input
    x = tl.load(
        x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0
    )

    # Apply activation
    if ACTIVATION == 1:  # LeakyReLU
        out = tl.where(x > 0, x, 0.01 * x)
    elif ACTIVATION == 2:  # GELU
        cdf = 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
        out = x * cdf

    # Store result
    tl.store(
        out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn),
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def triton_gemm_with_activations(x, weight, bias=None, output_scale=1.0):
    # Ensure inputs are contiguous
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = x.shape
    N, _ = weight.shape

    # Compute strides
    stride_xm, stride_xk = x.stride()
    stride_wk, stride_wn = weight.stride()
    stride_b = bias.stride()[0] if bias is not None else 0

    # Output tensor
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Configure block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64

    # Grid definition
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    # Launch kernel with activation
    gemm_kernel[
        grid
    ](
        x, weight, out,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        out.stride()[0], out.stride()[1],
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        1,  # activation type: LeakyReLU
        bias is not None,  # use bias
        bias,  # bias pointer
        output_scale,
    )

    return out


def triton_logsumexp(x):
    x = x.contiguous()
    M, N = x.shape

    out = torch.empty(M, 1, dtype=x.dtype, device=x.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(1, META["BLOCK_SIZE_N"]))

    logsumexp_kernel[
        grid
    ](
        x, out,
        M, N,
        x.stride()[0], x.stride()[1],
        out.stride()[0], out.stride()[1],
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )

    return out


def triton_fused_activation(x, activation_type=1):
    x = x.contiguous()
    M, N = x.shape

    out = torch.empty_like(x)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]), triton.cdiv(N, META["BLOCK_SIZE_N"]))

    fused_activation_kernel[
        grid
    ](
        x, out,
        M, N,
        x.stride()[0], x.stride()[1],
        out.stride()[0], out.stride()[1],
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        activation_type
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.output_scale = 1.0

    def forward(self, x):
        # Gemm + LeakyReLU (first activation)
        x = triton_gemm_with_activations(x, self.linear.weight, self.linear.bias, self.output_scale)
        
        # LogSumExp (along dim=1, keepdim=True)
        x = triton_logsumexp(x)
        
        # Two LeakyReLUs
        x = triton_fused_activation(x, activation_type=1)
        x = triton_fused_activation(x, activation_type=1)
        
        # Two GELUs
        x = triton_fused_activation(x, activation_type=2)
        x = triton_fused_activation(x, activation_type=2)
        
        return x
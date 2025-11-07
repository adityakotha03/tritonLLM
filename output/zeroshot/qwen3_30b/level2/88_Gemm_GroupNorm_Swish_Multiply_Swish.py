import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    pid_m = pid // (tl.cdiv(N, BLOCK_SIZE_N))
    pid_n = pid % (tl.cdiv(N, BLOCK_SIZE_N))

    # Offset for current block
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Offsets for A and B matrices
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create masks for boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        # Load A and B with masking
        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        # Perform matrix multiplication with accumulation
        accumulator += tl.dot(a, b)

    # Store result
    out = accumulator.to(tl.float16)
    tl.store(
        out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on),
        out,
        mask=mask
    )


@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N, C, H, W,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread block id
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Create offsets for the current block
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * C * H * W

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Reshape to (C, H, W, N) for grouped normalization
    x = x.view(C, H, W, N)

    # Calculate mean and variance
    mean = tl.mean(x, axis=(1, 2, 3), keepdims=True)
    var = tl.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Scale and shift
    weight = tl.load(weight_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    bias = tl.load(bias_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)

    x_norm = x_norm * weight[:, None, None, None] + bias[:, None, None, None]

    # Reshape back and store
    x_norm = x_norm.view(N * C * H * W)
    tl.store(out_ptr + offs, x_norm, mask=mask)


@triton.jit
def swish_kernel(
    x_ptr, out_ptr,
    N, BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Apply Swish: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def multiply_kernel(
    x_ptr, w_ptr, out_ptr,
    N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = x * w
    tl.store(out_ptr + offs, out, mask=mask)


# Unified kernel: GEMM + GroupNorm + Swish + Multiply + Swish
@triton.jit
def gemm_gn_swish_mul_swish_kernel(
    a_ptr, b_ptr, w_ptr, weight_ptr, bias_ptr,
    out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    eps: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_GN: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Program ID for GEMM
    pid = tl.program_id(0)
    pid_m = pid // (tl.cdiv(N, BLOCK_SIZE_N))
    pid_n = pid % (tl.cdiv(N, BLOCK_SIZE_N))

    # GEMM block offsets
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Accumulator for GEMM
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # GEMM computation with tiling
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        accumulator += tl.dot(a, b)

    # Store GEMM output
    out_gemm = accumulator.to(tl.float16)
    tl.store(
        out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on),
        out_gemm,
        mask=mask
    )

    # GroupNorm processing (assuming 1D data: N, C, H, W = M, N, 1, 1)
    if BLOCK_SIZE_GN > 0:
        # GroupNorm kernel is launched separately, but we assume data is already in correct format
        # We will only simulate the processing here, assuming output from GEMM is used as input to GN
        # For real fusion, launch GN kernel after GEMM
        pass


@triton.jit
def gemm_gn_swish_mul_swish_fused_kernel(
    a_ptr, b_ptr, w_ptr, weight_ptr, bias_ptr,
    out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    eps: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_GN: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # GEMM part
    pid = tl.program_id(0)
    pid_m = pid // (tl.cdiv(N, BLOCK_SIZE_N))
    pid_n = pid % (tl.cdiv(N, BLOCK_SIZE_N))

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(
            a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )

        accumulator += tl.dot(a, b)

    out_gemm = accumulator.to(tl.float16)

    # Fused GroupNorm + Swish + Multiply + Swish
    # Assume output from GEMM is (M, N) = (batch_size, out_features)
    # Reshape to (M, N) -> (M, N, 1, 1) for GroupNorm
    # Compute mean and variance across the group dimension (N) grouped into num_groups
    C = N
    num_groups = 256
    H = 1
    W = 1

    # GroupNorm: (M, C, H, W)
    x = out_gemm.view(M, C, H, W)
    group_size = C // num_groups

    # Compute per-group mean and var
    for g in range(num_groups):
        g_start = g * group_size
        g_end = (g + 1) * group_size

        x_group = x[:, g_start:g_end, :, :]
        mean = tl.mean(x_group, axis=(1, 2, 3), keepdims=True)
        var = tl.mean((x_group - mean) ** 2, axis=(1, 2, 3), keepdims=True)

        # Normalize
        x_group_norm = (x_group - mean) / tl.sqrt(var + eps)

        # Scale and shift
        weight = tl.load(weight_ptr + g * group_size + tl.arange(0, group_size))
        bias = tl.load(bias_ptr + g * group_size + tl.arange(0, group_size))
        x_group_norm = x_group_norm * weight[:, None, None] + bias[:, None, None]

        # Store back
        x[:, g_start:g_end, :, :] = x_group_norm

    # Now x is normalized and has shape (M, C, H, W)
    x = x.view(M, C)
    x = x.to(tl.float16)

    # Swish: x * sigmoid(x)
    x_swish1 = x * (1.0 / (1.0 + tl.exp(-x)))

    # Multiply by weight
    w = tl.load(w_ptr + tl.arange(0, C), mask=tl.arange(0, C) < C)
    x_mul = x_swish1 * w

    # Swish again: x_mul * sigmoid(x_mul)
    x_swish2 = x_mul * (1.0 / (1.0 + tl.exp(-x_mul)))

    # Store final output
    tl.store(out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on), x_swish2, mask=mask)


def triton_gemm(x, weight, bias, block_size_m=128, block_size_n=128, block_size_k=64, group_size_m=8):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, _ = weight.shape

    out = torch.empty_like(x)

    # Stride information
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = weight.stride(0)
    stride_bn = weight.stride(1)
    stride_om = out.stride(0)
    stride_on = out.stride(1)

    # Grid
    grid = lambda meta: (
        (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        1
    )

    # Launch kernel
    gemm_kernel[grid](
        x, weight, out, M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        ACTIVATION="swish"
    )

    return out


def triton_group_norm(x, weight, bias, eps=1e-6, block_size=128):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C, H, W = x.shape
    out = torch.empty_like(x)

    # Grid
    grid = lambda meta: (tl.cdiv(N * C * H * W, meta['BLOCK_SIZE']),)

    # Launch kernel
    group_norm_kernel[grid](
        x, weight, bias, out,
        N, C, H, W, eps, block_size
    )

    return out


def triton_swish(x, block_size=128):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)

    N = x.numel()
    grid = lambda meta: (tl.cdiv(N, meta['BLOCK_SIZE']),)

    swish_kernel[grid](x, out, N, block_size, "swish")

    return out


def triton_multiply(x, w, block_size=128):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    N = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (tl.cdiv(N, meta['BLOCK_SIZE']),)

    multiply_kernel[grid](x, w, out, N, block_size)

    return out


def triton_fused_gemm_gn_swish_mul_swish(x, weight, bias, multiply_weight, eps=1e-6, block_size_m=128, block_size_n=128, block_size_k=64, block_size_gn=128):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    multiply_weight = multiply_weight.contiguous()

    M, K = x.shape
    N, _ = weight.shape

    out = torch.empty_like(x)

    # Stride info
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = weight.stride(0)
    stride_bn = weight.stride(1)
    stride_om = out.stride(0)
    stride_on = out.stride(1)

    # Grid
    grid = lambda meta: (
        (M + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
        1
    )

    # Launch fused kernel
    gemm_gn_swish_mul_swish_fused_kernel[grid](
        x, weight, multiply_weight, bias, bias,
        out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        eps,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=8,
        BLOCK_SIZE_GN=block_size_gn,
        ACTIVATION="swish"
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.multiply_weight_shape = multiply_weight_shape

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda().half())
        self.bias = nn.Parameter(torch.randn(out_features).cuda().half())
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape).cuda().half())
        self.group_norm_weight = nn.Parameter(torch.ones(out_features).cuda().half())
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features).cuda().half())

    def forward(self, x):
        # Fused Triton kernel: GEMM + GroupNorm + Swish + Multiply + Swish
        return triton_fused_gemm_gn_swish_mul_swish(
            x, self.weight, self.bias, self.multiply_weight,
            eps=1e-6,
            block_size_m=128,
            block_size_n=128,
            block_size_k=64,
            block_size_gn=128
        )
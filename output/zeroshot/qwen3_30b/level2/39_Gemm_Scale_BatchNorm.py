import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // (tl.cdiv(N, BLOCK_SIZE_N))
    pid_n = pid % (tl.cdiv(N, BLOCK_SIZE_N))
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    offs_am = offsets_m[:, None]
    offs_bn = offsets_n[None, :]
    offs_ak = tl.arange(0, BLOCK_SIZE_K)
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am * stride_am + offs_ak * stride_ak)
    b_ptrs = b_ptr + (offs_bk * stride_bk + offs_bn * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "relu":
        accumulator = tl.relu(accumulator)
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (offs_am * stride_cm + offs_bn * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))


@triton.jit
def scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_s,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    offs_xm = offsets_m[:, None]
    offs_xn = offsets_n[None, :]
    offs_s = offsets_n
    x_ptrs = x_ptr + (offs_xm * stride_xm + offs_xn * stride_xn)
    scale_ptrs = scale_ptr + offs_s * stride_s
    x = tl.load(x_ptrs, mask=(offs_xm[:, None] < M) & (offs_xn[None, :] < N), other=0.0)
    scale = tl.load(scale_ptrs, mask=offs_xn < N, other=1.0)
    out = x * scale
    out_ptrs = out_ptr + (offs_xm * stride_xm + offs_xn * stride_xn)
    tl.store(out_ptrs, out, mask=(offs_xm[:, None] < M) & (offs_xn[None, :] < N))


@triton.jit
def bn_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, inv_var_ptr,
    out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_w, stride_b,
    stride_m, stride_iv,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_m = pid * BLOCK_SIZE_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    offs_xm = offsets_m[:, None]
    offs_xn = offsets_n[None, :]
    x_ptrs = x_ptr + (offs_xm * stride_xm + offs_xn * stride_xn)
    weight_ptrs = weight_ptr + offs_xn * stride_w
    bias_ptrs = bias_ptr + offs_xn * stride_b
    mean_ptrs = mean_ptr + offs_xn * stride_m
    inv_var_ptrs = inv_var_ptr + offs_xn * stride_iv
    x = tl.load(x_ptrs, mask=(offs_xm[:, None] < M) & (offs_xn[None, :] < N), other=0.0)
    weight = tl.load(weight_ptrs, mask=offs_xn < N, other=0.0)
    bias = tl.load(bias_ptrs, mask=offs_xn < N, other=0.0)
    mean = tl.load(mean_ptrs, mask=offs_xn < N, other=0.0)
    inv_var = tl.load(inv_var_ptrs, mask=offs_xn < N, other=0.0)
    x = (x - mean) * inv_var
    x = x * weight + bias
    out_ptrs = out_ptr + (offs_xm * stride_xm + offs_xn * stride_xn)
    tl.store(out_ptrs, x, mask=(offs_xm[:, None] < M) & (offs_xn[None, :] < N))


def triton_matmul(x, w, activation="none"):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    M, K = x.shape
    K, N = w.shape
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation,
    )
    return out


def triton_scale(x, scale):
    assert x.is_cuda and scale.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    scale = scale.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    scale_kernel[grid](
        x, scale, out,
        M, N,
        x.stride(0), x.stride(1),
        scale.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


def triton_bn(x, weight, bias, mean, inv_var):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and mean.is_cuda and inv_var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    inv_var = inv_var.contiguous()
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]),)
    bn_kernel[grid](
        x, weight, bias, mean, inv_var, out,
        M, N,
        x.stride(0), x.stride(1),
        weight.stride(0), bias.stride(0),
        mean.stride(0), inv_var.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.bn_mean = nn.Parameter(torch.zeros(out_features))
        self.bn_var = nn.Parameter(torch.ones(out_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = triton_matmul(x, self.gemm_weight, activation="none")
        x = triton_scale(x, self.scale)
        # Update running statistics
        with torch.no_grad():
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            self.bn_mean = (1 - self.momentum) * self.bn_mean + self.momentum * batch_mean
            self.bn_var = (1 - self.momentum) * self.bn_var + self.momentum * batch_var
        # Compute inverse variance
        inv_var = 1.0 / torch.sqrt(self.bn_var + self.eps)
        x = triton_bn(x, self.bn_weight, self.bn_bias, self.bn_mean, inv_var)
        return x
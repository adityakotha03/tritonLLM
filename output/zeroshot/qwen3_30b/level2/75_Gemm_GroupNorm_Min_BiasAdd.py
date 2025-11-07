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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_BIAS: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid // (GROUP_SIZE_M // BLOCK_M)
    pid_n = pid % (GROUP_SIZE_M // BLOCK_M)
    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    # Bias addition
    if IS_BIAS:
        bias_ptr = c_ptr + offs_n
        bias = tl.load(bias_ptr, mask=offs_n < N, other=0.0)
        c += bias[None, :]

    # Store output
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute mean and invstd
    mask = offs < C * H * W
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    invstd = tl.load(invstd_ptr + offs, mask=mask, other=0.0)

    # Normalize
    x = (x - mean) * invstd

    # Store output
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def min_reduce_kernel(
    x_ptr, out_ptr,
    N, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    min_val = tl.min(x)

    # Reduce to single value per block
    out = tl.broadcast(min_val, (1,))

    # Write output (only one thread per block writes)
    if pid == 0:
        tl.store(out_ptr, out)


def triton_gemm(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, out: torch.Tensor):
    M, K = x.shape
    K, N = w.shape
    assert M == out.shape[0] and N == out.shape[1], "Output shape mismatch"

    x = x.contiguous()
    w = w.contiguous()
    bias = bias.contiguous()
    out = out.contiguous()

    # Kernel parameters
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    GROUP_SIZE_M = 64

    # Grid configuration
    grid = lambda meta: (tl.cdiv(M, meta['BLOCK_M']) * tl.cdiv(N, meta['BLOCK_N']) // GROUP_SIZE_M,)

    # Launch kernel
    gemm_kernel[
        grid
    ](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION="none",
        IS_BIAS=(bias is not None)
    )


def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    B, C, H, W = x.shape
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Prepare mean and std
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    invstd = 1.0 / (var + eps).sqrt()

    # Allocate buffers
    out = torch.empty_like(x)
    mean = mean.contiguous()
    invstd = invstd.contiguous()

    # Grid
    BLOCK_SIZE = 256
    grid = lambda meta: (tl.cdiv(C * H * W, meta['BLOCK_SIZE']),)

    # Launch kernel
    group_norm_kernel[grid](x, mean, invstd, out, C * H * W, C, H, W, BLOCK_SIZE=BLOCK_SIZE)

    # Scale and shift
    if weight is not None:
        weight = weight.contiguous()
        out = out * weight
    if bias is not None:
        bias = bias.contiguous()
        out = out + bias

    return out


def triton_min_reduce(x: torch.Tensor):
    B, C, H, W = x.shape
    assert x.is_cuda, "Input must be on CUDA"

    x = x.contiguous()
    out = torch.empty((B, 1, H, W), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 256
    grid = lambda meta: (tl.cdiv(C, meta['BLOCK_SIZE']),)

    min_reduce_kernel[grid](x, out, C, H, W, BLOCK_SIZE=BLOCK_SIZE)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.gemm_weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16, device='cuda'))
        self.group_norm_weight = nn.Parameter(torch.randn(out_features, dtype=torch.float16, device='cuda'))
        self.group_norm_bias = nn.Parameter(torch.randn(out_features, dtype=torch.float16, device='cuda'))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float16, device='cuda'))
        self.num_groups = num_groups

    def forward(self, x):
        # Ensure input is float16
        x = x.to(torch.float16)

        # Step 1: GEMM
        out_gemm = torch.empty(x.shape[0], self.gemm_weight.shape[0], dtype=x.dtype, device=x.device)
        triton_gemm(x, self.gemm_weight, self.bias, out_gemm)

        # Step 2: GroupNorm
        B, C, H, W = out_gemm.shape
        x_gn = out_gemm.view(B, self.num_groups, C // self.num_groups, H, W)
        x_gn = x_gn.permute(0, 2, 1, 3, 4).contiguous().view(B * C // self.num_groups, self.num_groups, H, W)
        x_gn = x_gn.view(-1, self.num_groups, H, W)
        out_gn = triton_group_norm(x_gn, self.group_norm_weight, self.group_norm_bias, eps=1e-6)

        # Reshape back
        out_gn = out_gn.view(B, C // self.num_groups, self.num_groups, H, W)
        out_gn = out_gn.permute(0, 2, 1, 3, 4).contiguous().view(B, C, H, W)

        # Step 3: Min reduction across channels
        out_min = triton_min_reduce(out_gn)

        # Step 4: Add bias
        out = out_min + self.bias

        return out
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bias_hardtanh_mish_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    BATCH: tl.constexpr, IN_FEATURES: tl.constexpr, OUT_FEATURES: tl.constexpr,
    stride_xb, stride_wi, stride_wo, stride_ob,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr, USE_TF32: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xb + offs_k[None, :] * stride_wi)
    w_ptrs = weight_ptr + (offs_k[:, None] * stride_wi + offs_n[None, :] * stride_wo)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(IN_FEATURES, BLOCK_K)):
        x_mask = (offs_m < BATCH)[:, None] & (offs_k < IN_FEATURES)[None, :]
        w_mask = (offs_k < IN_FEATURES)[:, None] & (offs_n < OUT_FEATURES)[None, :]

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        accumulator = tl.dot(x, w, acc=accumulator, out_dtype=tl.float32, allow_tf32=USE_TF32)

        x_ptrs += BLOCK_K * stride_wi
        w_ptrs += BLOCK_K * stride_wi

    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias_mask = offs_n < OUT_FEATURES
        bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        accumulator = accumulator + bias[None, :]

    # Hardtanh: clamp between -1 and 1
    accumulator = tl.clamp(accumulator, -1.0, 1.0)

    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # Softplus: ln(1 + exp(x))
    softplus = tl.where(accumulator > 20, accumulator, tl.log(tl.exp(accumulator) + 1.0))
    tanh_softplus = tl.tanh(softplus)
    accumulator = accumulator * tanh_softplus

    out_ptrs = out_ptr + (offs_m[:, None] * stride_ob + offs_n[None, :] * 1)
    out_mask = (offs_m < BATCH)[:, None] & (offs_n < OUT_FEATURES)[None, :]
    tl.store(out_ptrs, accumulator, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = C // num_groups
    start_c = pid_g * group_size
    end_c = start_c + group_size

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_c = start_c + tl.arange(0, BLOCK_SIZE_C)

    mask_n = offs_n < N
    mask_c = offs_c < end_c

    n_valid = tl.sum(mask_n)
    c_valid = tl.sum(mask_c)

    x_ptrs = x_ptr + (offs_n[:, None] * C + offs_c[None, :])
    mask = mask_n[:, None] & mask_c[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / c_valid
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / c_valid
    inv_std = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)

    out = (x - mean[:, None]) * inv_std[:, None] * gamma[None, :] + beta[None, :]

    out_ptrs = out_ptr + (offs_n[:, None] * C + offs_c[None, :])
    tl.store(out_ptrs, out, mask=mask)


def triton_matmul_bias_hardtanh_mish(x, weight, bias):
    BATCH, IN_FEATURES = x.shape
    OUT_FEATURES, _ = weight.shape

    out = torch.empty((BATCH, OUT_FEATURES), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(BATCH, META['BLOCK_M']),
            triton.cdiv(OUT_FEATURES, META['BLOCK_N']),
        )

    has_bias = bias is not None

    # Use autotuning
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        ],
        key=['IN_FEATURES'],
    )
    @triton.jit
    def kernel_caller(
        x_ptr, weight_ptr, bias_ptr, out_ptr,
        BATCH, IN_FEATURES, OUT_FEATURES,
        stride_xb, stride_wi, stride_wo, stride_ob,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        matmul_bias_hardtanh_mish_kernel(
            x_ptr, weight_ptr, bias_ptr, out_ptr,
            BATCH, IN_FEATURES, OUT_FEATURES,
            stride_xb, stride_wi, stride_wo, stride_ob,
            BLOCK_M, BLOCK_N, BLOCK_K,
            HAS_BIAS=True, USE_TF32=True
        )

    kernel_caller[grid](
        x, weight, bias, out,
        BATCH, IN_FEATURES, OUT_FEATURES,
        x.stride(0), weight.stride(1), weight.stride(0), out.stride(0)
    )

    return out


def triton_group_norm(x, gamma, beta, num_groups, eps=1e-5):
    N, C = x.shape
    H = 1  # treat as N, C, H=1

    out = torch.empty_like(x)

    def grid(META):
        return (triton.cdiv(N, META['BLOCK_SIZE_N']), num_groups)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_C': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_C': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_C': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_C': 32}, num_stages=3, num_warps=4),
        ],
        key=['C'],
    )
    @triton.jit
    def kernel_caller(
        x_ptr, gamma_ptr, beta_ptr, out_ptr,
        N, C, H, num_groups, eps,
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
    ):
        group_norm_kernel(
            x_ptr, gamma_ptr, beta_ptr, out_ptr,
            N, C, H, num_groups, eps,
            BLOCK_SIZE_N, BLOCK_SIZE_C
        )

    kernel_caller[grid](
        x, gamma, beta, out,
        N, C, H, num_groups, eps
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernels for GEMM + Bias + Hardtanh + Mish and Triton-based GroupNorm.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)
        # Register gamma and beta as parameters to access directly
        self.register_parameter('gn_gamma', nn.Parameter(self.groupnorm.weight))
        self.register_parameter('gn_beta', nn.Parameter(self.groupnorm.bias))
        self.eps = self.groupnorm.eps

    def forward(self, x):
        x = triton_matmul_bias_hardtanh_mish(x, self.weight, self.bias)
        x = triton_group_norm(x, self.gn_gamma, self.gn_beta, self.groupnorm.num_groups, self.eps)
        return x
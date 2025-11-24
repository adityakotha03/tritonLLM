import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(in_out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_out_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x

    tl.store(in_out_ptr + offsets, out, mask=mask)


@triton.jit
def multiply_sigmoid_kernel(x_ptr, weight_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    x_weighted = x * w
    sigmoid_xw = tl.sigmoid(x_weighted)
    out = x_weighted * sigmoid_xw

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def matmul_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)

    if ACTIVATION == "swish":
        sigmoid_c = tl.sigmoid(c)
        c = c * sigmoid_c

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def group_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    M, N, G,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_groups = G
    group_id = pid

    if group_id >= num_groups:
        return

    group_size = N // num_groups
    channels_per_block = tl.cdiv(group_size, BLOCK_SIZE_N) * BLOCK_SIZE_N

    for cid in range(0, group_size, channels_per_block):
        group_offs = group_id * group_size + tl.arange(0, BLOCK_SIZE_N)
        ch_offs = group_offs + cid
        mask = ch_offs < (group_id + 1) * group_size

        mean = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        mean_of_squares = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for row in range(0, M):
            row_start = row * N
            x = tl.load(x_ptr + row_start + ch_offs, mask=mask, other=0.0)
            mean += x
            mean_of_squares += x * x

        mean = mean / group_size
        mean_of_squares = mean_of_squares / group_size
        var = mean_of_squares - mean * mean
        inv_std = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + ch_offs, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch_offs, mask=mask, other=0.0)

        for row in range(0, M):
            row_start = row * N
            x = tl.load(x_ptr + row_start + ch_offs, mask=mask, other=0.0)
            x_hat = (x - mean) * inv_std
            y = x_hat * weight + bias
            tl.store(y_ptr + row_start + ch_offs, y, mask=mask)


def triton_matmul_swish(x, weight, bias=None):
    M, K = x.shape
    N, K = weight.shape
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    matmul_gemm_kernel[grid](
        x, weight, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        c.stride(0), c.stride(1),
        GROUP_SIZE_M=8,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        ACTIVATION="swish"
    )
    if bias is not None:
        c += bias
    return c


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)

    def grid(META):
        return (num_groups,)

    group_norm_kernel[grid](
        x, weight, bias, y,
        M, N, num_groups,
        eps,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32
    )
    return y


def triton_swish(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    swish_kernel[grid](out, n_elements, BLOCK_SIZE=1024)
    # Now do x * swish(x) -> but note: swish(x) = x * sigmoid(x), so we want x * (x * sigmoid(x)) = x^2 * sigmoid(x)
    # However, in the forward we have: x = x * torch.sigmoid(x)  --> that's one Swish
    # Then later: x = x * torch.sigmoid(x) again --> so it's Swish applied twice.
    # But our kernel computes swish(x) = x * sigmoid(x). So we need to apply it twice.
    # So first: out = swish(x)
    # Then: apply swish again on out
    swish_kernel[grid](out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_multiply_sigmoid(x, weight):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    multiply_sigmoid_kernel[grid](x, weight, out, n_elements, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused GEMM+Swish, GroupNorm, and Multiply+Swish.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.group_norm_weight = nn.Parameter(torch.ones(out_features))
        self.group_norm_bias = nn.Parameter(torch.zeros(out_features))
        self.num_groups = num_groups
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Fused GEMM + Swish
        x = triton_matmul_swish(x, self.weight, self.bias)
        # GroupNorm
        x = triton_group_norm(x, self.num_groups, self.group_norm_weight, self.group_norm_bias, eps=1e-5)
        # Multiply by learned weight and apply Swish (fused)
        x = triton_multiply_sigmoid(x, self.multiply_weight)
        return x
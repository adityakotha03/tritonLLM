import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# 1.  Matmul + Bias kernel (fused linear)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_a = tl.arange(0, BLOCK_M)[:, None] + block_m
        offs_b = tl.arange(0, BLOCK_N)[None, :] + block_n
        offs_k = tl.arange(0, BLOCK_K)

        a = tl.load(a_ptr + offs_a[:, None] * stride_am + offs_k[None, :] * stride_ak,
                   mask=(offs_a[:, None] < M) & (offs_k[None, :] < K), 
                   other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_b[None, :] * stride_bn,
                   mask=(offs_k[:, None] < K) & (offs_b[None, :] < N),
                   other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + block_n + tl.arange(0, BLOCK_N),
                       mask=(block_n + tl.arange(0, BLOCK_N) < N),
                       other=0.0).to(tl.float16)
        c += bias

    mask_m = block_m + tl.arange(0, BLOCK_M)[:, None] < M
    mask_n = block_n + tl.arange(0, BLOCK_N)[None, :] < N
    mask = mask_m & mask_n
    tl.store(c_ptr + block_m[:, None] * stride_cm + block_n[None, :] * stride_cn,
             c, mask=mask)


def linear_torch(a: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    """
    a: (batch, in_features)
    weight: (out_features, in_features)
    bias: (out_features,)
    returns: (batch, out_features)
    """
    assert a.is_cuda and weight.is_cuda
    a = a.contiguous().to(torch.float16)
    weight = weight.contiguous().to(torch.float16)
    bias = bias.contiguous().to(torch.float16) if bias is not None else None

    M, K = a.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.float16, device=a.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    linear_kernel[grid](
        a_ptr=a.data_ptr(),
        b_ptr=weight.data_ptr(),
        c_ptr=out.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
    )
    return out.to(torch.float16)


# ----------------------------------------------------------------------
# 2.  GroupNorm + LeakyReLU + Sum kernel (fused)
# ----------------------------------------------------------------------
@triton.jit
def gn_leaky_sum_kernel(
    x_ptr,
    out_ptr,
    bias_ptr,     # optional bias (used by groupnorm if needed)
    shape,       # batch_size
    C,           # num_channels
    G,           # num_groups
    eps: tl.constexpr,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid

    group_size = C // G
    # compute mean and var per group for this batch
    mean = tl.zeros([G, group_size], dtype=tl.float32)
    var  = tl.zeros([G, group_size], dtype=tl.float32)

    for g in range(G):
        offs_c = tl.arange(0, group_size) + g * group_size
        # load values
        vals = tl.load(x_ptr + batch * C + offs_c,
                       mask=offs_c < C, other=0.0).to(tl.float32)
        mean_g = tl.sum(vals) / group_size
        var_g  = tl.sum((vals - mean_g) ** 2) / group_size
        mean[g, :] = mean_g
        var[g, :]  = var_g

    # normalize, apply leaky ReLU and add element-wise
    for g in range(G):
        offs_c = tl.arange(0, group_size) + g * group_size
        vals = tl.load(x_ptr + batch * C + offs_c,
                       mask=offs_c < C, other=0.0).to(tl.float32)
        mean_g = mean[g, :]
        var_g  = var[g, :]
        inv_std = tl.math.rsqrt(var_g + eps)
        norm = (vals - mean_g) * inv_std
        # leaky ReLU
        relu = tl.where(norm > 0, norm, norm * negative_slope)
        # elementwise sum (x + x)
        out_val = relu + relu
        tl.store(out_ptr + batch * C + offs_c, out_val, mask=offs_c < C)


def gn_leaky_sum_torch(x: torch.Tensor, num_groups: int, eps=1e-5, negative_slope=0.01):
    """
    x: (batch, channels)
    """
    assert x.is_cuda
    batch, C = x.shape
    G = num_groups
    out = torch.empty_like(x, dtype=torch.float32)

    BLOCK_SIZE = 128
    grid = lambda meta: (batch,)

    gn_leaky_sum_kernel[grid](
        x_ptr=x.data_ptr(),
        out_ptr=out.data_ptr(),
        bias_ptr=None,
        shape=batch,
        C=C,
        G=G,
        eps=eps,
        negative_slope=negative_slope,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.to(torch.float16)


# ----------------------------------------------------------------------
# 3.  Final model that stitches everything together
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size, device='cuda', dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(hidden_size, device='cuda', dtype=torch.float16))
        self.num_groups = num_groups
        self.eps = eps
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor):
        # linear
        y = linear_torch(x, self.weight, self.bias)          # (B, H)
        # groupnorm + leaky relu + sum
        y = gn_leaky_sum_torch(y, self.num_groups, self.eps, self.negative_slope)
        return y


# ----------------------------------------------------------------------
# 4.  Helper functions for compatibility
# ----------------------------------------------------------------------
def get_inputs():
    return [torch.rand(1024, 8192, device='cuda', dtype=torch.float16)]

def get_init_inputs():
    return [8192, 8192, 512]
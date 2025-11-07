import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K blocks
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k * BLOCK_SIZE_K
        k_mask = offs_k < (K - k_offset)

        # Load A and B tiles
        a = tl.load(
            A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k_offset)),
            other=0.0
        )
        b = tl.load(
            B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < (K - k_offset)) & (offs_n[None, :] < N),
            other=0.0
        )

        # Perform matrix multiplication
        acc += tl.dot(a, b, allow_tf32=True)

    # Scale and store output
    acc = acc.to(tl.float16)  # Convert to bfloat16 if needed; here we keep float32 for precision, but output in float16
    # Output C with offset and broadcasting
    offs_m = offs_m[:, None]
    offs_n = offs_n[None, :]
    mask = (offs_m < M) & (offs_n < N)
    tl.store(C + (offs_m * stride_cm + offs_n * stride_cn), acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def triton_subtract_kernel(
    x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x - y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def triton_logsumexp_kernel(
    x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_max = tl.broadcast_to(x_max, (BLOCK_SIZE,))
    x_shifted = x - x_max
    x_exp = tl.exp(x_shifted)
    x_exp_sum = tl.sum(x_exp, axis=0)
    logsumexp = x_max + tl.log(x_exp_sum)
    tl.store(out_ptr + tl.arange(0, 1), logsumexp, mask=tl.arange(0, 1) < 1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def triton_gelu_kernel(
    x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_val = tl.tanh(inner)
    gelu = 0.5 * x * (1 + tanh_val)
    tl.store(out_ptr + offsets, gelu, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def triton_residual_add_kernel(
    x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_gemm(x, w, bias=None):
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = x.shape
    K2, N = w.shape

    assert K == K2, "Matrix dimensions mismatch"
    C = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Determine grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))

    # Launch kernel
    triton_gemm_kernel[grid](
        x, w, C,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=64,
    )

    if bias is not None:
        C += bias

    return C


def triton_subtract(x, y):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    triton_subtract_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_logsumexp(x, dim=1, keepdim=True):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    assert x.dim() == 2, "LogSumExp only supports 2D input"
    B, N = x.shape

    # GlobalAvgPool + LogSumExp fused
    # Output is (B, 1)
    out = torch.empty(B, 1, dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    triton_logsumexp_kernel[grid](x, out, N, BLOCK_SIZE=128)

    if keepdim:
        return out
    else:
        return out.squeeze(1)


def triton_gelu(x):
    assert x.is_cuda, "Tensor must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    triton_gelu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_residual_add(x, y):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    triton_residual_add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).cuda())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features).cuda())
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm + Bias (if any)
        x = triton_gemm(x, self.weight, self.bias)

        # Subtract
        x = triton_subtract(x, self.subtract)

        # GlobalAvgPool
        x = x.mean(dim=1, keepdim=True)

        # LogSumExp
        x = triton_logsumexp(x, dim=1, keepdim=True)

        # GELU
        x = triton_gelu(x)

        # ResidualAdd
        x = triton_residual_add(x, original_x)

        return x
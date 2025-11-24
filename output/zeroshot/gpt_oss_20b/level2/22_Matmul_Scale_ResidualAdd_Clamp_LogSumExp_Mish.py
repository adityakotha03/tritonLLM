import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -------------------------------------------------------------
# 1) Custom matmul + scaling (scale_factor * 2)
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 512, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_scaled_kernel(
    a_ptr, b_ptr, c_ptr,  # pointers to input, weight, output
    M, N, K,              # dimensions: M=batch, K=input_size, N=hidden_size
    scale,                # scaling factor
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # program_id for grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets in grid
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # accumulation register
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for offs_k in range(0, K, BLOCK_SIZE_K):
        # Load tiles of A and B
        a = tl.load(
            a_ptr + offs_m[:, None] * K + offs_k[None, :] + tl.arange(0, BLOCK_SIZE_K),
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * N + offs_n[None, :] + tl.arange(0, BLOCK_SIZE_K),
            mask=offs_k[:, None] < K,
            other=0.0,
        )
        # Compute block multiplication
        acc += tl.dot(a, b)

    # Scale and store
    acc = acc * scale
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)


def matmul_scaled(a: torch.Tensor, weight: torch.Tensor, scale: float):
    """
    a: (B, K)
    weight: (K, N)
    """
    assert a.is_cuda and weight.is_cuda
    a = a.contiguous()
    weight = weight.contiguous()
    B, K = a.shape
    K2, N = weight.shape
    assert K == K2

    out = torch.empty(B, N, device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        (B + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
    )
    matmul_scaled_kernel[grid](
        a, weight, out,
        B, N, K,
        scale,
        BLOCK_SIZE_M=meta['BLOCK_SIZE_M'],
        BLOCK_SIZE_K=meta['BLOCK_SIZE_K'],
        BLOCK_SIZE_N=meta['BLOCK_SIZE_N'],
    )
    return out


# -------------------------------------------------------------
# 2) Custom LogSumExp + Mish kernel (fused)
# -------------------------------------------------------------
@triton.jit
def logsumexp_mish_kernel(
    inp_ptr,        # pointer to input (B, N)
    out_ptr,       # pointer to output (B, 1)
    B, N,          # batch, hidden size
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offs = offset + tl.arange(0, BLOCK_SIZE)

    mask = offs < N
    # load input row
    x = tl.load(inp_ptr + pid * N + offs, mask=mask, other=0.0)
    # clamp
    x = tl.max(tl.min(x, clamp_max), clamp_min)

    # compute max for numerical stability
    max_val = tl.max(x, axis=0)
    max_val = tl.broadcast_to(max_val, [BLOCK_SIZE])

    # compute sum of exp(x - max)
    exp_diff = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_diff, axis=0)
    log_sum_exp = max_val[0] + tl.log(sum_exp)

    # Mish: x * tanh(softplus(x))
    # We compute mish on the scalar value log_sum_exp
    y = log_sum_exp
    softplus = tl.log1p(tl.exp(y))
    mish = y * tl.tanh(softplus)

    # store
    tl.store(out_ptr + pid, mish)


def logsumexp_mish(inp: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    inp: (B, N)
    """
    B, N = inp.shape
    out = torch.empty(B, 1, device=inp.device, dtype=inp.dtype)
    BLOCK_SIZE = 256  # tunable
    grid = lambda meta: ( (B + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], )
    logsumexp_mish_kernel[grid](
        inp, out, B, N,
        clamp_min, clamp_max,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# -------------------------------------------------------------
# 3) ModelNew with custom kernels
# -------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for:
      - MatMul + scaling + residual (scaled by 2)
      - Clamping
      - LogSumExp + Mish (fused)
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        # linear layer without bias
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size, device='cuda'))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # x: (B, input_size)
        # matmul + scaling (scale_factor * 2)
        out = matmul_scaled(x, self.weight.t(), self.scale_factor * 2.0)

        # fused LogSumExp + Mish
        out = logsumexp_mish(out, self.clamp_min, self.clamp_max)
        return out
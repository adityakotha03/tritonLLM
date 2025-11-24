import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    coeff = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x3)
    tanh_inner = tl.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)

    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

def triton_gelu(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

@triton.jit
def matmul_no_tiling(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
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
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask)

def triton_matmul(a, b):
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA."
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_no_tiling[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    return c

@triton.jit
def fused_matmul_gelu_kernel(
    a_ptr, w1_ptr, w2_ptr, b1_ptr, b2_ptr,
    out_ptr, residual_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1n, stride_w1k,
    stride_w2k, stride_w2n,
    stride_b1, stride_b2,
    stride_out_m, stride_out_n,
    stride_res_m, stride_res_n,
    use_bias: tl.constexpr,
    use_residual: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w2_ptrs = w2_ptr + (offs_bn[:, None] * stride_w2n + offs_k[None, :] * stride_w2k)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, w1)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k

    if use_bias:
        b1 = tl.load(b1_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + b1[None, :]

    # GELU activation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x3 = accumulator * accumulator * accumulator
    inner = sqrt_2_over_pi * (accumulator + coeff * x3)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * accumulator * (1.0 + tanh_inner)

    # Second linear projection
    accumulator_2 = tl.zeros((BLOCK_SIZE_M, N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        w2 = tl.load(w2_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K, other=0.0)
        acc_slice = gelu_out[:, k * BLOCK_SIZE_K : min((k+1)*BLOCK_SIZE_K, N)]
        accumulator_2 += tl.dot(acc_slice, w2)
        w2_ptrs += BLOCK_SIZE_K * stride_w2k

    if use_bias:
        b2 = tl.load(b2_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator_2 = accumulator_2 + b2[None, :]

    if use_residual:
        residual = tl.load(residual_ptr + offs_am[:, None] * stride_res_m + offs_bn[None, :] * stride_res_n,
                           mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N), other=0.0)
        accumulator_2 += residual

    out_ptrs = out_ptr + (offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_n)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(out_ptrs, accumulator_2.to(tl.float16), mask=mask)

def fused_mlp_forward(x, c_fc_weight, c_proj_weight, c_fc_bias, c_proj_bias, residual=None):
    assert x.is_cuda and c_fc_weight.is_cuda and c_proj_weight.is_cuda
    x = x.contiguous()
    c_fc_weight = c_fc_weight.t().contiguous()
    c_proj_weight = c_proj_weight.t().contiguous()
    M, K = x.shape
    N = c_proj_weight.shape[1]
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    use_bias = c_fc_bias is not None and c_proj_bias is not None
    use_residual = residual is not None

    if use_residual:
        residual = residual.contiguous()

    fused_matmul_gelu_kernel[grid](
        x, c_fc_weight, c_proj_weight,
        c_fc_bias, c_proj_bias,
        out, residual,
        M, N, K,
        x.stride(0), x.stride(1),
        c_fc_weight.stride(0), c_fc_weight.stride(1),
        c_proj_weight.stride(0), c_proj_weight.stride(1),
        c_fc_bias.stride(0) if use_bias else 0,
        c_proj_bias.stride(0) if use_bias else 0,
        out.stride(0), out.stride(1),
        residual.stride(0) if use_residual else 0, residual.stride(1) if use_residual else 0,
        use_bias, use_residual,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
    )
    return out

class FusedCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.scale = 1.0 / math.sqrt(n_embd // n_head)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        qkv = qkv.view(B, T, 3, self.n_head, C // self.n_head).transpose(1, 3)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = FusedCausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            fc_bias = nn.Parameter(torch.zeros(4 * n_embd)),
            proj_bias = nn.Parameter(torch.zeros(n_embd)),
            dropout = nn.Dropout(resid_pdrop),
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        residual = x
        x = self.ln_2(x)
        x = fused_mlp_forward(
            x.view(-1, x.size(-1)),
            self.mlp.c_fc.weight, self.mlp.c_proj.weight,
            self.mlp.fc_bias, self.mlp.proj_bias,
            residual.view(-1, x.size(-1))
        )
        x = x.view(residual.shape)
        return x
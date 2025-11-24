import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    tanh_inner = tl.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_inner)

    # Store result
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

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return triton_gelu(x)

@triton.jit
def fused_matmul_add_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, M, K,
    stride_xn, stride_xk,
    stride_wn, stride_wk,
    stride_bn,
    stride_outn, stride_outk,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xn + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    b_ptrs = b_ptr + offs_n * stride_bn
    out_ptrs = out_ptr + (offs_m[:, None] * stride_outn + offs_n[None, :] * stride_outk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
        w_mask = (offs_k < K)[:, None] & (offs_n < N)[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    acc = accumulator.to(tl.float32)
    b = tl.load(b_ptrs, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    
    # GELU activation
    sqrt_2_over_pi = 0.7978845608028654
    coeff = 0.044715
    acc_cubed = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + coeff * acc_cubed)
    tanh_inner = tl.tanh(inner)
    result = 0.5 * acc * (1.0 + tanh_inner)

    out_mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(out_ptrs, result, mask=out_mask)

def fused_matmul_add_gelu(a, w, b):
    assert a.is_cuda and w.is_cuda and b.is_cuda
    M, K = a.shape
    K, N = w.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    fused_matmul_add_gelu_kernel[grid](
        a, w, b, c,
        N, M, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        b.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return c

@triton.jit
def fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, bias_ptr,
    B, T, C, H, HS,
    s_qb, s_qh, s_qt, s_qs,
    s_kb, s_kh, s_kt, s_ks,
    s_vb, s_vh, s_vt, s_vs,
    s_ob, s_oh, s_ot, s_os,
    s_bb, s_bh, s_bt, s_btt,
    BIAS: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    offs_t = tl.arange(0, BLOCK_SIZE_T)
    offs_s = tl.arange(0, HS)

    q_ptrs = q_ptr + b_id * s_qb + h_id * s_qh + offs_t[:, None] * s_qt + offs_s[None, :] * s_qs
    k_ptrs = k_ptr + b_id * s_kb + h_id * s_kh + offs_t[:, None] * s_kt + offs_s[None, :] * s_ks
    v_ptrs = v_ptr + b_id * s_vb + h_id * s_vh + offs_t[:, None] * s_vt + offs_s[None, :] * s_vs
    out_ptrs = out_ptr + b_id * s_ob + h_id * s_oh + offs_t[:, None] * s_ot + offs_s[None, :] * s_os
    bias_ptrs = bias_ptr + b_id * s_bb + h_id * s_bh + offs_t[:, None] * s_bt + offs_t[None, :] * s_btt

    q = tl.load(q_ptrs, mask=(offs_t[:, None] < T) & (offs_s[None, :] < HS), other=0.0)
    q = q.to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_T, HS), dtype=tl.float32)
    for start_t in range(0, T, BLOCK_SIZE_T):
        offs_kt = start_t + offs_t
        k_mask = (offs_kt[:, None] < T) & (offs_s[None, :] < HS)
        k = tl.load(k_ptrs + (offs_kt[:, None] * s_kt), mask=k_mask, other=0.0)
        k = tl.trans(k.to(tl.float32))

        qk = tl.dot(q, k)
        qk = qk * (1.0 / tl.sqrt(HS.to(tl.float32)))

        if BIAS:
            bias_mask = (offs_t[:, None] < T) & (offs_kt[None, :] < T)
            bias = tl.load(bias_ptrs + (offs_t[:, None] * s_bt + offs_kt[None, :] * s_btt), mask=bias_mask, other=0.0)
            qk = tl.where(bias != 0, qk, float('-inf'))

        att = tl.where(qk >= 0, qk, 0.0)  # ReLU

        v_mask = (offs_kt[:, None] < T) & (offs_s[None, :] < HS)
        v = tl.load(v_ptrs + (offs_kt[:, None] * s_vt), mask=v_mask, other=0.0)
        v = v.to(tl.float32)

        acc += tl.dot(att.to(v.dtype), v)

    out_mask = (offs_t[:, None] < T) & (offs_s[None, :] < HS)
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)

def fused_attention(q, k, v, bias, B, T, C, H, HS):
    assert q.is_cuda and k.is_cuda and v.is_cuda and bias.is_cuda
    out = torch.empty((B, H, T, HS), device=q.device, dtype=q.dtype)
    grid = (B, H)
    fused_attention_kernel[grid](
        q, k, v, out, bias,
        B, T, C, H, HS,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3),
        BIAS=True,
        BLOCK_SIZE_T=64
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))

    def forward(self, x):
        B, T, C = x.size()

        # Fused matmul + add + GELU for c_attn
        qkv = fused_matmul_add_gelu(x.view(-1, C), self.c_attn.weight, self.c_attn.bias)
        qkv = qkv.view(B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Fused attention with causal mask and ReLU
        y = fused_attention(q, k, v, self.bias[:, :, :T, :T], B, T, C, self.n_head, C // self.n_head)

        # Final projection (we leave this as standard Linear for simplicity, could be fused too)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y
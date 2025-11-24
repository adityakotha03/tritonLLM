import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B, H, T, D,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                y.stride(0), y.stride(1), y.stride(2),
                BID: tl.constexpr, HID: tl.constexpr,
                T: tl.constexpr, D: tl.constexpr,
                BT: tl.constexpr, BD: tl.constexpr,
                USE_MASK: tl.constexpr,
                bias_ptr, bias_stride_h, bias_stride_t,
                IS_CAUSAL: tl.constexpr):
    start_t = tl.program_id(2)
    off_bt = start_t * BT + tl.arange(0, BT)
    off_d = tl.arange(0, BD)
    q_offset = BID * q_s_b + HID * q_s_h
    k_offset = BID * k_s_b + HID * k_s_h
    v_offset = BID * v_s_b + HID * v_s_h
    q_ptrs = Q + q_offset + (off_bt[:, None] * q_s_t + off_d[None, :])
    k_ptrs = K + k_offset + (off_d[:, None] * k_s_t + off_bt[None, :])
    v_ptrs = V + v_offset + (off_bt[:, None] * v_s_t + off_d[None, :])
    y_offset = BID * y_s_b + HID * y_s_h
    y_ptrs = Y + y_offset + (off_bt[:, None] * y_s_t + off_d[None, :])

    m_i = tl.zeros([BT], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BT], dtype=tl.float32)
    acc = tl.zeros([BT, BD], dtype=tl.float32)

    for start_d in range(0, D, BD):
        q = tl.load(q_ptrs)
        q = (q * sm_scale).to(Q.dtype.element_ty)
        k = tl.load(k_ptrs)
        qk = tl.zeros([BT, BD], dtype=tl.float32)
        qk += tl.dot(q, k)
        if USE_MASK and IS_CAUSAL:
            off_d = start_d + tl.arange(0, BD)
            causal_mask = (off_bt[:, None] >= (start_d + off_d[None, :]))
            qk = tl.where(causal_mask, qk, float("-inf"))
        if USE_MASK and bias_ptr is not None:
            bias = tl.load(bias_ptr + bias_stride_h * HID + bias_stride_t * off_bt, mask=off_bt < T, other=0.0)
            qk += bias[:, None]
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i *= alpha
        l_i += beta * l_ij
        acc *= alpha
        p = p.to(V.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        m_i = m_i_new
        k_ptrs += BD * k_s_t
        v_ptrs += BD
    acc = acc / l_i[:, None]
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty))

def _flash_attn_forward(q, k, v, bias=None, dropout_p=0.0, softmax_scale=None, causal=False):
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(q.shape[-1])
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    B, H, T, D = q.shape
    assert D <= 128
    BD = 64 if D <= 64 else 128
    BT = 64 if T <= 64 else 128
    BC = min(BT, 64)
    num_stages = 3
    num_warps = 4 if D <= 64 else 8
    grid = (triton.cdiv(T, BT), 1, B * H)
    Y = torch.empty_like(q)
    _fwd_kernel[grid](
        Q=q, K=k, V=v, sm_scale=softmax_scale,
        B=B, H=H, T=T, D=D,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BID=0, HID=0,
        T=T, D=D, BT=BT, BD=BD,
        USE_MASK=(bias is not None), bias_ptr=bias,
        bias_stride_h=bias.stride(1) if bias is not None else 0,
        bias_stride_t=bias.stride(2) if bias is not None else 0,
        IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages
    )
    return Y

class ModelNew(nn.Module):
    """
    Optimized version of multi-head self-attention using Triton-based Flash Attention.
    """

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
        self.max_seqlen = max_seqlen

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = _flash_attn_forward(q, k, v, bias=self.bias[:, :, :T, :T], causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
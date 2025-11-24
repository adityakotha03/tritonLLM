import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    L, M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    z_heads, seqlen_q, seqlen_k,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    offs_k = off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_km + offs_d[:, None] * stride_kk
    offs_v = off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vk

    q_ptrs = Q + offs_q
    k_ptrs = K + offs_k
    v_ptrs = V + offs_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q: (BLOCK_M, BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    q = (q * softmax_scale).to(q.dtype)

    loop_steps = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, loop_steps):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_km, mask=(start_n + offs_n)[None, :] < seqlen_k, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        if IS_CAUSAL:
            offs_n_causal = start_n * BLOCK_N + offs_n[None, :]
            causal_mask = offs_m[:, None] >= offs_n_causal
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i = alpha * l_i + beta * l_ij

        p = p.to(q.dtype)
        v = tl.load(v_ptrs + start_n * stride_vm, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v, out_dtype=tl.float32)

        m_i = m_i_new

    acc = acc / l_i[:, None]
    offs_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    out_ptrs = Out + offs_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seqlen_q)

    if L is not None:
        l_ptrs = L + off_z * z_heads + off_h * 1 + offs_m
        tl.store(l_ptrs, l_i, mask=offs_m < seqlen_q)
    if M is not None:
        m_ptrs = M + off_z * z_heads + off_h * 1 + offs_m
        tl.store(m_ptrs, m_i, mask=offs_m < seqlen_q)


def _fwd(Q, K, V, Out=None, L=None, M=None, softmax_scale=None, causal=False):
    if softmax_scale is None:
        softmax_scale = 1.0 / (Q.shape[-1] ** 0.5)

    if Out is None:
        Out = torch.empty_like(Q)
    if L is None:
        L = torch.empty((Q.shape[0] * Q.shape[1], Q.shape[2]), device=Q.device, dtype=torch.float32)
    if M is None:
        M = torch.empty((Q.shape[0] * Q.shape[1], Q.shape[2]), device=Q.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(Q.shape[2], META['BLOCK_M']),
        Q.shape[1],
        Q.shape[0],
    )

    _fwd_kernel[grid](
        Q, K, V, Out,
        L, M,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        Q.shape[0] * Q.shape[1], Q.shape[2], K.shape[2],
        softmax_scale,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_DMODEL=Q.shape[-1],
        IS_CAUSAL=causal,
        BIAS=False,
        num_warps=8,
        num_stages=4,
    )
    return Out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return _fwd(Q, K, V, causal=False)
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    M, N,
    stride_xm, stride_ym,
    stride_weight, stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] 
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    mean = tl.sum(x, axis=1) / N
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / N
    inv_var = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs_n * stride_weight, mask=offs_n < N, other=1.0)
    bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)

    output = (diff * inv_var[:, None]) * weight[None, :] + bias[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, output, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, N = x.shape
    out = torch.empty_like(x)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    BLOCK_M = 32
    BLOCK_N = 128

    _layer_norm_kernel[grid](
        x, weight, bias, out,
        M, N,
        x.stride(0), out.stride(0),
        weight.stride(0), bias.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        eps=eps,
    )
    return out


@triton.jit
def _attn_kernel(
    q_ptr, k_ptr, v_ptr, 
    output_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    q_tile_id = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)

    offs_m = q_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + z * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = k_ptr + z * stride_kz + h * stride_kh + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    v_ptrs = v_ptr + z * stride_vz + h * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < D), other=0.0)
    q = (q * scale).to(tl.float16)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, (N + BLOCK_N - 1) // BLOCK_N * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        current_n = start_n + offs_n

        k = tl.load(k_ptrs + start_n * stride_kn, 
                    mask=(current_n[:, None] < N) & (offs_k[None, :] < D), other=0.0)
        k = k.to(tl.float16)

        qk = tl.dot(q, k, trans_b=True)

        if IS_CAUSAL:
            q_bounds = offs_m < M
            kv_bounds = current_n < N
            causal_mask = (offs_m[:, None] >= (current_n + 1)[None, :])
            qk = tl.where(q_bounds[:, None] & kv_bounds[None, :] & causal_mask, qk, float("-inf"))

        p = tl.softmax(qk, axis=1)
        p = p.to(tl.float16)

        v = tl.load(v_ptrs + start_n * stride_vn,
                    mask=(current_n[:, None] < N) & (offs_k[None, :] < D), other=0.0)
        v = v.to(tl.float16)

        acc += tl.dot(p, v)

    acc = acc.to(tl.float32)

    offs_m_out = q_tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_out = tl.arange(0, BLOCK_D)

    out_ptrs = output_ptr + z * stride_oz + h * stride_oh + offs_m_out[:, None] * stride_om + offs_d_out[None, :] * stride_ok
    mask = (offs_m_out[:, None] < M) & (offs_d_out[None, :] < D)
    tl.store(out_ptrs, acc, mask=mask)


def triton_scaled_dot_product_attention(q, k, v, is_causal=False):
    assert all(t.is_cuda for t in [q, k, v])
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    Z, H, M, D = q.shape
    N = k.shape[2]

    out = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), H, Z)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    _attn_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, M, N, D,
        scale=1.0 / D ** 0.5,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ModelNew, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm = nn.LayerNorm(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        total_elements = B * N

        x = x.view(B, C, N).permute(2, 0, 1).contiguous()  # (N, B, C)

        # Project inputs to q, k, v
        x_2d = x.view(total_elements, C)
        qkv = F.linear(x_2d, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(N, B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(0, 1).contiguous()  # (B, H, N, D)
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        # Custom Triton attention
        attn_output = triton_scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_output = attn_output.transpose(0, 1).contiguous()  # (N, B, C)
        attn_output = attn_output.view(N, B, C)

        # Linear projection
        attn_output = attn_output.view(N * B, C)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(N, B, C)

        # Residual and layer norm using Triton kernel
        residual = x
        norm_input = (attn_output + residual).view(N * B, C)
        norm_output = triton_layer_norm(
            norm_input,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps
        )
        norm_output = norm_output.view(N, B, C).permute(1, 2, 0).view(B, C, H, W)

        return norm_output
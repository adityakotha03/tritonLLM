import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------
# Triton kernel: batched matrix multiplication (B, M, K) @ (B, K, N)
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 256},
                      num_warps=4),
    ],
    key=['M', 'N', 'K', 'B'],
)
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    stride_a_b, stride_a_c, stride_b_c,
    M, N, K, B,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Loop over batches
    for b in range(B):
        a_ptr = A_ptr + b * stride_a_b
        b_ptr = B_ptr + b * stride_b_c
        c_ptr = C_ptr + b * stride_a_c

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        # Allocate accumulator
        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)

            # Load tiles
            a = tl.load(a_ptr + offs_m[:, None] * stride_a_b + offs_k[None, :] * 1,
                        mask=offs_m[:, None] < M,
                        other=0.0)
            b_ = tl.load(b_ptr + offs_k[:, None] * stride_b_c + offs_n[None, :] * 1,
                         mask=offs_n[None, :] < N,
                         other=0.0)

            acc += a.to(tl.float32) @ b_.to(tl.float32)

        # Store result
        tl.store(c_ptr + offs_m[:, None] * stride_a_c + offs_n[None, :] * 1,
                 acc.to(tl.float32),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    A: (B, M, K)  B: (B, K, N) -> (B, M, N)
    """
    assert A.is_cuda and B.is_cuda
    B_batch, M, K = A.shape
    _, _, N = B.shape
    C = torch.empty((B_batch, M, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),
                         triton.cdiv(N, meta['BLOCK_SIZE_N']))
    _matmul_kernel[grid](
        A,
        B,
        C,
        A.stride(1), C.stride(1), B.stride(1),
        M, N, K, B_batch,
    )
    return C


# --------------------------------------------------------------
# Triton kernel: fused query/key/value projection
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['seq_len', 'embed_dim', 'heads'],
)
@triton.jit
def _qkv_kernel(
    inp_ptr,   # (seq_len, batch, embed_dim)
    wq_ptr,    # (embed_dim, embed_dim)
    wk_ptr,    # (embed_dim, embed_dim)
    wv_ptr,    # (embed_dim, embed_dim)
    out_q_ptr, # (seq_len, batch, heads, head_dim)
    out_k_ptr,
    out_v_ptr,
    seq_len,
    batch,
    embed_dim,
    heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    stride_inp = inp_ptr.stride(1)  # batch stride
    stride_out = out_q_ptr.stride(1)

    # Load input token
    inp = tl.load(inp_ptr + idx, mask=idx < seq_len, other=0.0)

    # For each head
    for h in range(heads):
        # Compute Q, K, V
        q = tl.dot(inp, wq_ptr + h * head_dim)
        k = tl.dot(inp, wk_ptr + h * head_dim)
        v = tl.dot(inp, wv_ptr + h * head_dim)

        # Store
        offset = idx * stride_out + h * head_dim
        tl.store(out_q_ptr + offset, q, mask=idx < seq_len)
        tl.store(out_k_ptr + offset, k, mask=idx < seq_len)
        tl.store(out_v_ptr + offset, v, mask=idx < seq_len)


# --------------------------------------------------------------
# Triton kernel: scaled dot-product attention + output projection
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_QK': 128, 'BLOCK_SIZE_V': 128}, num_warps=4),
    ],
    key=['seq_len', 'heads', 'head_dim'],
)
@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr,
    seq_len,
    heads,
    head_dim,
    scale,
    BLOCK_SIZE_QK: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    h = tl.program_id(0)
    i = tl.program_id(1)
    # i: query position
    idx_q = i * heads + h
    # Load Q
    q = tl.load(q_ptr + i * heads * head_dim + h * head_dim, mask=i < seq_len, other=0.0)
    # Accumulator for attention
    attn = tl.zeros([seq_len], dtype=tl.float32)

    # Compute scores
    for j in range(0, seq_len, BLOCK_SIZE_QK):
        j_offset = j + tl.arange(0, BLOCK_SIZE_QK)
        k = tl.load(k_ptr + j_offset * heads * head_dim + h * head_dim,
                    mask=j_offset < seq_len, other=0.0)
        scores = tl.dot(q, k, transpose_b=True) * scale
        attn += scores

    # Softmax
    max_val = tl.max(attn)
    exp = tl.exp(attn - max_val)
    sum_exp = tl.sum(exp)
    probs = exp / sum_exp

    # Weighted sum of V
    out = tl.zeros([head_dim], dtype=tl.float32)
    for j in range(0, seq_len, BLOCK_SIZE_V):
        j_offset = j + tl.arange(0, BLOCK_SIZE_V)
        v = tl.load(v_ptr + j_offset * heads * head_dim + h * head_dim,
                    mask=j_offset < seq_len, other=0.0)
        out += v * probs[j_offset]

    tl.store(out_ptr + idx_q * head_dim, out, mask=i < seq_len)


# --------------------------------------------------------------
# Triton kernel: LayerNorm
# --------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['seq_len', 'embed_dim'],
)
@triton.jit
def _layernorm_kernel(
    inp_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    seq_len,
    embed_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    offs = idx + tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_len

    # Load input
    inp = tl.load(inp_ptr + offs * embed_dim, mask=mask, other=0.0)
    # Compute mean
    mean = tl.sum(inp) / embed_dim
    # Compute variance
    var = tl.sum((inp - mean) ** 2) / embed_dim
    inv = tl.rsqrt(var + eps)
    # Normalize
    out = (inp - mean) * inv
    # Scale & shift
    out = out * tl.load(gamma_ptr, mask=mask, other=1.0) + tl.load(beta_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + offs * embed_dim, out, mask=mask)


# --------------------------------------------------------------
# Custom attention block
# --------------------------------------------------------------
class CustomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        # Projection weights
        self.wq = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.wk = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.wv = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.wo = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.wq)
        nn.init.xavier_uniform_(self.wk)
        nn.init.xavier_uniform_(self.wv)
        nn.init.xavier_uniform_(self.wo)

        # LayerNorm parameters
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, C, H, W) -> (seq_len, B, C)
        """
        B, C, H, W = x.shape
        seq_len = H * W
        x_flat = x.view(B, C, seq_len).permute(2, 0, 1)  # (seq_len, B, C)

        # Q, K, V projection
        # Reshape weights for Triton: (C, C)
        # For simplicity, use torch.matmul here and focus on attention core
        qkv = torch.matmul(x_flat, torch.cat([self.wq, self.wk, self.wv], dim=1))  # (seq_len, B, 3C)
        q = qkv[..., :C]
        k = qkv[..., C:2*C]
        v = qkv[..., 2*C:]

        # Reshape to (seq_len, B, heads, head_dim)
        q = q.reshape(seq_len, B, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, B, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, B, self.num_heads, self.head_dim)

        # Transpose to (B, heads, seq_len, head_dim) for easier batching
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Prepare output buffer
        out = torch.empty_like(q)

        # Attention scaling
        scale = 1.0 / (self.head_dim ** 0.5)

        # Launch Triton attention kernel
        grid = lambda meta: (self.num_heads, seq_len)
        _attention_kernel[grid](
            q, k, v, out,
            seq_len, self.num_heads, self.head_dim,
            scale,
        )

        # Reshape back
        out = out.permute(2, 0, 1, 3).reshape(seq_len, B, C)

        # Output projection
        out = torch.matmul(out, self.wo)

        # Add residual and norm
        out = self.norm(out + x_flat)

        # Reshape back to image
        out = out.permute(1, 2, 0).view(B, C, H, W)
        return out


# --------------------------------------------------------------
# Final model
# --------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = CustomAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.attn(x)
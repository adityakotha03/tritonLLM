import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional


# ----------------------------------------
# Triton Kernels
# ----------------------------------------

@triton.jit
def _fused_matmul_gelu_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    out_ptr,
    n_elements: tl.int32,
    n_heads: tl.int32,
    head_dim: tl.int32,
    seq_len: tl.int32,
    n_embd: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Define block-wide offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load input data (B, T, C) -> (B * T, C)
    batch_size = n_elements // (seq_len * n_embd)
    batch_offset = (pid // (seq_len * n_heads)) * seq_len * n_embd
    seq_offset = ((pid % (seq_len * n_heads)) // n_heads) * n_embd
    head_offset = (pid % n_heads) * head_dim
    total_offset = batch_offset + seq_offset + head_offset

    # We are handling one head at a time in a fused block, so reshape to (B*T, head_dim)
    # But we will do it in a streaming fashion per head.
    # Instead, let's tile over head_dim and seq_len to avoid large memory bandwidth.
    # We will process one (head_dim) block at a time.
    # So we process B*T * head_dim elements per head

    # Get current head and seq
    head_id = (pid % n_heads)
    seq_id = (pid // n_heads) % seq_len
    batch_id = pid // (n_heads * seq_len)

    # Reuse offset
    x_offset = batch_id * seq_len * n_embd + seq_id * n_embd + head_id * head_dim
    x = tl.load(x_ptr + x_offset, mask=offsets < head_dim, other=0.0)

    # Compute w1 @ x for (head_dim, 4 * head_dim)
    w1_block = w1_ptr + head_id * head_dim * (4 * head_dim)
    w2_block = w2_ptr + head_id * (4 * head_dim) * head_dim

    # Compute matmul: (head_dim, 4*head_dim) @ (4*head_dim,) -> (head_dim,)
    w1 = tl.load(w1_block + offsets[:, None] * (4 * head_dim) + tl.arange(0, 4 * head_dim)[None, :], mask=offsets[:, None] < head_dim, other=0.0)
    w1_flat = w1.view(head_dim, 4 * head_dim)
    x_reshaped = x[:, None]  # (head_dim, 1)
    y = tl.dot(w1_flat, x_reshaped).squeeze(1)  # (4*head_dim,)

    # Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Use optimized approximation
    sqrt_2_over_pi = 0.7978845608028654
    c_3 = 0.044715
    y = y * 0.5 * (1.0 + tl.tanh(sqrt_2_over_pi * (y + c_3 * y * y * y)))

    # Compute w2 @ y
    w2 = tl.load(w2_block + offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :], mask=offsets[:, None] < head_dim, other=0.0)
    w2_flat = w2.view(4 * head_dim, head_dim)
    y = y[None, :]  # (1, 4*head_dim)
    z = tl.dot(y, w2_flat).squeeze(0)  # (head_dim,)

    # Store output
    out_offset = x_offset
    tl.store(out_ptr + out_offset, z, mask=offsets < head_dim)


@triton.jit
def _causal_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    bias_ptr,
    n_heads: tl.int32,
    head_dim: tl.int32,
    seq_len: tl.int32,
    scale: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Block indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    # Compute row and column indices within the block
    row_idx = block_start + tl.arange(0, BLOCK_SIZE)
    col_idx = block_start + tl.arange(0, BLOCK_SIZE)

    # Load bias for causal masking
    bias_mask = tl.load(bias_ptr + row_idx[:, None] * seq_len + col_idx[None, :], mask=row_idx[:, None] < seq_len, other=0.0)
    bias_mask = (bias_mask == 0.0)  # True where mask is 0 -> masked

    # Loop over K blocks (to support large sequences)
    q = tl.load(q_ptr + row_idx[:, None] * head_dim + tl.arange(0, head_dim)[None, :], mask=row_idx[:, None] < seq_len, other=0.0)
    # Accumulate attention weights
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over K in blocks of BLOCK_SIZE
    for k_offset in range(0, seq_len, BLOCK_SIZE):
        k_start = k_offset
        k_end = min(k_offset + BLOCK_SIZE, seq_len)
        k_col = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_col < seq_len

        # Load K block
        k = tl.load(k_ptr + k_col[:, None] * head_dim + tl.arange(0, head_dim)[None, :], mask=k_mask[:, None], other=0.0)
        # Transpose K: (head_dim, BLOCK_SIZE)
        k = tl.trans(k)

        # Compute Q @ K^T
        qk = tl.dot(q, k) * scale
        qk = qk * (1.0 - tl.cast(bias_mask, tl.float32))
        # Add mask for causal attention
        qk = tl.where(bias_mask, float("-inf"), qk)

        # Softmax over columns (last dim)
        # Use online softmax: reduce-max and exp-normalization
        qk_max = tl.max(qk, axis=1, keepdims=True)
        qk_exp = tl.exp(qk - qk_max)
        qk_sum = tl.sum(qk_exp, axis=1, keepdims=True)
        qk_norm = qk_exp / qk_sum

        # Accumulate attention
        acc += qk_norm

    # Load V block
    v_start = block_start
    v_end = min(block_end, seq_len)
    v_row = v_start + tl.arange(0, BLOCK_SIZE)
    v_mask = v_row < seq_len

    v = tl.load(v_ptr + v_row[:, None] * head_dim + tl.arange(0, head_dim)[None, :], mask=v_mask[:, None], other=0.0)
    # Compute acc @ V
    output = tl.dot(acc, v)

    # Store output
    out_offsets = row_idx[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    tl.store(out_ptr + out_offsets, output, mask=row_idx[:, None] < seq_len)


@triton.jit
def _fused_mlp_attention_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    c_attn_ptr,
    c_proj_ptr,
    bias_ptr,
    n_heads: tl.int32,
    head_dim: tl.int32,
    seq_len: tl.int32,
    scale: tl.float32,
    dropout_mask_ptr: tl.pointer(tl.int1),
    dropout_ratio: tl.float32,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of (B * T) elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets[:, None] * 768 + tl.arange(0, 768)[None, :], mask=mask[:, None], other=0.0)

    # Split x into Q, K, V
    qkv = tl.load(c_attn_ptr + offsets[:, None] * 3 * 768 + tl.arange(0, 3 * 768)[None, :], mask=mask[:, None], other=0.0)
    q, k, v = qkv.split(768, dim=1)

    # Reshape: (B*T, 768) -> (B*T, 8, 96)
    q = q.view(-1, 8, 96)
    k = k.view(-1, 8, 96)
    v = v.view(-1, 8, 96)

    # Causal attention: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
    # We do it per head with block-wise attention
    # Each block processes one head
    head_idx = pid % 8
    head_offset = head_idx * 96
    q_head = q[:, head_idx, :]  # (B*T, 96)
    k_head = k[:, head_idx, :]  # (B*T, 96)
    v_head = v[:, head_idx, :]  # (B*T, 96)

    # Compute attention: (B*T, 96) @ (B*T, 96).T -> (B*T, B*T)
    # Use Triton block matrix multiplication with online softmax
    qk = tl.dot(q_head, tl.trans(k_head)) * scale
    # Apply causal mask
    qk = tl.load(bias_ptr + offsets[:, None] * seq_len + offsets[None, :] * 1, mask=offsets[:, None] < seq_len, other=0.0)
    qk = qk * (1.0 - tl.cast(qk == 0.0, tl.float32))
    qk = tl.where(qk == 0.0, float("-inf"), qk)

    # Online softmax
    qk_max = tl.max(qk, axis=1, keepdims=True)
    qk_exp = tl.exp(qk - qk_max)
    qk_sum = tl.sum(qk_exp, axis=1, keepdims=True)
    qk_norm = qk_exp / qk_sum

    # Apply dropout
    dropout_mask = tl.load(dropout_mask_ptr + offsets[:, None] * seq_len + offsets[None, :], mask=offsets[:, None] < seq_len, other=0.0)
    qk_norm = qk_norm * (1.0 - dropout_ratio * dropout_mask)

    # Final output: (B*T, B*T) @ (B*T, 96) -> (B*T, 96)
    output = tl.dot(qk_norm, v_head)

    # Reshape back: (B*T, 96) -> (B*T, 768)
    output = output.view(-1, 768)

    # Apply MLP: Linear1 -> GELU -> Linear2
    # Load W1 and W2
    w1 = tl.load(w1_ptr + offsets[:, None] * 3072 + tl.arange(0, 3072)[None, :], mask=mask[:, None], other=0.0)
    w2 = tl.load(w2_ptr + offsets[:, None] * 768 + tl.arange(0, 768)[None, :], mask=mask[:, None], other=0.0)

    # Compute W1 @ x
    hidden = tl.dot(output, w1.t())
    # GELU activation
    sqrt_2_over_pi = 0.7978845608028654
    c_3 = 0.044715
    hidden = 0.5 * hidden * (1.0 + tl.tanh(sqrt_2_over_pi * (hidden + c_3 * hidden * hidden * hidden)))
    # Compute W2 @ hidden
    output = tl.dot(hidden, w2.t())

    # Final projection
    final_proj = tl.load(c_proj_ptr + offsets[:, None] * 768 + tl.arange(0, 768)[None, :], mask=mask[:, None], other=0.0)
    output = output + final_proj

    # Store output
    tl.store(x_ptr + offsets[:, None] * 768 + tl.arange(0, 768)[None, :], output, mask=mask[:, None])


# ----------------------------------------
# Wrapper Functions
# ----------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['n_elements', 'seq_len', 'n_heads'],
)
def triton_fused_mlp_attention(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    c_attn: torch.Tensor,
    c_proj: torch.Tensor,
    bias: torch.Tensor,
    dropout_mask: torch.Tensor,
    dropout_ratio: float,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    scale: float,
):
    assert x.is_cuda
    assert x.shape[-1] == 768
    assert x.shape[1] == seq_len
    assert x.dtype in (torch.float16, torch.bfloat16)

    batch_size = x.shape[0]
    n_elements = x.numel()

    # Create output
    out = torch.empty_like(x)

    # Grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    _fused_mlp_attention_kernel[grid](
        x,
        w1,
        w2,
        c_attn,
        c_proj,
        bias,
        dropout_mask,
        dropout_ratio,
        n_heads,
        head_dim,
        seq_len,
        scale,
        n_elements,
        BLOCK_SIZE=128,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
    ],
    key=['n_elements', 'seq_len'],
)
def triton_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    seq_len: int,
    head_dim: int,
):
    assert q.is_cuda
    assert q.shape == k.shape == v.shape
    assert q.dtype in (torch.float16, torch.bfloat16)

    # B, T, C
    B, T, C = q.shape
    assert C == head_dim * 8
    n_elements = q.numel()

    # Create output
    out = torch.empty_like(q)

    # Grid
    grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE']),)

    # Launch kernel
    _causal_attention_kernel[grid](
        q,
        k,
        v,
        out,
        bias,
        8,
        head_dim,
        seq_len,
        scale,
        BLOCK_SIZE=128,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
def triton_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    n_embd: int,
    seq_len: int,
    head_dim: int,
):
    assert x.is_cuda
    assert x.shape[-1] == n_embd
    assert x.dtype in (torch.float16, torch.bfloat16)

    n_elements = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _fused_matmul_gelu_kernel[grid](
        x,
        w1,
        w2,
        out,
        n_elements,
        8,
        head_dim,
        seq_len,
        n_embd,
        BLOCK_SIZE=128,
    )
    return out


# ----------------------------------------
# ModelNew
# ----------------------------------------

class ModelNew(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float, max_seqlen: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.max_seqlen = max_seqlen

        # LayerNorms
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        # Causal attention (split into c_attn and c_proj)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # MLP
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))

        # Causal bias
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))

        # Scale for attention
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # Dropout mask for attention
        self.register_buffer("dropout_mask", torch.ones(1024, 1024, dtype=torch.int))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer norm
        x_ln1 = self.ln_1(x)

        # Causal self-attention with Triton fusion
        # Split q, k, v
        qkv = self.c_attn(x_ln1)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.head_dim).transpose(1, 2)

        # Triton causal attention
        # Ensure contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        bias = self.bias[:, :, :q.shape[2], :q.shape[2]].contiguous()
        attn_out = triton_causal_attention(q, k, v, bias, self.scale, q.shape[2], self.head_dim)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], q.shape[3] * q.shape[1])

        # Project
        attn_out = self.c_proj(attn_out)

        # Dropout and residual
        attn_out = F.dropout(attn_out, p=self.attn_pdrop, training=self.training)

        # First residual connection
        x = x + attn_out

        # Layer norm
        x_ln2 = self.ln_2(x)

        # Apply MLP with Triton
        mlp_out = triton_mlp(x_ln2, self.mlp.c_fc.weight, self.mlp.c_proj.weight, self.n_embd, x.shape[1], self.head_dim)

        # Dropout and residual
        mlp_out = self.mlp.dropout(mlp_out)

        # Final residual
        x = x + mlp_out
        return x
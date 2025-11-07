import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def causal_flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    bias_ptr,
    seqlen, n_head, head_dim,
    stride_q, stride_k, stride_v,
    stride_out, stride_bias,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQLen: tl.constexpr,
    ENABLE_MASK: tl.constexpr,
):
    # Shared memory for block-level reductions
    pid = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_block = tl.program_id(2)

    # Block size is the same as head_dim for efficiency, but we'll handle larger seqlens
    BLOCK_SIZE = tl.constexpr(128)
    HEAD_DIM = tl.constexpr(64)

    # Calculate offsets
    block_start = pid_block * BLOCK_SIZE
    offs_q = block_start + tl.arange(0, BLOCK_SIZE)
    offs_k = block_start + tl.arange(0, BLOCK_SIZE)
    offs_head = pid_head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    offs_batch = pid * 1  # Single batch for now

    # Load q, k, v (batch, head, seq, dim)
    q = tl.load(q_ptr + offs_batch * stride_q + pid_head * HEAD_DIM * seqlen + offs_q[:, None] * HEAD_DIM + offs_head[None, :], 
                mask=offs_q[:, None] < seqlen, other=0.0)
    k = tl.load(k_ptr + offs_batch * stride_k + pid_head * HEAD_DIM * seqlen + offs_k[:, None] * HEAD_DIM + offs_head[None, :], 
                mask=offs_k[:, None] < seqlen, other=0.0)
    v = tl.load(v_ptr + offs_batch * stride_v + pid_head * HEAD_DIM * seqlen + offs_k[:, None] * HEAD_DIM + offs_head[None, :], 
                mask=offs_k[:, None] < seqlen, other=0.0)

    # Initialize softmax and output accumulation
    acc = tl.zeros((BLOCK_SIZE, HEAD_DIM), dtype=tl.float32)
    lse = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')

    # Causal mask: only allow leftward attention
    # We can optimize by only computing up to the diagonal
    for i in range(0, seqlen, BLOCK_SIZE):
        i_offset = i
        if i_offset >= seqlen:
            break

        # Load k and v from the i-th block
        k_i = tl.load(k_ptr + offs_batch * stride_k + pid_head * HEAD_DIM * seqlen + 
                      (i_offset + offs_k[:, None]) * HEAD_DIM + offs_head[None, :], 
                      mask=(i_offset + offs_k[:, None]) < seqlen, other=0.0)

        v_i = tl.load(v_ptr + offs_batch * stride_v + pid_head * HEAD_DIM * seqlen + 
                      (i_offset + offs_k[:, None]) * HEAD_DIM + offs_head[None, :], 
                      mask=(i_offset + offs_k[:, None]) < seqlen, other=0.0)

        # Compute q @ k.T, only for valid positions
        qk = tl.dot(q, k_i.T)
        qk = qk * (1.0 / tl.sqrt(float(HEAD_DIM)))
        qk += tl.load(bias_ptr + offs_batch * stride_bias + pid_head * seqlen * seqlen + 
                      offs_q[:, None] * seqlen + (i_offset + offs_k[None, :]), 
                      mask=(offs_q[:, None] >= i_offset + offs_k[None, :]), other=float('-inf'))
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        p_ij = tl.exp(qk - m_ij[:, None])
        l_ij = tl.exp(m_ij - lse)
        lse_new = lse + l_ij

        # Update accumulation
        acc = acc * (lse / lse_new)[:, None] + p_ij @ v_i
        lse = lse_new

    # Store output
    tl.store(out_ptr + offs_batch * stride_out + pid_head * HEAD_DIM * seqlen + 
             offs_q[:, None] * HEAD_DIM + offs_head[None, :], 
             acc, mask=offs_q[:, None] < seqlen)


@triton.jit
def fused_mlp_kernel(
    x_ptr, w1_ptr, w2_ptr, out_ptr,
    n_embd, n_hidden, seqlen, n_head,
    stride_x, stride_w1, stride_w2,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_head = tl.program_id(1)

    # Block size for MLP: typically 128 or 256
    BLOCK_SIZE = tl.constexpr(256)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_head = pid_head * 64 + tl.arange(0, 64)

    # Load x
    x = tl.load(x_ptr + offs[:, None] * stride_x + offs_head[None, :], 
                mask=offs[:, None] < seqlen, other=0.0)

    # Load W1 and apply GELU
    w1 = tl.load(w1_ptr + offs[:, None] * stride_w1 + offs_head[None, :], 
                 mask=offs[:, None] < n_hidden, other=0.0)
    hidden = x @ w1
    hidden = tl.where(hidden > 0, hidden, 0.5 * hidden * (1 + tl.tanh(0.7978845608028654 * (hidden + 0.044715 * hidden**3))))

    # Load W2 and apply dropout
    w2 = tl.load(w2_ptr + offs[:, None] * stride_w2 + offs_head[None, :], 
                 mask=offs[:, None] < n_embd, other=0.0)
    out = hidden @ w2

    # Store output
    tl.store(out_ptr + offs[:, None] * stride_out + offs_head[None, :], 
             out, mask=offs[:, None] < seqlen)


class TritonCausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd
        self.max_seqlen = max_seqlen

        # Projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen))

        # Register buffers for Triton
        self.register_buffer("head_dim", torch.tensor(self.head_dim, dtype=torch.int32))

    def forward(self, x):
        B, T, C = x.size()

        # Project to q, k, v
        q, k, v = self.c_attn(x).split(C, dim=-1)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Prepare for Triton kernel
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        bias = self.bias[:, :, :T, :T].to(torch.bfloat16)

        # Allocate output
        out = torch.empty_like(q)

        # Define grid
        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE']), meta['n_head'], 1)

        # Launch Triton kernel
        causal_flash_attn_kernel[grid](
            q, k, v, out,
            bias,
            T, self.n_head, self.head_dim,
            q.stride(0), k.stride(0), v.stride(0),
            out.stride(0), bias.stride(0),
            BLOCK_SIZE=128,
            HEAD_DIM=64,
            SEQLen=T,
            ENABLE_MASK=True
        )

        # Reshape and project
        y = out.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class TritonMLP(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.w1 = nn.Linear(n_embd, n_hidden, bias=False)
        self.w2 = nn.Linear(n_hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # Convert to bfloat16 for Triton
        x = x.to(torch.bfloat16)
        out = torch.empty_like(x)

        # Launch Triton kernel
        grid = lambda meta: (triton.cdiv(x.size(1), meta['BLOCK_SIZE']), 1)

        fused_mlp_kernel[grid](
            x, self.w1.weight, self.w2.weight, out,
            x.size(-1), self.w1.out_features, x.size(1), 1,
            x.stride(0), self.w1.weight.stride(0), self.w2.weight.stride(0),
            out.stride(0),
            BLOCK_SIZE=256
        )

        return self.dropout(out.to(torch.float32))


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn = TritonCausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.mlp = TritonMLP(n_embd, 4 * n_embd)

    def forward(self, x):
        # Attention + Residual
        x = x + self.attn(x)
        # MLP + Residual
        x = x + self.mlp(x)
        return x
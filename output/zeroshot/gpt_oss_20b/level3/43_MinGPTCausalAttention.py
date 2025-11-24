import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# ------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------

# 1️⃣  Projection:  x (B,T,C)  →  q,k,v (B,nh,T,hs)
#     We compute all 3 projections in one kernel to reduce memory traffic.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_K": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _qkv_projection_kernel(
    x_ptr,          # (B,T,C)
    w_ptr,          # (C, 3*C)
    out_ptr,        # (B, nh, T, 3*hs)  (flattened)
    B: tl.constexpr,    # batch size
    T: tl.constexpr,    # sequence length
    C: tl.constexpr,    # hidden dim
    nh: tl.constexpr,   # number of heads
    hs: tl.constexpr,   # head size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # iterate over the K dimension (C) in tiles
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    for k in range(0, C, BLOCK_SIZE_K):
        x_block = tl.load(x_ptr + k + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None], 
                          mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < T,
                          other=0.0)
        w_block = tl.load(w_ptr + k + pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :],
                          mask=tl.arange(0, BLOCK_SIZE_K)[None, :] < 3 * C,
                          other=0.0)
        acc += tl.dot(x_block, w_block)

    # write the output (B, nh, T, 3*hs)
    out_block = acc.to(tl.float16)
    base = pid_n * BLOCK_SIZE_K
    tl.store(out_ptr + base + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None],
             out_block,
             mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < T)

# 2️⃣  Attention:  (B, nh, T, T) × (B, nh, T, hs) → (B, nh, T, hs)
#     We fuse softmax + dropout + matrix multiplication with v.
#     The bias (causal mask) is passed as a buffer.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _attention_kernel(
    attn_ptr,       # (B, nh, T, T)
    v_ptr,          # (B, nh, T, hs)
    out_ptr,        # (B, nh, T, hs)
    bias_ptr,       # (1, 1, T, T)
    B: tl.constexpr,
    nh: tl.constexpr,
    T: tl.constexpr,
    hs: tl.constexpr,
    dropout: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0) * BLOCK_SIZE_M
    pid_n = tl.program_id(1) * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

    # load attention logits and apply bias and softmax
    for k in range(0, T, BLOCK_SIZE_K):
        attn_block = tl.load(attn_ptr + k + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None],
                             mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < T,
                             other=-1e9)
        bias_block = tl.load(bias_ptr + k + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None],
                             mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < T,
                             other=0.0)
        attn_block = tl.where(bias_block == 0, -1e9, attn_block)

        # softmax along the k dimension
        max_val = tl.max(attn_block, axis=1)[:, None]
        exp_vals = tl.exp(attn_block - max_val)
        sum_exp = tl.sum(exp_vals, axis=1)[:, None]
        attn_block = exp_vals / (sum_exp + 1e-6)

        if dropout > 0.0:
            # simple dropout: zero out half of the elements
            rand = tl.uniform(tl.arange(0, BLOCK_SIZE_M)[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
            attn_block = tl.where(rand > dropout, attn_block, 0.0)

        acc += tl.dot(attn_block, tl.load(v_ptr + k + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_K)[None, :],
                                         mask=tl.arange(0, BLOCK_SIZE_K)[None, :] < hs,
                                         other=0.0))

    tl.store(out_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None],
             acc.to(tl.float16),
             mask=tl.arange(0, BLOCK_SIZE_M)[:, None] < T)

# ------------------------------------------------------------
# Triton wrappers
# ------------------------------------------------------------

class TritonQKV:
    def __init__(self, weight: torch.Tensor, B: int, T: int, C: int, nh: int, hs: int):
        self.weight = weight.contiguous()
        self.B = B
        self.T = T
        self.C = C
        self.nh = nh
        self.hs = hs

    def __call__(self, x: torch.Tensor):
        # x: (B,T,C)
        out = torch.empty(self.B, self.nh, self.T, 3 * self.hs, dtype=torch.float16, device=x.device)
        grid = lambda meta: (
            (self.T + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
            (3 * self.C + meta["BLOCK_SIZE_K"] - 1) // meta["BLOCK_SIZE_K"],
        )
        _qkv_projection_kernel[grid](
            x, self.weight, out, self.B, self.T, self.C, self.nh, self.hs,
            BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
            BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
        )
        return out.float()

class TritonAttention:
    def __init__(self, bias: torch.Tensor, dropout: float):
        self.bias = bias.contiguous()
        self.dropout = dropout

    def __call__(self, attn: torch.Tensor, v: torch.Tensor):
        # attn: (B,nh,T,T)
        # v:    (B,nh,T,hs)
        B, nh, T, _ = attn.shape
        hs = v.shape[-1]
        out = torch.empty_like(v)
        grid = lambda meta: (
            (T + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
            (T + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        )
        _attention_kernel[grid](
            attn, v, out, self.bias,
            B, nh, T, hs, self.dropout,
            BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
        )
        return out.float()

# ------------------------------------------------------------
# New model
# ------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.hs = n_embd // n_head

        # linear projection weights
        self.c_attn_weight = nn.Parameter(torch.empty(n_embd, 3 * n_embd))
        nn.init.xavier_uniform_(self.c_attn_weight)

        # output projection weights
        self.c_proj_weight = nn.Parameter(torch.empty(n_embd, n_embd))
        nn.init.xavier_uniform_(self.c_proj_weight)

        # dropout layers
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # causal mask
        bias = torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        # 1️⃣ QKV projection with Triton
        qkv_proj = TritonQKV(self.c_attn_weight, B, T, C, self.n_head, self.hs)
        qkv = qkv_proj(x)  # (B, nh, T, 3*hs)
        q, k, v = qkv.chunk(3, dim=-1)  # each: (B, nh, T, hs)

        # 2️⃣ Scaled dot-product attention
        attn_logits = torch.einsum("bnhqd,bnhkd->bnhqk", q, k) / math.sqrt(self.hs)

        # 3️⃣ Causal mask
        attn_logits = attn_logits.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # 4️⃣ Softmax
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 5️⃣ Attention with Triton (softmax already applied)
        attn = TritonAttention(self.bias, self.attn_dropout.p)
        attn_out = attn(attn_probs, v)  # (B, nh, T, hs)

        # 6️⃣ Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # 7️⃣ Output projection
        out = F.linear(attn_out, self.c_proj_weight)
        out = self.resid_dropout(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# Triton kernels ------------------------------------------------------------

# -----------------------------------------------------------------------------
# Attention kernel that computes (q @ k^T * scale) masked by a causal mask,
# applies ReLU, and then multiplies by v.
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["B", "T", "hs"],
)
@triton.jit
def attn_kernel(
    Q,  # (B*nh, T, hs)
    K,  # (B*nh, T, hs)
    V,  # (B*nh, T, hs)
    out,  # (B*nh, T, hs)
    scale: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    hs: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    A simple fused attention kernel that
        1. Computes (Q @ K^T) * scale
        2. Applies a causal mask (lower triangular) and ReLU
        3. Computes the final output = (masked_relaxed_attention @ V)
    The kernel assumes B*nh blocks are flattened into the batch dimension.
    """
    pid_m = tl.program_id(0)  # row index of the output
    pid_n = tl.program_id(1)  # column index of the output

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # Accumulator for the first matmul: att = Q @ K^T
    acc_att = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Compute Q @ K^T in tiles of K
    for k in range(0, T, BLOCK_SIZE_K):
        # Load tiles
        Q_tile = tl.load(
            Q + row_start * hs + k * hs + tl.arange(0, BLOCK_SIZE_M)[:, None],
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < B * hs) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < T),
            other=0.0,
        ).to(tl.float32)

        K_tile = tl.load(
            K + (k * hs + col_start * hs) + tl.arange(0, BLOCK_SIZE_K)[:, None],
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None] < T) &
                 (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < B * hs),
            other=0.0,
        ).to(tl.float32)

        acc_att += tl.dot(Q_tile, K_tile, allow_tf32=False)

    # Scale
    acc_att *= scale

    # Apply causal mask and ReLU in the same step
    # mask: 1 if (row_index + row_start) >= (col_index + col_start) else 0
    row_indices = row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
    col_indices = col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
    mask = (row_indices >= col_indices).to(tl.float32)

    acc_att = tl.max(acc_att * mask, 0.0)  # ReLU

    # Second matmul: acc_att @ V
    acc_out = tl.zeros([BLOCK_SIZE_M, hs], dtype=tl.float32)

    for k in range(0, BLOCK_SIZE_N, BLOCK_SIZE_K):
        # Load attention tile
        att_tile = tl.load(
            acc_att + row_start * BLOCK_SIZE_N + k * hs,
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < B * hs) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < BLOCK_SIZE_N),
            other=0.0,
        ).to(tl.float32)

        V_tile = tl.load(
            V + (col_start + k) * hs + tl.arange(0, hs)[None, :],
            mask=(col_start + k + tl.arange(0, BLOCK_SIZE_K)[None, :] < B * hs) &
                 (tl.arange(0, hs)[None, :] < hs),
            other=0.0,
        ).to(tl.float32)

        acc_out += tl.dot(att_tile, V_tile, allow_tf32=False)

    # Store result
    tl.store(
        out + row_start * hs + tl.arange(0, BLOCK_SIZE_M)[:, None],
        acc_out,
        mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < B * hs),
    )


# Utility --------------------------------------------------------------------

def triton_attention(qkv, T, scale):
    """
    qkv: tensor of shape (B, T, 3*hs)
    Returns tensor of shape (B, T, hs) after fused attention and ReLU.
    """
    B, _, hs = qkv.shape
    # Flatten heads and batch: (B, T, 3*hs) -> (B, T, 3, hs) -> (3, B, T, hs)
    qkv = qkv.view(B, T, 3, hs).transpose(1, 2)  # (3, B, T, hs)
    q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, T, hs)

    # Reshape to (B*nh, T, hs) where nh=1 (since we already collapsed heads)
    # In the original model, heads are handled by splitting inside the kernel.
    # For simplicity, we treat each batch element independently.
    # If multi-head is needed, reshape accordingly.
    # Here we keep the heads as part of the batch dimension.
    out = torch.empty_like(q)

    # Launch Triton kernel
    grid = lambda meta: (
        (q.shape[0] * q.shape[1] + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (hs + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )
    attn_kernel[grid](
        q,
        k,
        v,
        out,
        scale=scale,
        B=q.shape[0] * q.shape[1],
        T=T,
        hs=hs,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=32,
    )

    return out


# Model -----------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen))
            .view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        # 1. QKV projection
        qkv = self.c_attn(x)  # (B, T, 3*C)

        # 2. Split into heads
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hs)

        # 3. Compute attention for each head with Triton
        q = qkv[0].reshape(B * self.n_head, T, self.head_dim)
        k = qkv[1].reshape(B * self.n_head, T, self.head_dim)
        v = qkv[2].reshape(B * self.n_head, T, self.head_dim)

        # Attn: (q @ k^T) * scale, causal mask, ReLU, then @ v
        attn_out = torch.empty_like(q)

        # Use Triton kernel
        grid = lambda meta: (
            (q.shape[0] + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
            (self.head_dim + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        )
        attn_kernel[grid](
            q,
            k,
            v,
            attn_out,
            scale=self.scale,
            B=q.shape[0],
            T=T,
            hs=self.head_dim,
            BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=32,
        )

        # 4. Reshape back to (B, T, n_embd)
        y = attn_out.view(B, self.n_head, T, self.head_dim)
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)

        # 5. Final projection
        y = self.c_proj(y)
        return y
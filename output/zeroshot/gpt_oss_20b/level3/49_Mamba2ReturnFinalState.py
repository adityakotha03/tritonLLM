import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_L': 256, 'BLOCK_SIZE_R': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_L': 512, 'BLOCK_SIZE_R': 512}, num_warps=8),
    ],
    key=['m', 'n', 'k'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # pointers
    m, n, k,              # sizes
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_L: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_l = tl.program_id(axis=0)
    pid_r = tl.program_id(axis=1)

    row_start = pid_l * BLOCK_SIZE_L
    col_start = pid_r * BLOCK_SIZE_R

    row_end = tl.min(row_start + BLOCK_SIZE_L, m)
    col_end = tl.min(col_start + BLOCK_SIZE_R, n)

    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_R), dtype=tl.float32)

    for p in range(0, k, BLOCK_SIZE_R):
        cur_k = min(k - p, BLOCK_SIZE_R)

        a = tl.load(
            a_ptr + (row_start + tl.arange(0, BLOCK_SIZE_L))[:, None] * stride_am
            + (p + tl.arange(0, cur_k))[None, :] * stride_an,
            mask=(
                (row_start + tl.arange(0, BLOCK_SIZE_L))[:, None] < m
                & (p + tl.arange(0, cur_k))[None, :] < k
            ),
            other=0.0,
        )

        b = tl.load(
            b_ptr + (p + tl.arange(0, cur_k))[:, None] * stride_bk
            + (col_start + tl.arange(0, BLOCK_SIZE_R))[None, :] * stride_bn,
            mask=(
                (p + tl.arange(0, cur_k))[:, None] < k
                & (col_start + tl.arange(0, BLOCK_SIZE_R))[None, :] < n
            ),
            other=0.0,
        )

        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(
        c_ptr
        + (row_start + tl.arange(0, BLOCK_SIZE_L))[:, None] * stride_cm
        + (col_start + tl.arange(0, BLOCK_SIZE_R))[None, :] * stride_cn,
        c,
        mask=(
            (row_start + tl.arange(0, BLOCK_SIZE_L))[:, None] < m
            & (col_start + tl.arange(0, BLOCK_SIZE_R))[None, :] < n
        ),
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['size'],
)
@triton.jit
def _exp_kernel(x_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.math.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)


# ------------------------------------------------------------------
# Helper wrappers
# ------------------------------------------------------------------
def triton_exp(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    size = x.numel()
    grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
    _exp_kernel[grid](x, out, size, BLOCK_SIZE=128)
    return out


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_L"]),
                         triton.cdiv(n, meta["BLOCK_SIZE_R"]))
    _matmul_kernel[grid](
        a, b, out,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_L=256,
        BLOCK_SIZE_R=256,
    )
    return out


# ------------------------------------------------------------------
# Optimised Model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    # ------------------------------------------------------------------
    # 1. segment sum using cumulative max to avoid inf
    def segsum(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        # mask to keep only lower triangular part
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        x_segsum = x_segsum.masked_fill(~mask, float('-inf'))
        return x_segsum

    # ------------------------------------------------------------------
    def forward(self, X, initial_states=None):
        # ---- rearrange to blocks ----
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # ---- 1. diagonal block outputs ----
        L = torch.exp(self.segsum(A_blocks))          # shape (b,h,c,l)
        # reshape for matmul
        C_flat = C_blocks.reshape(-1, self.d_state)   # (b*h*c*l, d_state)
        B_flat = B_blocks.reshape(-1, self.d_state)   # (b*h*c*l, d_state)
        X_flat = X_blocks.reshape(-1, self.d_head)    # (b*h*c*l, d_head)
        L_flat = L.reshape(-1, 1)                     # (b*h*c*l, 1)

        # use triton matmul for each block
        temp = triton_matmul(C_flat, B_flat.t())      # (b*h*c*l, d_state)
        temp = temp * L_flat                         # broadcast
        temp = triton_matmul(temp, X_flat.t())       # (b*h*c*l, d_head)
        Y_diag = temp.t().reshape(self.batch_size, self.n_heads, self.seq_length // self.block_len, self.block_len, self.d_head)

        # ---- 2. intra-chunk states ----
        decay_states = torch.exp((A_cumsum[..., :, -1:] - A_cumsum))  # (b,h,c,l)
        B_flat2 = B_blocks.reshape(-1, self.d_state)                 # (b*h*c*l, d_state)
        decay_flat = decay_states.reshape(-1, 1)                     # (b*h*c*l,1)
        X_flat2 = X_blocks.reshape(-1, self.d_head)                  # (b*h*c*l, d_head)

        states = triton_matmul(B_flat2, X_flat2.t())                 # (b*h*c*l, d_state)
        states = states * decay_flat                                 # (b*h*c*l, d_state)
        states = states.t().reshape(self.batch_size, self.n_heads, self.seq_length // self.block_len, self.block_len, self.d_state)

        # ---- 3. inter-chunk recurrence ----
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :, :1, :, :])  # (b,h,1,l,d_state)
        states = torch.cat([initial_states, states], dim=2)           # (b,h,c+1,l,d_state)

        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[..., :, -1], (1, 0))))  # (b,h,c+1,l)
        decay_chunk = decay_chunk.unsqueeze(-1)                                 # (b,h,c+1,l,1)

        new_states = triton_matmul(states.reshape(-1, self.d_state).t(),
                                   decay_chunk.reshape(-1, 1).t())               # (b*h*(c+1)*l, d_state)
        new_states = new_states.t().reshape(self.batch_size, self.n_heads,
                                            self.seq_length // self.block_len + 1,
                                            self.block_len, self.d_state)

        return new_states[:, :, -1]  # (b,h,l,d_state)
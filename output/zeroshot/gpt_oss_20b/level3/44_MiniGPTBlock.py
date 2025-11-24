import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.jit
def softmax_kernel(
    a_ptr,          # input matrix
    out_ptr,        # output matrix
    rows, cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    col_end = min(col_start + BLOCK_SIZE, cols)

    # Load a row segment
    col_idx = tl.arange(0, BLOCK_SIZE) + col_start
    mask = col_idx < cols
    vals = tl.load(a_ptr + row * cols + col_idx, mask=mask, other=0.0)

    # compute max for numeric stability
    max_val = tl.max(vals, axis=0)
    vals = vals - max_val
    exp_vals = tl.exp(vals)

    # sum of exp
    exp_sum = tl.sum(exp_vals, axis=0)
    out = exp_vals / exp_sum

    # store
    tl.store(out_ptr + row * cols + col_idx, out, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of softmax over the last dimension.
    """
    assert x.is_cuda
    rows, cols = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (rows, (cols + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    softmax_kernel[grid](x, out, rows, cols, BLOCK_SIZE=256)
    return out


@triton.jit
def layernorm_kernel(
    inp_ptr,        # input
    out_ptr,        # output
    gamma_ptr,      # weight
    beta_ptr,       # bias
    n, dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    offset = idx * dim
    idxs = tl.arange(0, BLOCK_SIZE) + offset
    mask = idxs < offset + dim

    # load input
    inp = tl.load(inp_ptr + idxs, mask=mask, other=0.0)

    # mean
    mean = tl.sum(inp, axis=0) / dim

    # variance
    var = tl.sum((inp - mean) ** 2, axis=0) / dim

    # normalize
    norm = (inp - mean) * tl.rsqrt(var + eps)

    # apply gamma and beta
    gamma = tl.load(gamma_ptr + idxs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + idxs, mask=mask, other=0.0)
    out = gamma * norm + beta

    tl.store(out_ptr + idxs, out, mask=mask)


def triton_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps=1e-5):
    """
    Triton implementation of LayerNorm over the last dimension.
    """
    assert x.is_cuda
    batch, seq, dim = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (batch * seq,)
    layernorm_kernel[grid](
        x, out, weight, bias, batch * seq, dim, eps, BLOCK_SIZE=256
    )
    return out


# ---------- Model with Triton kernels ----------

class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) *
                                          (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.bias = nn.Parameter(
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
            requires_grad=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # scaled dot‑product
        att = torch.einsum("bnth,bnsh->bntt", q, k) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = triton_softmax(att)
        att = self.attn_dropout(att)
        y = torch.einsum("bntt,bnsh->bnth", att, v)
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd, bias=False),
                c_proj=nn.Linear(4 * n_embd, n_embd, bias=False),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        self.mlpf = lambda x: self.mlp.dropout(
            self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x)))
        )

    def forward(self, x):
        # LayerNorm via Triton
        x_ln1 = triton_layernorm(x, self.ln_1.weight, self.ln_1.bias, eps=self.ln_1.eps)
        x = x + self.attn(x_ln1)
        x_ln2 = triton_layernorm(x, self.ln_2.weight, self.ln_2.bias, eps=self.ln_2.eps)
        x = x + self.mlpf(x_ln2)
        return x
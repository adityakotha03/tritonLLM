import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# ---------- Triton matmul kernel ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                      num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)

    c = acc.to(tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def triton_matmul(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    return c


# ---------- Triton dropout kernel ----------
@triton.jit
def dropout_kernel(
    inp_ptr,
    out_ptr,
    mask_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    rng_state_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Random uniform numbers (xorshift128+)
    seed = tl.load(rng_state_ptr + pid)
    lo = (seed >> 32) & 0xffffffff
    hi = seed & 0xffffffff
    lo = (lo ^ (lo << 13)) & 0xffffffff
    hi = (hi ^ (hi << 17)) & 0xffffffff
    rand = (hi + lo) / 0xffffffff
    tl.store(rng_state_ptr + pid, (lo, hi))

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    mask_vals = rand >= dropout_p
    out = tl.where(mask_vals, inp / (1.0 - dropout_p), 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)
    tl.store(mask_ptr + offsets, mask_vals, mask=mask)


def triton_dropout(x: torch.Tensor, dropout_p: float, rng_state: torch.Tensor):
    assert x.is_cuda
    n = x.numel()
    out = torch.empty_like(x)
    mask = torch.empty(n, dtype=torch.bool, device=x.device)

    BLOCK_SIZE = 256
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    dropout_kernel[grid](
        x, out, mask, n, dropout_p, rng_state, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out, mask


# ---------- Triton softmax kernel ----------
@triton.jit
def softmax_kernel(
    inp_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_in,
    stride_out,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_ids = tl.arange(0, BLOCK_SIZE)

    if row_id >= n_rows:
        return

    row_offset = row_id * stride_in
    out_offset = row_id * stride_out

    mask = col_ids < n_cols
    vals = tl.load(inp_ptr + row_offset + col_ids, mask=mask, other=0.0)
    max_val = tl.max(vals, mask=mask)
    exp_vals = tl.exp(vals - max_val)
    sum_exp = tl.sum(exp_vals, mask=mask)
    probs = exp_vals / sum_exp
    tl.store(out_ptr + out_offset + col_ids, probs, mask=mask)


def triton_softmax(x: torch.Tensor, dim: int = 1):
    assert x.is_cuda
    assert dim == 1
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    softmax_kernel[grid](
        x, out, n_rows, n_cols, x.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------- Model with Triton kernels ----------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_p: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda"))
        self.bias = nn.Parameter(torch.randn(out_features, device="cuda"))
        self.dropout_p = dropout_p
        # Random state for dropout
        self.rng_state = torch.empty((torch.cuda.device_count(),), dtype=torch.uint64, device="cuda")
        self.rng_state.fill_(int(123456789))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matmul
        mat = triton_matmul(x, self.weight.t())
        mat = mat + self.bias
        # Dropout
        mat, _ = triton_dropout(mat, self.dropout_p, self.rng_state)
        # Softmax
        out = triton_softmax(mat, dim=1)
        return out
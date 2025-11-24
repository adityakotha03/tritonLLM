import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# ---------------------------------------------
# Triton kernel: Matmul (Q @ K.T) with BF16/Tensor Cores
# ---------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_qk_kernel(
    a_ptr,  # Q
    b_ptr,  # K
    c_ptr,  # output logits
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)).reshape(-1, 1) * stride_am
                  + (k + tl.arange(0, BLOCK_SIZE_K)).reshape(1, -1) * stride_an,
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + (col_start + tl.arange(0, BLOCK_SIZE_N)).reshape(-1, 1) * stride_bm
                  + (k + tl.arange(0, BLOCK_SIZE_K)).reshape(1, -1) * stride_bn,
            mask=(col_start + tl.arange(0, BLOCK_SIZE_N)[:, None] < N) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

    acc = acc.to(tl.float16)

    tl.store(
        c_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)).reshape(-1, 1) * stride_cm
              + (col_start + tl.arange(0, BLOCK_SIZE_N)).reshape(1, -1) * stride_cn,
        acc,
        mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
             (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < N),
    )

# ---------------------------------------------
# Triton kernel: Online softmax (logits -> attention probs)
# ---------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512}, num_warps=16),
    ],
    key=["M"],
)
@triton.jit
def softmax_kernel(
    logits_ptr,
    probs_ptr,
    M,
    stride_m,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE_M

    # Load block of logits
    offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    mask = offsets < M
    logits = tl.load(logits_ptr + offsets * stride_m, mask=mask, other=0.0)

    # Max for numerical stability
    max_logit = tl.max(logits, axis=0, mask=mask)
    logits -= max_logit

    # exp and sum
    exp_logits = tl.exp(logits)
    sum_exp = tl.sum(exp_logits, axis=0, mask=mask)

    # Normalize
    probs = exp_logits / sum_exp

    # Store
    tl.store(probs_ptr + offsets * stride_m, probs, mask=mask)

# ---------------------------------------------
# Triton kernel: Matmul (attention @ V)
# ---------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 32}, num_warps=16),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_attn_kernel(
    a_ptr,  # attention probs
    b_ptr,  # V
    c_ptr,  # output
    M, N, K,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(
            a_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)).reshape(-1, 1) * stride_am
                  + (k + tl.arange(0, BLOCK_SIZE_K)).reshape(1, -1) * stride_an,
            mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            b_ptr + (col_start + tl.arange(0, BLOCK_SIZE_N)).reshape(-1, 1) * stride_bm
                  + (k + tl.arange(0, BLOCK_SIZE_K)).reshape(1, -1) * stride_bn,
            mask=(col_start + tl.arange(0, BLOCK_SIZE_N)[:, None] < N) &
                 (k + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

    acc = acc.to(tl.float16)

    tl.store(
        c_ptr + (row_start + tl.arange(0, BLOCK_SIZE_M)).reshape(-1, 1) * stride_cm
              + (col_start + tl.arange(0, BLOCK_SIZE_N)).reshape(1, -1) * stride_cn,
        acc,
        mask=(row_start + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
             (col_start + tl.arange(0, BLOCK_SIZE_N)[None, :] < N),
    )

# ---------------------------------------------
# Helper wrappers
# ---------------------------------------------
def triton_matmul_qk(Q, K):
    B, H, S, D = Q.shape
    M = S
    N = S
    Kdim = D
    # Transpose K to shape (B, H, D, S)
    K_t = K.transpose(-1, -2)
    # Allocate output logits (B, H, S, S)
    logits = torch.empty((B, H, M, N), dtype=torch.float16, device=Q.device)
    grid = lambda meta: ( (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
                          (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"] )
    matmul_qk_kernel[grid](
        Q.contiguous().data_ptr(),
        K_t.contiguous().data_ptr(),
        logits.data_ptr(),
        M, N, Kdim,
        Q.stride(0), Q.stride(1),
        K_t.stride(0), K_t.stride(1),
        logits.stride(0), logits.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return logits

def triton_softmax(logits):
    B, H, S, _ = logits.shape
    probs = torch.empty_like(logits)
    grid = lambda meta: ( (S + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"], )
    softmax_kernel[grid](
        logits.contiguous().data_ptr(),
        probs.contiguous().data_ptr(),
        S,
        logits.stride(-1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
    )
    return probs

def triton_matmul_attn(attn, V):
    B, H, S, D = V.shape
    M = S
    N = D
    Kdim = S
    out = torch.empty((B, H, M, N), dtype=torch.float16, device=V.device)
    grid = lambda meta: ( (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
                          (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"] )
    matmul_attn_kernel[grid](
        attn.contiguous().data_ptr(),
        V.contiguous().data_ptr(),
        out.data_ptr(),
        M, N, Kdim,
        attn.stride(0), attn.stride(1),
        V.stride(0), V.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out

# ---------------------------------------------
# Optimized Model
# ---------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        logits = triton_matmul_qk(Q, K)                       # (B, H, S, S)
        probs = triton_softmax(logits)                       # (B, H, S, S)
        out = triton_matmul_attn(probs, V)                   # (B, H, S, D)
        return out
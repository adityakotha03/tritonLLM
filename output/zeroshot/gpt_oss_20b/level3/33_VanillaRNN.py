import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------- Triton kernels -------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_tanh_kernel(
    A_ptr, B_ptr, out_ptr, M, N, K,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_outm: tl.constexpr, stride_outn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(
            A_ptr + (row[:, None] * stride_am + k[None, :] * stride_ak),
            mask=(row[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            B_ptr + (k[:, None] * stride_bk + col[None, :] * stride_bn),
            mask=(k[:, None] < K) & (col[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(A, B)

    # Apply tanh fused
    acc = tl.math.tanh(acc)

    tl.store(
        out_ptr + (row[:, None] * stride_outm + col[None, :] * stride_outn),
        acc,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 512, "BLOCK_SIZE_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, out_ptr, M, N, K,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_outm: tl.constexpr, stride_outn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        A = tl.load(
            A_ptr + (row[:, None] * stride_am + k[None, :] * stride_ak),
            mask=(row[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        B = tl.load(
            B_ptr + (k[:, None] * stride_bk + col[None, :] * stride_bn),
            mask=(k[:, None] < K) & (col[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(A, B)

    tl.store(
        out_ptr + (row[:, None] * stride_outm + col[None, :] * stride_outn),
        acc,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


# ---------- Triton wrappers ------------------------------------------------

def triton_matmul_tanh(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A: (M, K), B: (K, N) -> (M, N) with fused tanh."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )
    matmul_tanh_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matmul A: (M, K), B: (K, N) -> (M, N)."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    out = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )
    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out


# ---------- Optimised model ------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Pre‑allocate weight tensors in the same shape as nn.Linear
        self.W = torch.randn((hidden_size, input_size + hidden_size), device='cuda', dtype=torch.float32)
        self.b = torch.randn((hidden_size,), device='cuda', dtype=torch.float32)

        self.W_out = torch.randn((output_size, hidden_size), device='cuda', dtype=torch.float32)
        self.b_out = torch.randn((output_size,), device='cuda', dtype=torch.float32)

        # Hidden state buffer
        self.register_buffer('hidden', torch.randn((batch_size, hidden_size), dtype=torch.float32))

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor | None = None) -> torch.Tensor:
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)

        # Concatenate input and hidden
        cat = torch.cat([x, self.hidden], dim=1)          # (B, input+hidden)
        # Linear -> tanh fused
        hidden = triton_matmul_tanh(cat, self.W.t()) + self.b
        self.hidden = hidden
        # Output linear
        out = triton_matmul(hidden, self.W_out.t()) + self.b_out
        return out
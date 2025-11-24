import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------- Triton kernels --------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_sigmoid_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    b_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_m[:, None] * K + k + tl.arange(0, BLOCK_K)[None, :], mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(B_ptr + k + tl.arange(0, BLOCK_K)[:, None] * N + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)

    acc = tl.sigmoid(acc)          # Apply sigmoid
    out = tl.dot(acc, B_ptr + K + tl.arange(0, BLOCK_K)[:, None] * N + offs_n[None, :])  # Fake second matmul
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=offs_m[:, None] < M, offs_n[None, :] < N)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def logsumexp_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=-1e9)
    max_val = tl.max(x, axis=0)
    exp_val = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    out = max_val + tl.log(sum_exp)
    tl.store(out_ptr, out, mask=mask)

# -------------------- Helper functions --------------------

def triton_matmul_sigmoid(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (
        (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )
    matmul_sigmoid_kernel[grid](A, B, C, M, N, K, BLOCK_M=meta["BLOCK_M"], BLOCK_N=meta["BLOCK_N"], BLOCK_K=meta["BLOCK_K"])
    return C

def triton_logsumexp(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    N = x.size(dim)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    logsumexp_kernel[grid](x, out, N, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out

# -------------------- Model definition --------------------

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty((input_size, hidden_size), device="cuda", dtype=torch.float32))
        self.bias1 = nn.Parameter(torch.empty((hidden_size,), device="cuda", dtype=torch.float32))
        self.weight2 = nn.Parameter(torch.empty((hidden_size, output_size), device="cuda", dtype=torch.float32))
        self.bias2 = nn.Parameter(torch.empty((output_size,), device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        nn.init.zeros_(self.bias1)
        nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        nn.init.zeros_(self.bias2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First matmul + sigmoid + second matmul
        h = triton_matmul_sigmoid(x, self.weight1) + self.bias1
        h = triton_matmul_sigmoid(h, self.weight2) + self.bias2
        # LogSumExp over features (dim=1)
        out = triton_logsumexp(h, dim=1)
        return out
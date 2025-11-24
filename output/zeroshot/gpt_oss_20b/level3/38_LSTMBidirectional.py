import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton implementation of a fused Linear layer (matrix multiplication + bias)
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 512}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def matmul_linear_kernel(
    X_ptr,   # (M, K)
    W_ptr,   # (N, K)
    B_ptr,   # (N)
    Y_ptr,   # (M, N)
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    mask_m = row_offsets < M
    mask_n = col_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_M):
        X_block = tl.load(X_ptr + row_offsets[:, None] * K + k + tl.arange(0, BLOCK_SIZE_M), mask=mask_m[:, None], other=0.0)
        W_block = tl.load(W_ptr + col_offsets[None, :] * K + k + tl.arange(0, BLOCK_SIZE_M), mask=mask_n[None, :], other=0.0)
        acc += tl.dot(X_block, W_block, transpose_a=True)

    if pid_n == 0:
        bias = tl.load(B_ptr + col_offsets, mask=mask_n, other=0.0)
        acc += bias[None, :]

    tl.store(Y_ptr + row_offsets[:, None] * N + col_offsets, acc, mask=mask_m[:, None] & mask_n[None, :])

def triton_linear(X: torch.Tensor, W: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = X.shape
    N = W.shape[0]
    Y = torch.empty((M, N), dtype=torch.float32, device=X.device)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_linear_kernel[grid](
        X,
        W.t(),
        B,
        Y,
        M, N, K,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
    )
    return Y

# -------------------------------------------------------------
# Optimized Model with Triton Linear
# -------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        # Linear weights and bias will be registered as parameters
        self.weight = nn.Parameter(torch.randn(output_size, hidden_size * 2, device="cuda"))
        self.bias = nn.Parameter(torch.randn(output_size, device="cuda"))

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        out, _ = self.lstm(x, (h0, c0))          # out: (B, T, 2*hidden)
        last = out[:, -1, :]                    # (B, 2*hidden)
        logits = triton_linear(last, self.weight, self.bias)   # (B, output_size)
        return logits
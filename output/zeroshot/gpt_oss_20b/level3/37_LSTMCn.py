import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel for a fused fully‑connected layer (matmul + bias)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr,          # (M, K)
    W_ptr,          # (N, K)
    B_ptr,          # (N,)
    Y_ptr,          # (M, N)
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    M_pad = (row_start + BLOCK_M <= M)
    N_pad = (col_start + BLOCK_N <= N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        X_tile = tl.load(
            X_ptr + (row_start[:, None] + k[None, :]),
            mask=(row_start[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )  # [BLOCK_M, BLOCK_K]
        W_tile = tl.load(
            W_ptr + (col_start[None, :] + k[:, None]),
            mask=(col_start[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(X_tile, W_tile)

    acc = acc + tl.load(B_ptr + col_start[None, :], mask=col_start[None, :] < N)

    # Store result
    tl.store(
        Y_ptr + (row_start[:, None] + col_start[None, :]),
        acc,
        mask=(row_start[:, None] < M) & (col_start[None, :] < N),
    )


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Apply a fully‑connected layer using a Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        (M + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
        (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
    )

    linear_kernel[grid](
        x, weight.t(), bias, out,
        M, N, K,
        BLOCK_M=meta['BLOCK_M'],
        BLOCK_N=meta['BLOCK_N'],
        BLOCK_K=meta['BLOCK_K'],
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.fc_weight = nn.Parameter(torch.randn(output_size, hidden_size, device='cuda'))
        self.fc_bias = nn.Parameter(torch.randn(output_size, device='cuda'))

    def forward(self, x, h0, c0):
        # LSTM forward
        out, (hn, cn) = self.lstm(x, (h0, c0))          # out: (B, L, H)

        # Take the last time step
        last_hidden = out[:, -1, :]                     # (B, H)

        # Custom linear layer with Triton
        logits = triton_linear(last_hidden, self.fc_weight, self.fc_bias)   # (B, O)

        return logits, (hn, cn)
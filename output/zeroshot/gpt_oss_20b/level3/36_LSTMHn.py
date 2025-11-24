import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Triton kernel for a fully‑connected layer
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_ctas=1),
        triton.Config({}, num_warps=2, num_ctas=2),
        triton.Config({}, num_warps=4, num_ctas=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr,        # pointer to input matrix (M x K)
    W_ptr,        # pointer to weight matrix (N x K)
    B_ptr,        # pointer to bias vector (N)
    Y_ptr,        # pointer to output matrix (M x N)
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
):
    """
    Matrix multiplication Y = X * W^T + B
    X : (M, K)
    W : (N, K)
    Y : (M, N)
    """
    # Compute the program id for the current block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Starting indices for the current block
    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    # Accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # Iterate over K dimension in tiles
    for k in range(0, K, BLOCK_SIZE_K):
        # Load a tile from X
        x_offsets = (
            row[:, None] * stride_xm
            + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_xm
            + k[None, :] * stride_xk
        )
        x = tl.load(X_ptr + x_offsets, mask=(row[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
                                     (k[None, :] + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
                    other=0.0)

        # Load a tile from W (transposed)
        w_offsets = (
            col[:, None] * stride_wn
            + k[None, :] * stride_wk
        )
        w = tl.load(W_ptr + w_offsets, mask=(col[:, None] + tl.arange(0, BLOCK_SIZE_N)[None, :] < N) &
                                             (k[None, :] + tl.arange(0, BLOCK_SIZE_K)[None, :] < K),
                    other=0.0)

        # Matrix multiply tile
        acc += tl.dot(x, w)

    # Add bias
    if pid_n == 0:
        bias = tl.load(B_ptr + col)
        acc += bias[None, :]

    # Store result
    y_offsets = (
        row[:, None] * stride_ym
        + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_ym
        + col[None, :] * stride_yn
    )
    tl.store(Y_ptr + y_offsets, acc, mask=(row[:, None] + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) &
                                         (col[None, :] + tl.arange(0, BLOCK_SIZE_N)[None, :] < N))


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper for the Triton linear kernel.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    assert x.shape[1] == weight.shape[1], "Dimension mismatch between input and weight."

    M, K = x.shape
    N = weight.shape[0]

    # Allocate output
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)

    # Kernel launch parameters
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 256

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        stride_xm=K,
        stride_xk=1,
        stride_wk=1,
        stride_wn=K,
        stride_ym=N,
        stride_yn=1,
    )
    return y


# -----------------------------
# Optimized model using the Triton linear kernel
# -----------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        # Re‑use the existing Linear weight and bias
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        # LSTM forward
        out, state = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden)
        # Use Triton linear kernel for the final projection
        last_hidden = out[:, -1, :]           # (batch, hidden)
        logits = triton_linear(last_hidden, self.fc.weight, self.fc.bias)
        return logits
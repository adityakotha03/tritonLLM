import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64},
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128},
                      num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_tanh_kernel(
    a_ptr,              # (batch, K)
    w_ptr,              # (N, K)  weights transposed
    bias_ptr,           # (N)
    out_ptr,            # (batch, N)
    batch: tl.constexpr,
    M: tl.constexpr,    # batch
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Kernel for:  out = tanh( a @ W.T + bias )
    a : (batch, K)
    W : (N, K)  (stored as N rows)
    bias : (N)
    out : (batch, N)
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    for k in range(0, K, BLOCK_SIZE_K):
        a_offsets = row_offsets[:, None] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        w_offsets = col_offsets[None, :] * K + tl.arange(0, BLOCK_SIZE_K)[None, :]

        a = tl.load(a_ptr + a_offsets, mask=row_offsets[:, None] < M, other=0.0)
        w = tl.load(w_ptr + w_offsets, mask=col_offsets[None, :] < N, other=0.0)

        acc += tl.dot(a, w)

    # add bias
    bias = tl.load(bias_ptr + col_offsets, mask=col_offsets < N, other=0.0)
    acc = acc + bias[None, :]

    # tanh activation
    acc = tl.math.tanh(acc)

    # store
    out_offsets = row_offsets[:, None] * N + col_offsets[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(row_offsets[:, None] < M) & (col_offsets[None, :] < N))


# --------------------------------------------------------------------------- #
# Helper function to launch the kernel
# --------------------------------------------------------------------------- #

def matmul_tanh(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    a: (batch, K)   fp16
    w: (N, K)       fp16 (weights already transposed)
    bias: (N,)      fp16
    Returns: (batch, N)
    """
    assert a.is_cuda and w.is_cuda and bias.is_cuda
    batch, K = a.shape
    N = w.shape[0]
    out = torch.empty((batch, N), dtype=torch.float16, device=a.device)

    grid = lambda meta: (
        (batch + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M'],
        (N + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N'],
    )

    matmul_tanh_kernel[grid](
        a,
        w,
        bias,
        out,
        batch=batch,
        M=batch,
        N=N,
        K=K,
    )
    return out


# --------------------------------------------------------------------------- #
# Optimised Model
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Linear layers stored as weight transposed for our kernel
        self.i2h_w = nn.Parameter(
            torch.randn(hidden_size, input_size + hidden_size, dtype=torch.float16, device='cuda')
        )
        self.i2h_b = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float16, device='cuda')
        )
        self.h2o_w = nn.Parameter(
            torch.randn(output_size, hidden_size, dtype=torch.float16, device='cuda')
        )
        self.h2o_b = nn.Parameter(
            torch.randn(output_size, dtype=torch.float16, device='cuda')
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch, input_size)
        h0: (batch, hidden_size)
        """
        seq_len, batch, _ = x.shape
        hidden = h0.to(x.dtype).contiguous()
        outputs = []

        # Pre-allocate concatenated tensor
        combined = torch.empty((batch, self.input_size + self.hidden_size),
                               dtype=torch.float16,
                               device=x.device)

        for t in range(seq_len):
            # Concatenate input and hidden
            combined.copy_(torch.cat((x[t].to(torch.float16), hidden), dim=1))

            # Hidden state update: tanh( W @ combined^T + b )
            hidden = matmul_tanh(combined, self.i2h_w, self.i2h_b)

            # Output linear: W @ hidden^T + b
            out = matmul_tanh(hidden, self.h2o_w, self.h2o_b)
            outputs.append(out)

        return torch.stack(outputs, dim=0).to(x.dtype)
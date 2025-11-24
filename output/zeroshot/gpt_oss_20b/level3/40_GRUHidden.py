import torch
import torch.nn as nn
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernel for batched matrix multiplication (B, M, K) @ (B, K, N)
# ------------------------------------------------------------------
@triton.jit
def batched_matmul_kernel(
    A_ptr,  # (B, M, K)
    B_ptr,  # (B, K, N)
    C_ptr,  # (B, M, N)
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCKS_PER_COL: tl.constexpr,
    BLOCKS_PER_K: tl.constexpr,
):
    # program_id gives the batch index and block index
    batch_id = tl.program_id(0)
    # compute global block indices
    block_m = tl.program_id(1)
    block_n = tl.program_id(2)

    # offsets into the matrices
    m_start = block_m * BLOCK_SIZE_M
    n_start = block_n * BLOCK_SIZE_N

    # allocate local registers for accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # iterate over K tiles
    for k in range(BLOCKS_PER_K):
        k_start = k * BLOCK_SIZE_K

        # load tile of A and B
        A_tile = tl.load(
            A_ptr + batch_id * M * K + (m_start * K) + (k_start),
            mask=(
                (tl.arange(0, BLOCK_SIZE_M) + m_start) < M
                & (tl.arange(0, BLOCK_SIZE_K) + k_start) < K
            ),
            other=0.0,
        )
        B_tile = tl.load(
            B_ptr + batch_id * K * N + (k_start * N) + (n_start),
            mask=(
                (tl.arange(0, BLOCK_SIZE_K) + k_start) < K
                & (tl.arange(0, BLOCK_SIZE_N) + n_start) < N
            ),
            other=0.0,
        )

        # matmul of tiles
        acc += tl.dot(A_tile, B_tile)

    # store the result
    tl.store(
        C_ptr + batch_id * M * N + (m_start * N) + (n_start),
        acc,
        mask=(
            (tl.arange(0, BLOCK_SIZE_M) + m_start) < M
            & (tl.arange(0, BLOCK_SIZE_N) + n_start) < N
        ),
    )


def batched_matmul(A: torch.Tensor, B: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton.
    A: (B, M, K), B: (B, K, N)
    Returns: (B, M, N)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    B_batch, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = torch.empty((B_batch, M, N), dtype=A.dtype, device=A.device)

    # compute grid dimensions
    BLOCK_SIZE_M = tl.constexpr(block_size)
    BLOCK_SIZE_N = tl.constexpr(block_size)
    BLOCK_SIZE_K = tl.constexpr(block_size)
    BLOCKS_PER_ROW = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    BLOCKS_PER_COL = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    BLOCKS_PER_K   = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    grid = lambda meta: (
        B_batch,
        BLOCKS_PER_ROW,
        BLOCKS_PER_COL,
    )

    batched_matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        B=B_batch,
        M=M,
        K=K,
        N=N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCKS_PER_ROW=BLOCKS_PER_ROW,
        BLOCKS_PER_COL=BLOCKS_PER_COL,
        BLOCKS_PER_K=BLOCKS_PER_K,
    )
    return C


# ------------------------------------------------------------------
# Custom GRU implementation using the Triton batched matmul
# ------------------------------------------------------------------
class TritonGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # weight matrices for all layers
        self.weight_ih = nn.Parameter(
            torch.randn(num_layers, 3 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(num_layers, 3 * hidden_size, hidden_size)
        )
        if bias:
            self.bias_ih = nn.Parameter(
                torch.randn(num_layers, 3 * hidden_size)
            )
            self.bias_hh = nn.Parameter(
                torch.randn(num_layers, 3 * hidden_size)
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        """
        x: (seq_len, batch, input_size)   or (batch, seq_len, input_size) if batch_first
        h0: (num_layers, batch, hidden_size)
        Returns: h_n (num_layers, batch, hidden_size)
        """
        seq_len, batch, _ = x.shape
        h = h0
        for layer in range(self.num_layers):
            weight_ih = self.weight_ih[layer]      # (3h, in)
            weight_hh = self.weight_hh[layer]      # (3h, h)
            bias_ih = self.bias_ih[layer] if self.bias else None
            bias_hh = self.bias_hh[layer] if self.bias else None

            h_next = torch.empty_like(h[layer])

            for t in range(seq_len):
                xt = x[t]                         # (batch, in)
                ht = h[layer]                     # (batch, h)

                # compute gates
                # combine linear ops into a single batched matmul
                # input part
                I = batched_matmul(xt.unsqueeze(1), weight_ih.unsqueeze(0))
                # hidden part
                H = batched_matmul(ht.unsqueeze(1), weight_hh.unsqueeze(0))
                # sum and add bias
                if self.bias:
                    I = I + bias_ih
                    H = H + bias_hh

                gate_inputs = I + H  # (batch, 3h)
                gate_inputs = gate_inputs.squeeze(1)  # (batch, 3h)

                z = torch.sigmoid(gate_inputs[:, : self.hidden_size])
                r = torch.sigmoid(gate_inputs[:, self.hidden_size : 2 * self.hidden_size])
                n = torch.tanh(
                    gate_inputs[:, 2 * self.hidden_size :]
                    + r * (batched_matmul(ht.unsqueeze(1), weight_hh[2 * self.hidden_size :, :].unsqueeze(0)).squeeze(1))
                )

                h_next[t] = (1 - z) * n + z * ht

            h[layer] = h_next

        return h


# ------------------------------------------------------------------
# Optimized model using TritonGRU
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.gru = TritonGRU(input_size, hidden_size, num_layers, bias)

    def forward(self, x, h0):
        # Input assumed to be (seq_len, batch, input_size)
        return self.gru(x, h0)
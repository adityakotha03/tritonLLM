import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_fwd_kernel(
    a_ptr,    # (M, K)
    b_ptr,    # (K, N)
    bias_ptr, # (N,)
    out_ptr,  # (M, N)
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :], mask=offs_k[:, None] < K, other=0.0)

        acc += tl.dot(a, b)

    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=offs_m[:, None] < M)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def sigmoid_fwd_kernel(out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = offs < N
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, out, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def tanh_fwd_kernel(out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    mask = offs < N
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    out = tl.tanh(x)
    tl.store(out_ptr + offs, out, mask=mask)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def matmul_bias_fwd(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    """
    a: (M, K)   torch.float32
    b: (K, N)   torch.float32
    bias: (N,)  torch.float32
    Returns: (M, N) torch.float32
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), dtype=torch.float32, device=a.device)

    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
    )

    matmul_bias_fwd_kernel[grid](
        a, b, bias, out,
        M, N, K,
        BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
    )
    return out

def sigmoid_fwd(x: torch.Tensor):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sigmoid_fwd_kernel[grid](out, N, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out

def tanh_fwd(x: torch.Tensor):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    tanh_fwd_kernel[grid](out, N, BLOCK_SIZE=meta["BLOCK_SIZE"])
    return out

# ------------------------------------------------------------------
# Model with custom GRU using Triton kernels
# ------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = True
        self.num_directions = 2

        # Weight tensors: (num_layers * num_directions, 3*hidden_size, input_size)
        self.weight_ih = nn.Parameter(
            torch.randn(num_layers * self.num_directions, 3 * hidden_size, input_size)
        )
        self.weight_hh = nn.Parameter(
            torch.randn(num_layers * self.num_directions, 3 * hidden_size, hidden_size)
        )
        if bias:
            self.bias_ih = nn.Parameter(
                torch.randn(num_layers * self.num_directions, 3 * hidden_size)
            )
            self.bias_hh = nn.Parameter(
                torch.randn(num_layers * self.num_directions, 3 * hidden_size)
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, x, h0):
        """
        x: (seq_len, batch, input) if batch_first=False
        h0: (num_layers*2, batch, hidden)
        """
        seq_len, batch, _ = x.shape if not self.batch_first else x.shape[1:3]
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, inp)

        hidden = []
        # Initialize hidden states per layer & direction
        h_t = h0
        for layer in range(self.num_layers):
            h_layer = []
            for direction in range(self.num_directions):
                offset = layer * self.num_directions + direction
                weight_ih = self.weight_ih[offset]
                weight_hh = self.weight_hh[offset]
                bias_ih = self.bias_ih[offset] if self.bias_ih is not None else torch.zeros(3 * self.hidden_size, device=x.device)
                bias_hh = self.bias_hh[offset] if self.bias_hh is not None else torch.zeros(3 * self.hidden_size, device=x.device)

                h_prev = h_t[layer * self.num_directions + direction]
                outputs = []
                for t in range(seq_len if direction == 0 else seq_len):
                    idx = t if direction == 0 else seq_len - 1 - t
                    x_t = x[idx]  # (batch, input)

                    # x_t * W_ih^T  +  h_prev * W_hh^T  +  bias
                    # Flatten batch for matmul
                    x_t_flat = x_t.reshape(batch, -1)  # (batch, input)
                    h_prev_flat = h_prev.reshape(batch, -1)  # (batch, hidden)

                    preact = matmul_bias_fwd(x_t_flat, weight_ih.T, bias_ih)
                    preact += matmul_bias_fwd(h_prev_flat, weight_hh.T, bias_hh)

                    # Split gates
                    r, z, n = preact.chunk(3, dim=1)
                    r = sigmoid_fwd(r)
                    z = sigmoid_fwd(z)
                    n = tanh_fwd(n)

                    h_new = (1 - z) * n + z * h_prev_flat
                    h_prev = h_new.reshape(batch, self.hidden_size)
                    outputs.append(h_prev)

                layer_output = torch.stack(outputs, dim=0)  # (seq, batch, hidden)
                h_layer.append(h_prev)
            hidden.append(torch.stack(h_layer, dim=0))  # (2, batch, hidden)

        # Concatenate directions
        output = torch.cat([h for h in hidden[-1]], dim=-1)  # (seq, batch, 2*hidden)
        h_n = torch.stack([h for h in hidden], dim=0)        # (layers*2, batch, hidden)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n
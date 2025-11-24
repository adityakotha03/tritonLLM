import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for fused matmul + bias
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    bias_ptr,  # (N,)
    c_ptr,  # (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * K + k + tl.arange(0, BLOCK_K), mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptr + (k + tl.arange(0, BLOCK_K)) * N + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias

    # Store result
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    x: (batch, in_features)
    weight: (out_features, in_features)
    bias: (out_features)
    Returns: (batch, out_features)
    """
    batch, in_features = x.shape
    out_features = weight.shape[0]
    assert weight.shape[1] == in_features

    # Transpose weight to match (in_features, out_features)
    weight_t = weight.t()

    out = torch.empty((batch, out_features), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(batch, meta["BLOCK_M"]),
                         triton.cdiv(out_features, meta["BLOCK_N"]))

    matmul_bias_kernel[grid](
        x,
        weight_t,
        bias,
        out,
        M=batch,
        N=out_features,
        K=in_features,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.weight_fc = nn.Parameter(
            torch.empty(output_size, hidden_size, device="cuda")
        )
        self.bias_fc = nn.Parameter(
            torch.empty(output_size, device="cuda")
        )
        nn.init.kaiming_uniform_(self.weight_fc, a=math.sqrt(5))
        nn.init.zeros_(self.bias_fc)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = triton_linear(out, self.weight_fc, self.bias_fc)
        return out
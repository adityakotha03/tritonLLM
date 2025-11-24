import torch
import torch.nn as nn
import triton
import triton.language as tl

# ---------------------------------
# Triton kernels
# ---------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_fused_relu_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    RELU: tl.constexpr,
):
    # program id grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am) + (k + tl.arange(0, BLOCK_K)[None, :] * stride_an),
            mask=(offs_m[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K)[:, None] * stride_bk) + (offs_n[None, :] * stride_bn),
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    # add bias if present
    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_cm,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        acc += bias

    # apply relu
    if RELU:
        acc = tl.maximum(acc, 0.0)

    c = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(
        c,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_fused_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am) + (k + tl.arange(0, BLOCK_K)[None, :] * stride_an),
            mask=(offs_m[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K)[:, None] * stride_bk) + (offs_n[None, :] * stride_bn),
            mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)

    # add bias if present
    if bias_ptr is not None:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_cm,
            mask=offs_m[:, None] < M,
            other=0.0,
        )
        acc += bias

    c = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    tl.store(
        c,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

# ---------------------------------
# Helper functions
# ---------------------------------
def triton_linear_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Perform x @ weight.T + bias and apply ReLU in one fused kernel."""
    batch, in_features = x.shape
    out_features = weight.shape[0]
    result = torch.empty((batch, out_features), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (batch + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (out_features + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_fused_relu_kernel[grid](
        x,
        weight,
        result,
        bias,
        batch,
        out_features,
        in_features,
        1, 1,  # stride_am, stride_an (row‑major)
        1, 1,  # stride_bk, stride_bn
        1, 1,  # stride_cm, stride_cn
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
        RELU=True,
    )
    return result

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Perform x @ weight.T + bias in one fused kernel."""
    batch, in_features = x.shape
    out_features = weight.shape[0]
    result = torch.empty((batch, out_features), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        (batch + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
        (out_features + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],
    )

    matmul_fused_kernel[grid](
        x,
        weight,
        result,
        bias,
        batch,
        out_features,
        in_features,
        1, 1,
        1, 1,
        1, 1,
        BLOCK_M=meta["BLOCK_M"],
        BLOCK_N=meta["BLOCK_N"],
        BLOCK_K=meta["BLOCK_K"],
    )
    return result

# ---------------------------------
# Model definition
# ---------------------------------
class LinearFusion(nn.Module):
    def __init__(self, in_features, out_features, bias=True, relu=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device='cuda'))
        else:
            self.register_parameter('bias', None)
        self.relu = relu
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.relu:
            return triton_linear_relu(x, self.weight, self.bias)
        else:
            return triton_linear(x, self.weight, self.bias)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        in_size = input_size
        for out_size in hidden_layer_sizes:
            layers.append(LinearFusion(in_size, out_size, bias=True, relu=True))
            in_size = out_size
        layers.append(LinearFusion(in_size, output_size, bias=True, relu=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Triton kernels -----------------------------------------

# Matrix multiplication + bias + ReLU (fused)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                      num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_bias_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Allocate accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Iterate over K tiles
    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B
        a = tl.load(A_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_am
                    + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
                    mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M,
                    other=0.0)

        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk
                    + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
                    mask=(k + tl.arange(0, BLOCK_K))[:, None] < K,
                    other=0.0)

        acc += tl.dot(a, b)

    # Add bias and apply ReLU
    bias = tl.load(bias_ptr + col_start + tl.arange(0, BLOCK_N),
                   mask=col_start + tl.arange(0, BLOCK_N) < N,
                   other=0.0)

    acc = acc + bias
    acc = tl.maximum(acc, 0.0)

    # Store result
    tl.store(C_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_cm
                    + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
             acc,
             mask=((row_start + tl.arange(0, BLOCK_M))[:, None] < M) &
                  ((col_start + tl.arange(0, BLOCK_N))[None, :] < N))


# Simple matrix multiplication (used for the final linear layer without bias)
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                      num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
                      num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_am
                    + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
                    mask=(row_start + tl.arange(0, BLOCK_M))[:, None] < M,
                    other=0.0)

        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk
                    + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
                    mask=(k + tl.arange(0, BLOCK_K))[:, None] < K,
                    other=0.0)

        acc += tl.dot(a, b)

    tl.store(C_ptr + (row_start + tl.arange(0, BLOCK_M))[:, None] * stride_cm
                    + (col_start + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
             acc,
             mask=((row_start + tl.arange(0, BLOCK_M))[:, None] < M) &
                  ((col_start + tl.arange(0, BLOCK_N))[None, :] < N))


# Helper wrappers ----------------------------------------------------

def triton_matmul_bias_relu(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor):
    """A @ B + bias, ReLU fused. All tensors are fp16."""
    A = A.to(torch.float16)
    B = B.to(torch.float16)
    bias = bias.to(torch.float16)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out = torch.empty((M, N), dtype=torch.float16, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))

    matmul_bias_relu_kernel[grid](
        A, B, bias, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    """A @ B. All tensors are fp16."""
    A = A.to(torch.float16)
    B = B.to(torch.float16)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    out = torch.empty((M, N), dtype=torch.float16, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
                         triton.cdiv(N, meta["BLOCK_N"]))

    matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


# ---------- Optimized model -----------------------------------------

class LinearTRT(nn.Module):
    """Linear layer with Triton kernel (matmul + bias)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device="cuda"))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device="cuda"))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (batch, in_features)
        # weight: (out, in)
        # bias: (out,)
        output = triton_matmul_bias_relu(input, self.weight, self.bias)
        return output


class ReLUNew(nn.Module):
    """ReLU implemented with Triton (elementwise)."""

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        ],
        key=["N"],
    )
    @triton.jit
    def relu_kernel(
        out_ptr, in_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        out_vals = tl.maximum(in_vals, 0.0)
        tl.store(out_ptr + offsets, out_vals, mask=mask)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(input)
        n = input.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        self.relu_kernel[grid](out, input, n, BLOCK_SIZE=128)
        return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, layer_sizes: list[int], output_size: int):
        super().__init__()
        layers = []
        cur_in = input_size
        for out_dim in layer_sizes:
            layers.append(LinearTRT(cur_in, out_dim))
            layers.append(ReLUNew())
            cur_in = out_dim
        layers.append(LinearTRT(cur_in, output_size, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
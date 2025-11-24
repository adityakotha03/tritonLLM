import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------
# Triton kernel: matrix multiply followed by ReLU
# --------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_relu_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_a_m: tl.constexpr,
    stride_a_k: tl.constexpr,
    stride_b_k: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_c_n: tl.constexpr,
):
    """
    Compute C = max(0, A @ B) with tiling for Tensor Cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    block_start_m = pid_m * BLOCK_M
    block_start_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles
        A_tile = tl.load(
            A_ptr + (block_start_m + tl.arange(0, BLOCK_M))[:, None] * stride_a_m
            + (k + tl.arange(0, BLOCK_K))[None, :] * stride_a_k,
            mask=(block_start_m + tl.arange(0, BLOCK_M))[:, None] < M
            & (k + tl.arange(0, BLOCK_K))[None, :] < K,
            other=0.0,
        )
        B_tile = tl.load(
            B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_b_k
            + (block_start_n + tl.arange(0, BLOCK_N))[None, :] * stride_b_n,
            mask=(k + tl.arange(0, BLOCK_K))[:, None] < K
            & (block_start_n + tl.arange(0, BLOCK_N))[None, :] < N,
            other=0.0,
        )
        # Matmul
        acc += tl.dot(A_tile, B_tile)

    # ReLU
    acc = tl.where(acc > 0, acc, 0.0)

    # Store result
    tl.store(
        C_ptr + (block_start_m + tl.arange(0, BLOCK_M))[:, None] * stride_c_m
        + (block_start_n + tl.arange(0, BLOCK_N))[None, :] * stride_c_n,
        acc,
        mask=(block_start_m + tl.arange(0, BLOCK_M))[:, None] < M
        & (block_start_n + tl.arange(0, BLOCK_N))[None, :] < N,
    )


# --------------------------------------------------
# Python wrapper for the fused matmul + ReLU
# --------------------------------------------------
def triton_fused_linear_relu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    x: (batch, in_features)
    weight: (out_features, in_features)
    bias: (out_features,) or None
    Returns: (batch, out_features) with ReLU applied.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    M, K = x.shape[0], x.shape[1]
    N = weight.shape[0]

    # Convert to fp16 for tensor cores (use bf16 if supported)
    x_fp16 = x.to(torch.float16)
    weight_fp16 = weight.t().to(torch.float16)  # weight is (out, in) -> (in, out)
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    grid = lambda meta: (
        ( (M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"],
          (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"] ),
    )

    matmul_relu_kernel[grid](
        x_fp16,
        weight_fp16,
        out,
        M, N, K,
        BLOCK_M=triton.next_power_of_2(M),
        BLOCK_N=triton.next_power_of_2(N),
        BLOCK_K=triton.next_power_of_2(K),
        stride_a_m=x_fp16.stride(0),
        stride_a_k=x_fp16.stride(1),
        stride_b_k=weight_fp16.stride(0),
        stride_b_n=weight_fp16.stride(1),
        stride_c_m=out.stride(0),
        stride_c_n=out.stride(1),
    )

    out = out.to(torch.float32)
    if bias is not None:
        out += bias.to(torch.float32)

    return out


# --------------------------------------------------
# Optimized model using the custom Triton kernel
# --------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        # Pre‑allocate weight and bias buffers
        layers = []
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        current_in = input_size
        for hidden_size in hidden_layer_sizes:
            w = nn.Parameter(torch.empty(hidden_size, current_in, device="cuda"))
            b = nn.Parameter(torch.empty(hidden_size, device="cuda"))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
            self.weights.append(w)
            self.biases.append(b)
            current_in = hidden_size

        # Output layer
        w_out = nn.Parameter(torch.empty(output_size, current_in, device="cuda"))
        b_out = nn.Parameter(torch.empty(output_size, device="cuda"))
        nn.init.kaiming_uniform_(w_out, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w_out)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(b_out, -bound, bound)
        self.weights.append(w_out)
        self.biases.append(b_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            out = triton_fused_linear_relu(out, w, b)
        # Final linear without ReLU
        out = torch.nn.functional.linear(out, self.weights[-1], self.biases[-1])
        return out
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gemm_hardtanh_gelu_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    scaling_factor,
    hardtanh_min, hardtanh_max,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to blocks of X and W
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulate in registers
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matrix multiplication loop with BLOCK_K tiles
    for k in range(0, K, BLOCK_K):
        # Load X and W blocks
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # Use Tensor Cores via dot product
        accumulator = tl.dot(x, w, acc=accumulator, out_dtype=tl.float32, allow_tf32=USE_TF32)

        # Update pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias if present
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n * stride_wn
        bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    # Apply scaling
    accumulator *= scaling_factor

    # Apply Hardtanh: clamp between hardtanh_min and hardtanh_max
    accumulator = tl.maximum(accumulator, hardtanh_min)
    accumulator = tl.minimum(accumulator, hardtanh_max)

    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Constants
    PI = 3.141592653589793
    SQRT_2_OVER_PI = tl.sqrt(2 / PI)
    GELU_COEF = 0.044715

    x_gelu = accumulator
    x_cubed = x_gelu * x_gelu * x_gelu
    inner = SQRT_2_OVER_PI * (x_gelu + GELU_COEF * x_cubed)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * x_gelu * (1.0 + tanh_inner)

    # Write output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, gelu_out, mask=mask)


def triton_fused_gemm_hardtanh_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
    hardtanh_min: float,
    hardtanh_max: float,
):
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    M, K = x.shape
    K, N = weight.shape

    # Output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    USE_TF32 = torch.backends.cuda.matmul.allow_tf32 and weight.dtype == torch.float32
    HAS_BIAS = bias is not None

    # Grid configuration
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    fused_gemm_hardtanh_gelu_kernel[grid](
        x, weight, bias, out,
        scaling_factor,
        hardtanh_min, hardtanh_max,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        HAS_BIAS=HAS_BIAS,
        USE_TF32=USE_TF32,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel for GEMM + scaling + hardtanh + GELU.
    """
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return triton_fused_gemm_hardtanh_gelu(
            x, self.weight, self.bias,
            self.scaling_factor,
            self.hardtanh_min,
            self.hardtanh_max
        )
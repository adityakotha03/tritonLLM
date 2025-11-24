import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernel: fused matrix multiplication, division and GELU
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "UNROLL": 4}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 256, "UNROLL": 4}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_div_gelu_kernel(
    X_ptr,        # input matrix (M x K)
    W_ptr,        # weight matrix (N x K)   (stored as K x N for efficient column access)
    out_ptr,      # output matrix (M x N)
    M, N, K,
    divisor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Allocate register accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Load tiles of X (BLOCK_M x BLOCK_K)
        X_offsets = start_m[:, None] * K + (k + tl.arange(0, BLOCK_K)[None, :])
        X_tile = tl.load(X_ptr + X_offsets, mask=(start_m[:, None] < M) & (k + tl.arange(0, BLOCK_K)[None, :] < K), other=0.0)

        # Load tiles of W (BLOCK_N x BLOCK_K) but stored as K x N, so transpose during load
        W_offsets = (k + tl.arange(0, BLOCK_K)[:, None]) * N + start_n[None, :]
        W_tile = tl.load(W_ptr + W_offsets, mask=(k + tl.arange(0, BLOCK_K)[:, None] < K) & (start_n[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(X_tile, W_tile)

    # Apply division
    acc = acc / divisor

    # Apply GELU (fast approximation)
    # gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * x_cubed)
    gelu = 0.5 * acc * (1 + tl.tanh(inner))

    # Store
    out_offsets = start_m[:, None] * N + (start_n[None, :])
    tl.store(out_ptr + out_offsets, gelu, mask=(start_m[:, None] < M) & (start_n[None, :] < N))


# --------------------------------------------------------------------
# Wrapper for the Triton kernel
# --------------------------------------------------------------------
def matmul_div_gelu_torch(X: torch.Tensor, W: torch.Tensor, divisor: float) -> torch.Tensor:
    """
    X: (batch, K)   -> float32
    W: (out, K)     -> float32   (weights of Linear layer)
    divisor: float
    Returns:
        Y: (batch, out) -> float32
    """
    assert X.is_cuda and W.is_cuda
    assert X.dtype == torch.float32 and W.dtype == torch.float32

    M, K = X.shape
    out, N = W.shape

    # Allocate output
    Y = torch.empty((M, out), dtype=torch.float32, device=X.device)

    # Triton grid: (M/BLOCK_M, N/BLOCK_N)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(out, meta["BLOCK_N"]))

    # Launch kernel
    matmul_div_gelu_kernel[grid](
        X, W.T, Y, M, out, K,
        divisor=divisor,
        BLOCK_M=meta := 128,  # default, autotuner will override
        BLOCK_N=meta,
        BLOCK_K=meta,
        UNROLL=4,
    )
    return Y


# --------------------------------------------------------------------
# New model using Triton fused kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    A model that performs a matrix multiplication, divides by a scalar,
    and applies GELU activation using a fused Triton kernel.
    """
    def __init__(self, input_size: int, output_size: int, divisor: float):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.divisor = divisor
        # Linear weight without bias
        self.weight = nn.Parameter(torch.empty(output_size, input_size, device="cuda", dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return matmul_div_gelu_torch(x, self.weight, self.divisor)
import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------- Triton kernels --------------------------------

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=['N', 'K', 'M'],
)
@triton.jit
def matmul_sigmoid_scale_residual_kernel(
    a_ptr,          # [B, N] input
    w_ptr,          # [N, M] weight
    b_ptr,          # [M] bias
    out_ptr,        # [B, M] output
    N, M, K,        # dimensions: K=N
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SCALING_FACTOR: tl.constexpr,
    BATCH: tl.constexpr,
):
    """
    For each batch element, compute:
        y = A @ W + b
        y = sigmoid(y) * SCALING_FACTOR + y   (residual)
    """
    batch_id = tl.program_id(0)
    if batch_id >= BATCH:
        return

    # Compute row of output
    row = tl.arange(0, BLOCK_SIZE_M)

    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over tiles of K
    for k in range(0, K, BLOCK_SIZE_N):
        a_tile = tl.load(a_ptr + batch_id * N + k + tl.arange(0, BLOCK_SIZE_N),
                         mask=(k + tl.arange(0, BLOCK_SIZE_N) < N),
                         other=0.0, dtype=tl.float16)
        w_tile = tl.load(w_ptr + (k + tl.arange(0, BLOCK_SIZE_N))[:, None] + row[None, :],
                         mask=(k + tl.arange(0, BLOCK_SIZE_N)[:, None] < K) & (row[None, :] < M),
                         other=0.0, dtype=tl.float16)

        acc += tl.dot(a_tile, w_tile, allow_tf32=False)

    # Add bias
    bias = tl.load(b_ptr + row, mask=row < M, other=0.0, dtype=tl.float32)
    acc += bias

    # Sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Scale
    acc *= SCALING_FACTOR

    # Residual add (original pre‑activation)
    # We need the raw matmul output again; recompute it
    raw = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_N):
        a_tile = tl.load(a_ptr + batch_id * N + k + tl.arange(0, BLOCK_SIZE_N),
                         mask=(k + tl.arange(0, BLOCK_SIZE_N) < N),
                         other=0.0, dtype=tl.float16)
        w_tile = tl.load(w_ptr + (k + tl.arange(0, BLOCK_SIZE_N))[:, None] + row[None, :],
                         mask=(k + tl.arange(0, BLOCK_SIZE_N)[:, None] < K) & (row[None, :] < M),
                         other=0.0, dtype=tl.float16)
        raw += tl.dot(a_tile, w_tile, allow_tf32=False)
    raw += bias
    acc += raw

    # Store result
    tl.store(out_ptr + batch_id * M + row, acc, mask=row < M)

# --------------------------- Python helpers --------------------------------

def triton_matmul_sigmoid_scale_residual(
    a: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    scaling_factor: float,
):
    """
    Wrapper that launches the fused Triton kernel.
    """
    assert a.is_cuda and w.is_cuda and b.is_cuda
    batch, N = a.shape
    K, M = w.shape
    assert N == K, "Input dimension must match weight's first dimension"

    # Prepare output
    out = torch.empty((batch, M), dtype=a.dtype, device=a.device)

    # Choose block sizes (power of 2)
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128

    # Grid: one block per batch element
    grid = lambda meta: (batch,)

    # Launch kernel
    matmul_sigmoid_scale_residual_kernel[grid](
        a, w, b, out,
        N, M, K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        SCALING_FACTOR=scaling_factor,
        BATCH=batch,
    )
    return out

# --------------------------- New model -------------------------------------

class ModelNew(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd" using fused Triton kernel.
    """
    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        # Store weight and bias as parameters
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size, device='cuda', dtype=torch.float16))
        self.bias   = nn.Parameter(torch.empty(hidden_size, device='cuda', dtype=torch.float32))
        # Initialise like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the fused Triton kernel.
        """
        # Ensure input dtype matches weight dtype
        x = x.to(dtype=self.weight.dtype)
        return triton_matmul_sigmoid_scale_residual(
            x, self.weight.t(), self.bias, self.scaling_factor
        )
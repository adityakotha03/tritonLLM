import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Triton kernel that fuses:  matmul, divide by 2, sum over hidden_dim, 
# and final scaling.  It processes one batch element per program.
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["M", "K", "N"],
)
@triton.jit
def _gemm_fused_kernel(
    X_ptr,   # (batch, K)
    W_ptr,   # (N, K)
    out_ptr, # (batch, 1) result after sum and scaling
    M, K, N,
    scaling_factor,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per batch element (row of X)
    row = pid
    # Allocate registers for accumulating the dot products
    acc = tl.zeros([N], dtype=tl.float32)

    # Loop over K in tiles
    for k in range(0, K, BLOCK_K):
        k_offset = k + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # Load a tile of X (row, k)
        x = tl.load(X_ptr + row * K + k_offset, mask=k_mask, other=0.0)

        # Load a tile of W (N, k)
        # Broadcast k_offset across N
        w = tl.load(W_ptr + k_offset[:, None] * N + tl.arange(0, N), 
                    mask=k_mask[:, None], other=0.0)

        # Compute partial dot products
        acc += x[:, None] * w  # (N,)

    # After summing over K, compute sum over N
    # Use a warp-level reduction
    # We only need the sum of acc
    sum_val = tl.sum(acc)

    # Divide by 2 and scale
    sum_val = (sum_val / 2.0) * scaling_factor

    # Store result in output tensor of shape (batch, 1)
    tl.store(out_ptr + row, sum_val)

# --------------------------------------------------------------------
# Wrapper that launches the kernel
# --------------------------------------------------------------------
def fused_gemm(x: torch.Tensor, w: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    x: (batch, K)
    w: (N, K)  (weights transposed relative to normal matmul)
    Returns: (batch, 1)
    """
    assert x.is_cuda and w.is_cuda
    batch, K = x.shape
    N, _ = w.shape

    out = torch.empty((batch, 1), device=x.device, dtype=x.dtype)

    # Choose grid size: one program per batch element
    grid = lambda meta: (batch,)

    _gemm_fused_kernel[grid](
        x, w, out,
        batch, K, N,
        scaling_factor,
        BLOCK_M=128,
        BLOCK_K=256,
    )
    return out

# --------------------------------------------------------------------
# Optimized model using the custom Triton kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling
    using a fused Triton kernel for maximum performance on NVIDIA A100.
    """
    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        # We keep weight in shape (hidden_size, input_size)
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size, device="cuda", dtype=torch.float32))
        self.scaling_factor = scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # weight is (hidden_size, input_size) -> we need (N, K) where N=hidden_size, K=input_size
        return fused_gemm(x, self.weight, self.scaling_factor)
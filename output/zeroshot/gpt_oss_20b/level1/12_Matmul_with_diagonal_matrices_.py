import torch
import torch.nn as nn
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Triton kernel: element‑wise scaling of each row of B by A[i]
# --------------------------------------------------------------------------
@triton.jit
def diag_scale_kernel(
    A_ptr,      # Pointer to 1‑D tensor A (diagonal values)
    B_ptr,      # Pointer to 2‑D tensor B (contiguous, shape (N, M))
    out_ptr,    # Pointer to output tensor C
    N: tl.constexpr,   # Number of rows
    M: tl.constexpr,   # Number of columns
    BLOCK_SIZE: tl.constexpr,   # Number of elements processed per program
):
    # Each program handles a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * M

    # Load B value
    b_val = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # Compute row index for each offset
    row = offsets // M
    a_val = tl.load(A_ptr + row, mask=mask, other=0.0)

    # Scale and store
    out = a_val * b_val
    tl.store(out_ptr + offsets, out, mask=mask)


# --------------------------------------------------------------------------
# Helper wrapper to launch the Triton kernel
# --------------------------------------------------------------------------
def diag_scale(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes diag(A) @ B efficiently using Triton.

    Args:
        A: 1‑D tensor of shape (N,).
        B: 2‑D tensor of shape (N, M).

    Returns:
        Tensor of shape (N, M) containing the product.
    """
    assert A.is_cuda and B.is_cuda, "Both tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    N = A.shape[0]
    M = B.shape[1]
    out = torch.empty((N, M), dtype=A.dtype, device=A.device)

    # Choose a reasonable BLOCK_SIZE; this can be autotuned if desired.
    BLOCK_SIZE = 256

    # Compute grid size (number of programs)
    grid = lambda meta: ((N * M + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    diag_scale_kernel[grid](A, B, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return out


# --------------------------------------------------------------------------
# Optimized model using the Triton kernel
# --------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs diag(A) @ B using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return diag_scale(A, B)
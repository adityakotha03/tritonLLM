import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_mul_kernel(
    A_ptr,      # Pointer to the 1D diagonal vector A
    B_ptr,      # Pointer to the 2D matrix B
    C_ptr,      # Pointer to output matrix C
    N,          # Size of the diagonal (N)
    M,          # Number of columns in B (M)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each block handles a tile of (BLOCK_SIZE_N, BLOCK_SIZE_M)
    pid_n = tl.program_id(0)  # Block ID along N axis
    pid_m = tl.program_id(1)  # Block ID along M axis

    # Calculate starting offsets for this tile
    block_start_n = pid_n * BLOCK_SIZE_N
    block_start_m = pid_m * BLOCK_SIZE_M

    # Create row and column indices for the current tile
    offs_n = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_m = block_start_m + tl.arange(0, BLOCK_SIZE_M)

    # Create masks to handle boundary conditions
    mask_n = offs_n < N
    mask_m = offs_m < M
    mask = mask_n[:, None] & mask_m[None, :]  # Broadcast to (BLOCK_SIZE_N, BLOCK_SIZE_M)

    # Load diagonal values A[i] for the current block of rows
    # Only load values where the row index is within bounds
    A_vals = tl.load(A_ptr + offs_n, mask=mask_n, other=0.0)  # Shape: (BLOCK_SIZE_N,)

    # Load B values for the current tile
    B_vals = tl.load(B_ptr + (offs_n[:, None] * M + offs_m[None, :]), mask=mask, other=0.0)  # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_M)

    # Perform element-wise multiplication: C[i, j] = A[i] * B[i, j]
    C_vals = A_vals[:, None] * B_vals  # Broadcasting A across columns

    # Store the result
    tl.store(C_ptr + (offs_n[:, None] * M + offs_m[None, :]), C_vals, mask=mask)


def triton_diag_mul(A, B):
    """
    Performs C = diag(A) @ B using a custom Triton kernel.
    
    Args:
        A (torch.Tensor): 1D tensor of shape (N,)
        B (torch.Tensor): 2D tensor of shape (N, M)

    Returns:
        torch.Tensor: Output tensor of shape (N, M)
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA"
    assert A.ndim == 1 and B.ndim == 2, "Expected A to be 1D, B to be 2D"
    assert A.shape[0] == B.shape[0], "A and B must have same first dimension"

    N, M = B.shape
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor
    C = torch.empty(N, M, device=A.device, dtype=A.dtype)

    # Define block sizes
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_M = 128

    # Grid configuration: number of blocks along N and M dimensions
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))

    # Launch the Triton kernel
    diag_mul_kernel[grid](A, B, C, N, M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_M=BLOCK_SIZE_M)

    return C


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # Use the custom Triton kernel for efficient diag(A) @ B
        return triton_diag_mul(A, B)
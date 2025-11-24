import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def diag_matmul_kernel(
    A_ptr,  # Pointer to diagonal elements
    B_ptr,  # Pointer to matrix B
    out_ptr,  # Pointer to output matrix
    N: tl.constexpr,  # Number of rows/columns in diagonal matrix
    M: tl.constexpr,  # Number of columns in matrix B
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the block start index
    block_start = pid * BLOCK_SIZE
    # Compute the offset for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N

    # Load diagonal elements
    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    # Load matrix B
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    # Compute the product
    out = a * b
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def diag_matmul(A: torch.Tensor, B: torch.Tensor):
    """
    Custom Triton kernel for diagonal matrix multiplication with matrix B.
    """
    assert A.is_cuda and B.is_cuda, "Tensors must be on CUDA."
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor
    out = torch.empty((N, M), device=A.device, dtype=A.dtype)

    # Determine block size
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine number of blocks
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    diag_matmul_kernel[ (num_blocks, ) ](A, B, out, N, M, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        return diag_matmul(A, B)
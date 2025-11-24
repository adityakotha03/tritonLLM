import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def mul_scalar_kernel(
    x_ptr,        # Pointer to input matrix
    scalar,       # Scalar value to multiply
    out_ptr,      # Pointer to output matrix
    n_elements,   # Total number of elements in the matrix
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * scalar
    tl.store(out_ptr + offsets, output, mask=mask)


def triton_mul_scalar(x: torch.Tensor, scalar: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Use a block size that is a power of 2 and covers typical warp behavior
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mul_scalar_kernel[grid](x, scalar, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return triton_mul_scalar(A, s)
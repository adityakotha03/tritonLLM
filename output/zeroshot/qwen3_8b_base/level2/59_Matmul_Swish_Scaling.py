import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_swish_kernel(
    a_ptr,  # pointer to matrix A
    b_ptr,  # pointer to matrix B
    out_ptr,  # pointer to output matrix
    m, n, k,  # dimensions of matrices
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_m = pid * BLOCK_SIZE
    block_n = 0

    # Compute the block's row and column indices
    row_offsets = block_m + tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load matrix A and B
    a = tl.load(a_ptr + row_offsets[:, None] * k + col_offsets[None, :], mask=(row_offsets < m)[:, None] * (col_offsets < k)[None, :], other=0.0)
    b = tl.load(b_ptr + col_offsets[:, None] * n + row_offsets[None, :], mask=(col_offsets < n)[:, None] * (row_offsets < k)[None, :], other=0.0)

    # Compute the matrix multiplication
    c = tl.dot(a, b)

    # Apply Swish activation
    c = c * tl.sigmoid(c)

    # Store the result
    tl.store(out_ptr + row_offsets[:, None] * n + col_offsets[None, :], c, mask=(row_offsets < m)[:, None] * (col_offsets < n)[None, :])


def triton_matmul_swish(a: torch.Tensor, b: torch.Tensor, scaling_factor: float):
    """
    Triton kernel for matrix multiplication followed by Swish activation and scaling.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Output tensor
    out = torch.empty((a.size(0), b.size(1)), device=a.device, dtype=a.dtype)

    # Dimensions
    m, k = a.shape
    _, n = b.shape

    # Choose block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    matmul_swish_kernel[grid](a, b, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    return out * scaling_factor


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Perform matrix multiplication and Swish activation with Triton kernel
        x = triton_matmul_swish(x, torch.randn((in_features, out_features), device=x.device, dtype=x.dtype), self.scaling_factor)
        return x
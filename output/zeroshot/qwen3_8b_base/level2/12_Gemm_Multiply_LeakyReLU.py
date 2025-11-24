import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_relu_kernel(
    a_ptr,  # Pointer to first input (weight matrix)
    b_ptr,  # Pointer to second input (input tensor)
    out_ptr,  # Pointer to output
    n_rows,  # Number of rows in output
    n_cols,  # Number of columns in output
    n_elements_per_row,  # Number of elements per row in input
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the row index of the current program
    row_idx = tl.program_id(0)
    # Compute the offset for the current row
    row_start = row_idx * n_elements_per_row
    # Create a range of column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Compute the offset for the current block within the row
    block_start = row_start + col_offsets * n_elements_per_row
    # Mask to ensure we don't go out of bounds
    mask = block_start < n_rows * n_cols

    # Load weights
    a = tl.load(a_ptr + block_start, mask=mask, other=0.0)
    # Load input
    b = tl.load(b_ptr + block_start, mask=mask, other=0.0)
    # Compute the dot product
    dot = tl.dot(a, b)
    # Apply multiplier
    dot = dot * tl.load(tl.math._get_default_float_ptr(), 0.0)  # Placeholder for multiplier
    # Apply LeakyReLU
    dot = tl.where(dot > 0, dot, dot * tl.load(tl.math._get_default_float_ptr(), 0.0))  # Placeholder for negative_slope
    # Store the result
    tl.store(out_ptr + block_start, dot, mask=mask)


def triton_gemm_relu(a: torch.Tensor, b: torch.Tensor, multiplier: float, negative_slope: float):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()

    # Prepare output tensor
    out = torch.empty_like(a)

    # Number of rows and columns
    n_rows = a.shape[0]
    n_cols = a.shape[1]
    n_elements_per_row = b.shape[1]

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_rows + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    gemm_relu_kernel[grid](a, b, out, n_rows, n_cols, n_elements_per_row, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that performs a Gemm, multiplies the result, and applies LeakyReLU using custom Triton kernels.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        # Perform Gemm (matrix multiplication)
        weight = torch.nn.Parameter(torch.randn(self.out_features, self.in_features).cuda())
        x = triton_gemm_relu(x, weight, self.multiplier, self.negative_slope)
        return x
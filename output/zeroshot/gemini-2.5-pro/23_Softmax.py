import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 2048}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for row-wise Softmax.
    This kernel is memory-bound and fuses the softmax calculation into a single launch
    to minimize reads/writes to global memory. It performs three passes over the data
    internally without writing intermediate results back to HBM.

    Pass 1: Find the maximum value of the row for numerical stability.
    Pass 2: Calculate the sum of the exponentials (the denominator).
    Pass 3: Calculate the final softmax values and write to the output tensor.
    """
    # Each program instance computes softmax for one row.
    row_idx = tl.program_id(0)

    # Pointers to the start of the current row in input and output tensors.
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # The block size is the number of columns each program will process at a time.
    # We create a range of offsets to iterate over the columns.
    col_offsets = tl.arange(0, BLOCK_SIZE_M)

    # --- Pass 1: Find row-wise maximum ---
    # Initialize the maximum value to the smallest possible float.
    row_max = -float('inf')
    # Iterate over the columns of the row, BLOCK_SIZE_M elements at a time.
    for col_offset in range(0, tl.cdiv(n_cols, BLOCK_SIZE_M)):
        offsets = col_offset * BLOCK_SIZE_M + col_offsets
        # Create a mask to handle rows where n_cols is not a multiple of BLOCK_SIZE_M.
        mask = offsets < n_cols
        # Load a block of data. `other=-float('inf')` ensures that out-of-bounds
        # values do not affect the maximum calculation.
        input_block = tl.load(row_start_ptr + offsets, mask=mask, other=-float('inf'))
        # Update the row maximum.
        block_max = tl.max(input_block, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # --- Pass 2: Compute sum of exponentials (denominator) ---
    # Initialize the sum to zero.
    numerator_sum = 0.0
    # Iterate over the columns of the row again.
    for col_offset in range(0, tl.cdiv(n_cols, BLOCK_SIZE_M)):
        offsets = col_offset * BLOCK_SIZE_M + col_offsets
        mask = offsets < n_cols
        # Load a block of data. `other=0.0` ensures that out-of-bounds
        # values do not affect the sum.
        input_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Subtract the max for numerical stability (prevents overflow).
        shifted_input = input_block - row_max
        # Compute the exponential.
        numerators = tl.exp(shifted_input.to(tl.float32))
        # Add the sum of the block to the total sum for the row.
        numerator_sum += tl.sum(numerators, axis=0)
    denominator = numerator_sum

    # --- Pass 3: Normalize and write to output ---
    # Iterate over the columns of the row a final time.
    for col_offset in range(0, tl.cdiv(n_cols, BLOCK_SIZE_M)):
        offsets = col_offset * BLOCK_SIZE_M + col_offsets
        mask = offsets < n_cols
        # Load the input data again.
        input_block = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
        # Recompute shifted input and numerators.
        shifted_input = input_block - row_max
        numerators = tl.exp(shifted_input.to(tl.float32))
        # Divide by the denominator to get the final softmax value.
        softmax_output = numerators / denominator
        # Write the result to the output tensor in global memory.
        tl.store(output_row_start_ptr + offsets, softmax_output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function for the Triton softmax kernel.
    """
    # Input validation.
    assert x.is_cuda, "Input tensor must be on a CUDA device."
    assert x.dim() == 2, "Input must be a 2D tensor."
    x = x.contiguous() # Ensure tensor is contiguous in memory.

    n_rows, n_cols = x.shape

    # Allocate output tensor.
    output = torch.empty_like(x)

    # The grid defines the number of kernel instances to launch.
    # We launch one instance per row of the input tensor.
    grid = lambda meta: (n_rows,)

    # Launch the Triton kernel.
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols,
    )

    return output


class ModelNew(nn.Module):
    """
    Simple model that performs a Softmax activation using a custom Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor using the optimized Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        # Replace torch.softmax(x, dim=1) with our custom, fused Triton kernel.
        return triton_softmax(x)
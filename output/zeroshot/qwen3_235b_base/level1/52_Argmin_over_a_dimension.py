import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output indices tensor
    n_rows,         # Number of rows (total elements along reduced dimension)
    n_cols,         # Number of columns (size of dim being reduced)
    stride_xb,      # Stride for batch dimension
    stride_xn,      # Stride for the dimension being reduced
    stride_om,      # Stride for output batch dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    row = tl.program_id(0)
    # Compute offsets for this row
    row_offset = row * stride_xb
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load first element as initial minimum
    init_val = tl.load(x_ptr + row_offset)
    best_idx = tl.zeros((1,), dtype=tl.int32)

    # Iterate over blocks of the reduction dimension
    for start_col in range(0, n_cols, BLOCK_SIZE):
        cols = start_col + col_offsets
        col_mask = mask & (cols < n_cols)
        x = tl.load(x_ptr + row_offset + cols * stride_xn, mask=col_mask, other=float('inf'))
        min_val, min_idx = tl.reduce(x, axis=0, reduce_op=tl.minimum_with_index)
        # Compare with current best
        update = min_val < init_val
        init_val = tl.where(update, min_val, init_val)
        best_idx = tl.where(update, start_col + min_idx, best_idx)

    # Store result
    tl.store(output_ptr + row * stride_om, best_idx)


def triton_argmin(x: torch.Tensor, dim: int):
    """
    Custom Triton implementation of argmin along a given dimension.
    Supports reduction along any dimension, but optimized for reduction over dim=-1 or dim=1 in 3D.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    dim = dim if dim >= 0 else dim + x.ndim

    # We assume reduction over dim=1 (dim1=4096 in the example), so we flatten the other dims
    # and reduce over dim1
    if dim == 1:
        # Shape: [batch_size, dim1, dim2] -> treat each [dim1] slice across batch and dim2 as independent reduction
        batch_size, dim1, dim2 = x.shape
        x_reshaped = x.transpose(1, 2)  # Now [batch_size, dim2, dim1]
        x_reshaped = x_reshaped.reshape(-1, dim1)  # [batch_size * dim2, dim1]
        n_rows = x_reshaped.shape[0]
        n_cols = x_reshaped.shape[1]
        stride_xb = x_reshaped.stride(0)
        stride_xn = x_reshaped.stride(1)

        output = torch.empty(n_rows, dtype=torch.int64, device=x.device)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        def grid(meta):
            return (n_rows,)

        argmin_kernel[grid](
            x_reshaped,
            output,
            n_rows,
            n_cols,
            stride_xb,
            stride_xn,
            1,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape back: [batch_size, dim2]
        output = output.reshape(batch_size, dim2)
        return output
    else:
        # Fallback to PyTorch for other dims
        return torch.argmin(x, dim=dim)


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for argmin operation.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.cuda().contiguous()
        return triton_argmin(x, self.dim)
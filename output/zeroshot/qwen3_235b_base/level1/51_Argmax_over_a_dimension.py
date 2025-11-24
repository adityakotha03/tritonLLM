import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,          # pointer to input tensor
    output_ptr,     # pointer to output indices tensor
    n_rows,         # number of rows (total elements along reduced dimension)
    n_cols,         # number of columns (size of the dimension being reduced)
    stride_xb,      # stride for batch dimension
    stride_xn,      # stride for the dimension being reduced
    stride_xm,      # stride for output dimension
    BLOCK_M: tl.constexpr,  # block size for output dimension
    BLOCK_N: tl.constexpr,  # block size for reduced dimension
):
    # program ids
    pid_b = tl.program_id(axis=0)  # batch index
    pid_m = tl.program_id(axis=1)  # output dimension index (row)

    # compute offsets
    row_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offset = tl.arange(0, BLOCK_N)
    mask_m = row_offset < n_rows
    mask_n = col_offset < n_cols

    # initialize values and indices
    value = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    index = tl.zeros([BLOCK_M], dtype=tl.int64)

    # iterate over blocks in the reduced dimension
    for start_n in range(0, n_cols, BLOCK_N):
        col_block_offset = start_n + col_offset
        mask = mask_n[None, :] & (col_block_offset < n_cols)

        # load data block
        offsets = (
            pid_b * stride_xb +
            row_offset[:, None] * stride_xn +
            col_block_offset[None, :] * stride_xm
        )
        x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

        # get max values and update indices
        new_value, new_index = tl.max(x, axis=1, return_indices=True)

        # update global max and corresponding index
        update = new_value > value
        value = tl.where(update, new_value, value)
        index = tl.where(update, col_block_offset[new_index], index)

    # write result
    output_offsets = pid_b * n_rows + row_offset
    tl.store(output_ptr + output_offsets, index, mask=mask_m)


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for argmax.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ensure input is contiguous
        x = x.contiguous()

        # handle negative dim
        dim = self.dim
        if dim < 0:
            dim += x.ndim

        n_rows = 1
        for i in range(dim):
            n_rows *= x.shape[i]
        n_cols = x.shape[dim]

        # output shape: remove `dim`
        output_shape = x.shape[:dim] + x.shape[dim+1:]
        out = torch.empty(output_shape, dtype=torch.int64, device=x.device)

        # launch kernel
        def grid(meta):
            return (x.shape[dim-1] if dim > 0 else 1, triton.cdiv(n_rows, meta['BLOCK_M']))

        # use autotuned block sizes or fixed ones based on common practice
        BLOCK_M = 32
        BLOCK_N = 1024

        argmax_kernel[grid](
            x_ptr=x,
            output_ptr=out,
            n_rows=n_rows,
            n_cols=n_cols,
            stride_xb=x.stride(0) if x.ndim > 1 else 1,
            stride_xn=x.stride(dim),
            stride_xm=x.stride(dim-1) if dim > 0 else 1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return out
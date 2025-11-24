import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def min_reduce_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    input_stride_0, # Stride for dimension 0
    input_stride_1, # Stride for dimension 1
    input_stride_2, # Stride for dimension 2
    reduction_size, # Size of the dimension being reduced (dim1 = 4096)
    out_stride_0,   # Stride for output dimension 0
    out_stride_1,   # Stride for output dimension 1
    num_outputs,    # Total number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    if pid >= num_outputs:
        return

    # Decode output index into (batch_idx, col_idx) for output[batch_idx, col_idx]
    batch_idx = pid // out_stride_1
    col_idx = pid % out_stride_1

    # Base offset in input for this (batch_idx, col_idx)
    base_offset = batch_idx * input_stride_0 + col_idx * input_stride_2

    # Initialize min value with the first element
    first_offset = base_offset
    min_val = tl.load(x_ptr + first_offset)

    # Reduce along the reduction dimension (dim1) in blocks
    for start in range(0, reduction_size, BLOCK_SIZE):
        offsets = base_offset + tl.arange(0, BLOCK_SIZE) + start
        mask = (tl.arange(0, BLOCK_SIZE) < reduction_size - start) & (start + tl.arange(0, BLOCK_SIZE) < reduction_size)
        vals = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
        min_val = tl.minimum(min_val, vals)

    # Store the final min value
    tl.store(output_ptr + pid, min_val)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        assert self.dim in [0, 1, 2], "Reduction dim must be 0, 1, or 2"

        # Move to contiguous form
        x = x.contiguous()

        # Handle reduction along dim=1 (the middle dimension of shape [128, 4096, 4095])
        assert self.dim == 1, "Only dim=1 reduction is supported in this optimized kernel"
        
        # Output shape: [batch_size, dim2] = [128, 4095]
        output_shape = [x.shape[0], x.shape[2]]
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Strides
        input_stride_0, input_stride_1, input_stride_2 = x.stride()
        out_stride_0, out_stride_1 = out.stride()

        # Total number of output elements
        num_outputs = out.numel()

        # Set block size (must be power of 2, large enough to cover 4096)
        BLOCK_SIZE = 512  # Should be enough since 4096 / 512 = 8 iterations

        # Grid: one block per output element
        grid = lambda meta: (num_outputs,)

        # Launch kernel
        min_reduce_kernel[grid](
            x_ptr=x,
            output_ptr=out,
            input_stride_0=input_stride_0,
            input_stride_1=input_stride_1,
            input_stride_2=input_stride_2,
            reduction_size=x.shape[self.dim],
            out_stride_0=out_stride_0,
            out_stride_1=out_stride_1,
            num_outputs=num_outputs,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out
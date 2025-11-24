import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def max_reduce_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    input_stride_b, # Stride for batch dimension
    input_stride_m, # Stride for dim1
    input_stride_n, # Stride for dim2
    output_stride_b, # Stride for output batch
    output_stride_m, # Stride for output dim1
    n_elements_m,   # Size of dim1 (4096)
    n_elements_n,   # Size of dim2 (4095)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D block: each block handles a BLOCK_M x BLOCK_N tile
    pid_b = tl.program_id(0)  # Batch index
    pid_m = tl.program_id(1)  # Block row index

    # Compute offsets
    block_start_m = pid_m * BLOCK_M
    offsets_m = block_start_m + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, BLOCK_N)

    # Input and output pointers for this block
    input_block_ptr = tl.make_block_ptr(
        base=x_ptr + pid_b * input_stride_b,
        shape=(n_elements_m, n_elements_n),
        strides=(input_stride_m, input_stride_n),
        offsets=(block_start_m, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    
    # Load the entire block (with padding handled by mask)
    mask = (offsets_m[:, None] < n_elements_m) & (offsets_n[None, :] < n_elements_n)
    x = tl.load(input_block_ptr, boundary_check=(0,1), padding_option="minus_inf")

    # Perform max reduction along dim2 (axis 1)
    max_vals = tl.max(x, axis=1)

    # Write result to output
    output_offsets = pid_b * output_stride_b + offsets_m * output_stride_m
    output_mask = offsets_m < n_elements_m
    tl.store(output_ptr + output_offsets, max_vals, mask=output_mask)

class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for max reduction.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on GPU."
        x = x.contiguous()

        # We reduce along dim=-1 (last dimension), so permute if needed
        if self.dim == -1 or self.dim == x.ndim - 1:
            x_arg = x
        else:
            # Move reduction dimension to the end
            x_arg = x.transpose(self.dim, -1).contiguous()

        # Prepare output shape
        output_shape = x_arg.shape[:-1]
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        # Get strides
        input_stride_b, input_stride_m, input_stride_n = x_arg.stride()
        output_stride_b, output_stride_m = out.stride()

        n_elements_m = x_arg.shape[-2]  # 4096
        n_elements_n = x_arg.shape[-1]  # 4095

        # 1D grid over batch and dim1 blocks
        num_blocks_m = triton.cdiv(n_elements_m, 64)
        grid = (x_arg.shape[0], num_blocks_m)

        # Launch kernel
        max_reduce_kernel[grid](
            x_arg,
            out,
            input_stride_b, input_stride_m, input_stride_n,
            output_stride_b, output_stride_m,
            n_elements_m, n_elements_n,
            BLOCK_M=64,
            BLOCK_N=128,
        )

        return out
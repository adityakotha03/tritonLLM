import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def exclusive_cumsum_kernel(
    x_ptr,        # pointer to input tensor
    output_ptr,   # pointer to output tensor
    n_elements,   # total number of elements in the tensor
    stride,       # stride along the reduction dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute starting offset for this block
    block_start = pid * BLOCK_SIZE * stride
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride
    
    # Mask to avoid out-of-bounds access
    mask = offsets < n_elements
    
    # Load the first element (will be zero in exclusive cumsum)
    acc = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform exclusive cumsum: store accumulated value, then add current
    for i in range(BLOCK_SIZE):
        current_offset = block_start + i * stride
        current_mask = current_offset < n_elements
        if i == 0:
            # First element in block is zero (exclusive)
            tl.store(output_ptr + current_offset, 0.0, mask=current_mask)
        else:
            prev_offset = block_start + (i - 1) * stride
            prev_mask = prev_offset < n_elements
            acc = tl.load(output_ptr + prev_offset, mask=prev_mask, other=0.0) + tl.load(x_ptr + current_offset, mask=current_mask, other=0.0)
            tl.store(output_ptr + current_offset, acc, mask=current_mask)


def triton_exclusive_cumsum(x: torch.Tensor, dim: int):
    assert x.is_cuda, "Input tensor must be on GPU."
    x = x.contiguous()
    
    # Reshape to flatten all dimensions except the cumsum dim
    original_shape = x.shape
    dim = dim if dim >= 0 else dim + x.ndim
    n_elements = x.numel()
    stride = x.stride(dim)
    
    # Reshape to 2D: [prod(other_dims), size(dim)]
    dim_size = x.size(dim)
    other_dims = n_elements // dim_size
    x_reshaped = x.view(-1, dim_size) if dim == x.ndim - 1 else x.movedim(dim, -1).reshape(-1, dim_size)
    
    # Output buffer
    output = torch.empty_like(x_reshaped)
    
    # Grid and block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (x_reshaped.shape[0] * triton.cdiv(x_reshaped.shape[1], meta['BLOCK_SIZE']),)
    
    # Launch kernel for each row (each program handles one sequence along dim)
    exclusive_cumsum_kernel[grid](
        x_reshaped,
        output,
        n_elements=x_reshaped.shape[1],
        stride=1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back
    output = output.view(*original_shape)
    return output


class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_exclusive_cumsum(x, self.dim)
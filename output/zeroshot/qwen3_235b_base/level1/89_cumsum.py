import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cumsum_kernel(
    x_ptr,      # Pointer to input tensor
    out_ptr,    # Pointer to output tensor
    n_elements, # Total number of elements in the tensor
    stride,     # Stride along the reduction dimension
    size_dim,   # Size of the dimension along which cumsum is applied
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute starting offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Each block processes one "row" along the reduction dimension
    # We need to determine which row this block belongs to
    row_idx = offsets // stride % size_dim
    base_offset = (offsets // (stride * size_dim)) * (stride * size_dim) + (offsets % stride)

    # Load the entire segment for this block
    x = tl.load(x_ptr + base_offset + row_idx * stride, mask=mask, other=0.0)

    # Perform exclusive prefix sum (scan) within the block
    # Use Hillis-Steele scan for simplicity and good memory coalescing
    cumsum = x
    offset = 1
    while offset < BLOCK_SIZE:
        temp = tl.shift_up(cumsum, offset)
        if offset == 1:
            prev_val = tl.load(x_ptr + base_offset + tl.maximum(row_idx - 1, 0) * stride, mask=mask, other=0.0)
            temp = tl.where(row_idx == 0, 0.0, temp)
            temp = tl.where(row_idx > 0, temp, 0.0)
            temp = tl.where(row_idx > 0, temp + prev_val, temp)
        cumsum = cumsum + temp
        offset *= 2

    # Store result
    tl.store(out_ptr + base_offset + row_idx * stride, cumsum, mask=mask)


def triton_cumsum(x: torch.Tensor, dim: int):
    """
    Triton-based cumulative sum implementation.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    # Handle negative dim
    if dim < 0:
        dim = x.ndim + dim

    # Reshape tensor to 2D: (outer_dims * inner_dims, size_dim)
    size_dim = x.size(dim)
    outer_dims = x.numel() // (x.size(dim) * x.stride(dim)) if dim < x.ndim - 1 else x.size(0)
    inner_stride = x.stride(dim)
    
    # Flatten to 2D view
    x_reshaped = x.view(-1, size_dim) if dim == x.ndim - 1 else x.transpose(dim, -1).contiguous().view(-1, size_dim)

    # Prepare output
    out = torch.empty_like(x_reshaped)

    n_elements = x_reshaped.numel()
    BLOCK_SIZE = 1024  # Good balance for A100

    # Grid: one block per element in flattened outer dimensions
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    cumsum_kernel[grid](
        x_reshaped,
        out,
        n_elements,
        inner_stride,
        size_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back
    out = out.view(x.shape) if dim == x.ndim - 1 else out.view(*x.transpose(dim, -1).shape).transpose(dim, -1)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the Scan model using a custom Triton kernel for cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return triton_cumsum(x, self.dim)
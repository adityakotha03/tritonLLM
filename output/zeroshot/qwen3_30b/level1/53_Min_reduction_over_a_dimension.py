import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduction_kernel(
    x_ptr,
    out_ptr,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    out_stride_0,
    out_stride_1,
    n_elements,
    block_size: tl.constexpr,
    dim_size: tl.constexpr,
):
    # Each program handles one output element in the reduced dimension
    # We map each program to a unique output element along the non-reduced dims
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    # Compute the starting offset for this output element
    out_offset = pid * block_size
    mask = out_offset + tl.arange(0, block_size) < n_elements

    # Compute the output tensor shape
    # Output is of shape [batch_size, dim1] if dim=2, i.e., reduced dim2
    # We use a 1D grid where each program processes block_size outputs

    # Determine which output element this thread block is responsible for
    # We compute the global output index
    out_idx = out_offset + tl.arange(0, block_size)
    # Convert to (batch, dim1, dim2) style indexing
    # Since we reduce over dim2, each output is a single value from a (dim1, dim2) slice

    # We are reducing over dim2 (the last dimension), so:
    # output[batch, dim1] = min over dim2 of input[batch, dim1, dim2]

    # Compute batch and dim1 indices for this output
    # We need to map the linear output index to the actual (batch, dim1) indices
    batch_idx = (out_idx // dim1) % x_stride_0
    dim1_idx = (out_idx // dim1) % dim1
    # Actually, better approach: we can recompute this with broadcasting logic

    # Actually, the simplest: assume the output is flattened for 1D grid
    # So we loop over (batch, dim1) pairs
    # Each block handles block_size outputs
    # Total output elements = batch_size * dim1
    # We iterate over all (batch, dim1) pairs
    # Each program handles block_size output indices

    # Recompute the base index: each output is for (batch, dim1) = (b, d1)
    # So we need to know which (b, d1) pair this output belongs to
    # We use: total_output_size = batch_size * dim1
    # We can recompute batch and dim1 indices from linear index

    # The output tensor is flattened: [batch_size * dim1] of results
    # We'll process it as a 1D array
    # So output index i corresponds to (i // dim1, i % dim1)
    # But wait: our input is (batch_size, dim1, dim2), so output is (batch_size, dim1)
    # So we want to reduce over dim2: for each (b, d1), find min over d2

    # So the linear output index = b * dim1 + d1
    # Therefore, to recover b and d1:
    #   b = out_idx // dim1
    #   d1 = out_idx % dim1

    # But wait: we only have n_elements = batch_size * dim1
    # So we can compute:
    out_idx_flat = out_offset + tl.arange(0, block_size)
    b = out_idx_flat // dim1
    d1 = out_idx_flat % dim1

    # Now compute the base input pointer for this (b, d1) pair
    # Base input offset = b * x_stride_0 + d1 * x_stride_1
    # We will load dim2 elements from this base offset
    base_offset = b * x_stride_0 + d1 * x_stride_1
    # Each step of dim2 is x_stride_2

    # Initialize min value to a large number
    # Use float32 min for safety
    min_val = tl.full((block_size,), float('inf'), dtype=tl.float32)

    # Load each element in dim2 dimension
    for i in range(0, dim_size, block_size):
        # Current dim2 offset
        dim2_offset = i
        # Compute actual offsets in input tensor
        offsets = base_offset + dim2_offset + tl.arange(0, block_size) * x_stride_2
        # We need to check bounds for dim2
        valid = dim2_offset + tl.arange(0, block_size) < dim_size
        x_vals = tl.load(x_ptr + offsets, mask=valid, other=float('inf'))
        # Update min
        min_val = tl.minimum(min_val, x_vals)

    # Store result back
    out_offsets = out_idx_flat * out_stride_0 + d1 * out_stride_1
    tl.store(out_ptr + out_offsets, min_val, mask=mask)


def triton_min_reduction(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Performs min reduction along a given dimension using Triton kernel.

    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, dim1, dim2]
        dim (int): Dimension to reduce over (must be 2, i.e., dim2)

    Returns:
        torch.Tensor: Reduced tensor of shape [batch_size, dim1]
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    assert x.dim() == 3, "Input must be 3D tensor"
    assert dim == 2, "Only dim=2 supported currently"

    # Ensure contiguous memory
    x = x.contiguous()

    # Output shape: [batch_size, dim1]
    batch_size, dim1, dim2 = x.shape
    output_shape = (batch_size, dim1)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    # Determine block size and grid
    # We'll use 128 for optimal performance
    BLOCK_SIZE = 128
    total_output_elements = batch_size * dim1

    # Compute number of blocks needed
    grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Use autotuning to select best block size
    # For now, we fix to 128 for simplicity, but we could autotune
    # Here we do not use autotune decorator but could add it if needed

    # Get strides
    x_stride_0, x_stride_1, x_stride_2 = x.stride()
    out_stride_0, out_stride_1 = out.stride()

    # Launch kernel
    min_reduction_kernel[grid_size](
        x_ptr=x, out_ptr=out,
        x_stride_0=x_stride_0, x_stride_1=x_stride_1, x_stride_2=x_stride_2,
        out_stride_0=out_stride_0, out_stride_1=out_stride_1,
        n_elements=total_output_elements,
        block_size=BLOCK_SIZE,
        dim_size=dim2,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_reduction(x, self.dim)
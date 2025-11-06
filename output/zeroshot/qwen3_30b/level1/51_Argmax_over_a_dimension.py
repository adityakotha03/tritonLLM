import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    block_size: tl.constexpr,
    dim_size: tl.constexpr,
    batch_size: tl.constexpr,
    dim: tl.constexpr,
):
    # Calculate the global offset for this block
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)

    # Mask to handle out-of-bounds elements
    mask = offsets < n_elements

    # Compute the actual dimensions for indexing
    # The dimension to reduce is `dim`, so we need to compute the stride
    # For 3D tensor (batch_size, dim1, dim2), the stride depends on `dim`
    if dim == 1:
        # Reducing along dim1: each block handles a slice of dim1
        block_size_dim = block_size // dim_size
        # Each thread handles one element in the reduction dimension
        thread_id = offsets % dim_size
        batch_id = offsets // dim_size
        thread_id = tl.where(thread_id < dim_size, thread_id, 0)
        batch_id = tl.where(batch_id < batch_size, batch_id, 0)
    elif dim == 2:
        # Reducing along dim2: each block handles a slice of dim2
        block_size_dim = block_size // dim_size
        thread_id = offsets % dim_size
        batch_id = offsets // dim_size
        thread_id = tl.where(thread_id < dim_size, thread_id, 0)
        batch_id = tl.where(batch_id < batch_size, batch_id, 0)
    else:
        raise ValueError("Unsupported dim for argmax")

    # Calculate the base pointer for this batch and dim
    base_batch_offset = batch_id * dim1 * dim2
    base_dim_offset = base_batch_offset + (thread_id if dim == 1 else 0) * dim2
    if dim == 2:
        base_dim_offset = base_batch_offset + thread_id

    # Load the values from the input tensor
    x_values = tl.load(x_ptr + base_dim_offset + tl.arange(0, dim_size), mask=tl.arange(0, dim_size) < dim_size, other=-float('inf'))
    # Perform argmax in shared memory using a reduction with local voting
    # We use shared memory to store the current max value and its index
    shared_mem = tl.zeros((block_size_dim, 2), dtype=tl.int32)  # Store index and value

    # Initialize shared memory with thread indices and values
    thread_idx = thread_id
    local_max = x_values
    local_max_idx = thread_idx

    # Reduction: find the max index in shared memory
    for i in range(1, dim_size):
        # Create a new mask
        mask = i < dim_size
        # Load the next value
        val = tl.load(x_ptr + base_dim_offset + i, mask=mask, other=-float('inf'))
        # Update the max if needed
        new_max = tl.where(val > local_max, val, local_max)
        new_idx = tl.where(val > local_max, i, local_max_idx)
        local_max = new_max
        local_max_idx = new_idx

    # Write the result back to global memory
    # Use a single output per block
    out_offset = batch_id * dim_size
    tl.store(out_ptr + out_offset, local_max_idx, mask=mask)


def triton_argmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Ensure input is contiguous and on CUDA
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Get tensor dimensions
    batch_size, dim1, dim2 = x.shape
    if dim == 1:
        n_elements = batch_size * dim1
        dim_size = dim2
    elif dim == 2:
        n_elements = batch_size * dim2
        dim_size = dim1
    else:
        raise ValueError("Unsupported dim for argmax")

    # Prepare output tensor
    out = torch.empty(batch_size, dim_size, dtype=torch.int32, device=x.device)

    # Choose block size (must be power of 2)
    # For A100, 256 or 512 are good starting points
    BLOCK_SIZE = 512

    # Grid configuration: one block per batch * dim_size
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch kernel
    argmax_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        dim_size=dim_size,
        batch_size=batch_size,
        dim=dim
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmax(x, dim=self.dim)
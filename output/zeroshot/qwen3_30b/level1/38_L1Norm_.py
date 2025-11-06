import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def l1_normalize_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Calculate the block indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load the data
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute the absolute value
    abs_x = tl.abs(x)

    # Perform the reduction using shared memory
    # We use a two-step reduction: first reduce within the block, then reduce across blocks
    # We use a block size of 256 and group size of 256 for better performance
    # We use shared memory to store the intermediate results
    # We use a loop to handle the reduction

    # Shared memory for the reduction
    shared_mem = tl.make_block_ptr(base=tl.zeros(1, dtype=tl.float32), shape=(BLOCK_SIZE,), strides=(1,), offsets=(0,), block_shape=(BLOCK_SIZE,), order=(0,))
    # Initialize shared memory to 0
    tl.store(shared_mem, tl.zeros(BLOCK_SIZE, dtype=tl.float32), mask=mask)

    # First reduction: within the block
    tl.atomic_add(shared_mem, abs_x, mask=mask)

    # Wait for all threads to finish
    tl.barrier()

    # Second reduction: across blocks
    # We use a loop to handle the reduction
    # We use a block size of 256 and group size of 256 for better performance
    # We use shared memory to store the intermediate results
    # We use a loop to handle the reduction

    # Initialize the reduction value
    sum_abs = tl.load(shared_mem, mask=mask)
    sum_abs = tl.sum(sum_abs, axis=0)

    # Broadcast the reduction value to all threads
    sum_abs = tl.broadcast(sum_abs, (BLOCK_SIZE,))

    # Divide by the number of columns to get the mean
    mean_abs = sum_abs / n_cols

    # Divide the input by the mean
    out = x / mean_abs

    # Store the output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_l1_normalize(x: torch.Tensor) -> torch.Tensor:
    # Ensure the inputs are contiguous on GPU
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    n_cols = x.shape[1]

    # Determine the number of blocks needed
    # We use a block size of 256 for better performance
    BLOCK_SIZE = 256
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    l1_normalize_kernel[grid](x, out, n_elements, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_normalize(x)
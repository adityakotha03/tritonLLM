import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def log_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    batch_size,
    dim,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    # Calculate the row (batch) index
    row = tl.program_id(0)
    # Calculate the block index for the dim dimension
    col_block = tl.program_id(1)

    # Calculate offsets for the current block
    row_offset = row * dim
    col_offset = col_block * BLOCK_SIZE_DIM

    # Create a range of offsets for the current block
    offsets = col_offset + tl.arange(0, BLOCK_SIZE_DIM)
    mask = offsets < dim

    # Load the input data for the current block
    x = tl.load(x_ptr + row_offset + offsets, mask=mask, other=-float("inf"))

    # Step 1: Find the maximum value in the row (online)
    # Use a reduction to find max in blocks
    x_max = tl.max(x, axis=0)

    # Step 2: Compute logsumexp (online) using the max value
    # Shift by max to avoid overflow
    x_shifted = x - x_max
    # Use exp and sum in one pass
    x_exp = tl.exp(x_shifted)
    x_exp_sum = tl.sum(x_exp, axis=0)

    # Compute logsumexp
    logsumexp = tl.log(x_exp_sum) + x_max

    # Step 3: Compute the final log softmax: log(exp(x - logsumexp))
    out = x - logsumexp

    # Store the output
    tl.store(out_ptr + row_offset + offsets, out, mask=mask)


def triton_log_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Apply LogSoftmax using a custom Triton kernel with online computation
    and fusion of max, exp, sum, and log operations.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    x = x.contiguous()

    # Get the shape
    batch_size, dim = x.shape

    # Prepare output tensor
    out = torch.empty_like(x)

    # Set the block sizes (tuned for A100)
    BLOCK_SIZE = 128  # Optimal for warp size and coalescing
    BLOCK_SIZE_DIM = 256  # Must be a power of 2, aligned with A100 shared memory

    # Grid setup: one block per row, and multiple blocks per row for dim dimension
    grid = lambda meta: (batch_size, (dim + meta["BLOCK_SIZE_DIM"] - 1) // meta["BLOCK_SIZE_DIM"])

    # Launch the Triton kernel
    log_softmax_kernel[grid](
        x_ptr=x, out_ptr=out, n_elements=x.numel(), batch_size=batch_size,
        dim=dim, BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_DIM=BLOCK_SIZE_DIM
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_log_softmax(x)
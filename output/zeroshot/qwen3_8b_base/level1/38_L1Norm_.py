import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def l1_norm_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in the tensor
    dim,  # Dimension to normalize across
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute the index in the tensor
    idx = offsets
    # Compute the index in the batch
    batch_idx = idx // dim
    # Compute the index within the dimension
    dim_idx = idx % dim

    # Load the input values
    x = tl.load(x_ptr + idx, mask=idx < n_elements, other=0.0)

    # Compute the absolute value
    abs_x = tl.abs(x)

    # Compute the mean along the specified dimension
    # We need to sum the absolute values and divide by the dimension size
    # To do this, we need to accumulate the sum across all elements in the dimension
    # We'll use shared memory to accumulate the sum for each batch
    # Since we're using a single block, we can just compute the sum directly

    # Compute the sum of absolute values
    sum_abs = tl.sum(abs_x)

    # Compute the mean
    mean = sum_abs / tl.float32(dim)

    # Compute the output
    out = x / mean

    # Store the result
    tl.store(out_ptr + idx, out, mask=idx < n_elements)


def triton_l1_norm(x: torch.Tensor, dim: int):
    """
    Applies L1 normalization using a custom Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    l1_norm_kernel[grid](x, out, n_elements, dim, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_l1_norm(x, dim=1)
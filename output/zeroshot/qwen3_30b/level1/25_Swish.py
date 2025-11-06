import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block index
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bounds access
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Use a numerically stable version of sigmoid
    x_neg = -x
    exp_neg_x = tl.exp(x_neg)
    sigmoid_x = tl.math.recip(1.0 + exp_neg_x)
    # Apply Swish: x * sigmoid(x)
    out = x * sigmoid_x
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Use a BLOCK_SIZE that is a power of 2 and fits well within shared memory
    BLOCK_SIZE = 1024  # Optimized for A100
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel implementing Swish (x * sigmoid(x))
# The kernel operates on a flat array of elements.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def swish_kernel(
    x_ptr: tl.tensor,          # pointer to input tensor
    out_ptr: tl.tensor,        # pointer to output tensor
    n_elements: tl.int32,      # total number of elements
    BLOCK_SIZE: tl.constexpr,  # compile‑time constant for block size
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask to avoid out‑of‑bounds accesses on the last block
    mask = offsets < n_elements

    # load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # compute sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-x))

    # element‑wise Swish
    y = x * sigmoid

    # store result
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_swish(x: torch.Tensor) -> torch.Tensor:
    """
    Apply the Swish activation function using the Triton kernel.
    The input tensor must be on CUDA.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    # Ensure contiguous layout for coalesced memory accesses
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    # grid size: number of programs (blocks) needed to cover all elements
    grid = lambda meta: (
        (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch the Triton kernel
    swish_kernel[grid](x, out, n_elements)

    return out


class ModelNew(nn.Module):
    """
    Optimized model that applies the Swish activation using a custom Triton kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_swish(x)
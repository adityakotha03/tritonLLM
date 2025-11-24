import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    """Element‑wise LeakyReLU kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_leaky_relu(x: torch.Tensor, negative_slope: float) -> torch.Tensor:
    """Wrapper that launches the Triton kernel."""
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (
        (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=grid)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel for LeakyReLU.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_leaky_relu(x, self.negative_slope)


# ------------------------------------------------------------------
# Helper functions to generate inputs for benchmarking / training
# ------------------------------------------------------------------

batch_size = 4096
dim = 393216


def get_inputs():
    x = torch.rand(batch_size, dim, device="cuda")
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
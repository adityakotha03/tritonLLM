import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton kernel for element‑wise tanh
@triton.jit
def tanh_kernel(
    x_ptr,          # pointer to input tensor
    out_ptr,        # pointer to output tensor
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute tanh using the exp implementation: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    e2x = tl.exp(2.0 * x)
    t = (e2x - 1.0) / (e2x + 1.0)

    tl.store(out_ptr + offsets, t, mask=mask)


# Wrapper that launches the Triton kernel
def triton_tanh(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Block size tuned for large tensors; 2048 yields good occupancy on A100
    BLOCK_SIZE = 2048

    # Grid definition: one program per BLOCK_SIZE elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the kernel
    tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


# New model using the Triton tanh implementation
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_tanh(x)
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x * beta

    # Compute softplus: log(1 + exp(beta * x)) / beta
    # Use threshold to avoid overflow: if beta*x > threshold, then softplus(x) ≈ x
    safe_input = tl.where(x_scaled > threshold, 0.0, x_scaled)
    exp_safe = tl.exp(safe_input)
    log_exp = tl.log(1.0 + exp_safe)
    result_safe = log_exp / beta

    # Approximate as x when beta*x is large
    result = tl.where(x_scaled > threshold, x, result_safe)

    tl.store(out_ptr + offsets, result, mask=mask)

def triton_softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Heuristic block size; can be autotuned for specific shapes
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    softplus_kernel[grid](
        x,
        out,
        n_elements,
        beta=beta,
        threshold=threshold,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    """
    Optimized version of Model using a custom Triton kernel for Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardsigmoid_kernel(
    x_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the block index and start offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardSigmoid: min(max(0, x + 3) / 6, 1)
    # Compute x + 3
    x_plus_3 = x + 3.0
    # Apply clamping: clamp(x + 3, 0, 6)
    clamped = tl.clamp(x_plus_3, 0.0, 6.0)
    # Divide by 6
    out = clamped / 6.0
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Applies HardSigmoid using a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Number of elements
    n_elements = x.numel()
    
    # Define grid and block size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024  # Power of 2, suitable for A100
    
    # Launch kernel
    hardsigmoid_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_hardsigmoid(x)
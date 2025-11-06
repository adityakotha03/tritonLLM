import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softsign_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions (last block might be smaller)
    mask = offsets < n_elements
    
    # Load the input data with masking to avoid out-of-bounds access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply Softsign: x / (1 + |x|)
    # Use abs for absolute value, then compute denominator
    abs_x = tl.abs(x)
    denominator = 1.0 + abs_x
    out = x / denominator
    
    # Store the result back to output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softsign(x: torch.Tensor):
    """
    Applies Softsign activation using a custom Triton kernel.
    This function ensures:
        - Input is contiguous on GPU
        - Uses optimal block size via autotuning
        - Launches the Triton kernel with a grid size based on total elements
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()
    
    # Autotune the block size (BLOCK_SIZE) using Triton's autotuner
    # We target powers of 2: 128, 256, 512, 1024 (max 1024 for A100 with shared memory constraints)
    # Use float32 for full precision; no need to downcast since A100 has strong FP32 support
    # But since we are memory-bound, using larger blocks can improve bandwidth utilization

    # Define grid function
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel with autotuning
    softsign_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=1024,  # Default; autotune will explore 128, 256, 512, 1024
    )
    
    return out


# Autotuned version with proper @triton.autotune
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def softsign_kernel_autotuned(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle boundary conditions (last block might be smaller)
    mask = offsets < n_elements
    
    # Load the input data with masking to avoid out-of-bounds access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply Softsign: x / (1 + |x|)
    abs_x = tl.abs(x)
    denominator = 1.0 + abs_x
    out = x / denominator
    
    # Store the result back to output
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_softsign_autotuned(x: torch.Tensor):
    """
    Applies Softsign activation using a custom Triton kernel with autotuning.
    This version automatically selects the best BLOCK_SIZE and tuning parameters.
    """
    assert x.is_cuda, "Input tensor must be on CUDA"
    x = x.contiguous()

    # Output tensor
    out = torch.empty_like(x)

    # Total number of elements
    n_elements = x.numel()
    
    # Define grid function
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch autotuned kernel
    softsign_kernel_autotuned[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE=1024,  # Initial value; autotune will explore options
    )
    
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the autotuned Triton kernel for optimal performance
        return triton_softsign_autotuned(x)
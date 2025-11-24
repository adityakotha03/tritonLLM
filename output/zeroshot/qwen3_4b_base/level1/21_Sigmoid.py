import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute sigmoid: 1 / (1 + exp(-x))
    # We use the identity: sigmoid(x) = 1 / (1 + exp(-x))
    # To avoid underflow/overflow, we handle large positive and negative values
    # For x >= 0: exp(-x) is small, so we compute 1 / (1 + exp(-x)) directly
    # For x < 0: we compute exp(x) / (1 + exp(x)) to avoid underflow
    # We use the trick: sigmoid(x) = 1 - sigmoid(-x) when x < 0
    # But instead of branching, we use a single expression with masking

    # We use a numerically stable sigmoid computation:
    # exp(-x) when x >= 0, and exp(x) when x < 0
    # We avoid branching by using conditional logic in a fused way
    # But since we cannot easily branch in a warp, we use a single expression
    # Using: sigmoid(x) = 1 / (1 + exp(-x)) with clamping of exp(-x)

    # We compute exp(-x) safely
    # For x > 20, exp(-x) is ~0, so sigmoid(x) ≈ 1
    # For x < -20, exp(-x) is ~inf, so sigmoid(x) ≈ 0
    # We use a stable approach: exp(-x) = exp(-x), and clamp to avoid overflow

    # We use a stable sigmoid: avoid exp(-x) overflow/underflow
    # Use: sigmoid(x) = 1 / (1 + exp(-x))
    # We compute exp(-x) safely using a stable method
    # We avoid direct exp because it's expensive and may overflow

    # Instead, we use a fused approach: compute exp(-x) only when needed
    # But in Triton, we can't easily use conditional exp without branching
    # So we use a numerical trick: clamp x to avoid overflow

    # We clamp x to avoid exp overflow
    # For x > 20, use 1.0; for x < -20, use 0.0
    # For values in between, compute sigmoid(x)
    # We use a stable sigmoid with clipping

    # We use: exp(-x) = exp(-x) when x < 20, otherwise 0 or inf
    # We compute exp(-x) only when x is not too large
    # But since we are in a kernel, we can use a fused expression

    # Use the identity: sigmoid(x) = 1 / (1 + exp(-x))
    # We compute exp(-x) with a stable method using expm1 and log1p, but that's not directly helpful

    # Instead, we use a simple and safe method: compute exp(-x) only when x is not too large
    # We use a mask to avoid overflow
    # But to avoid branching, we use a single expression with clamping

    # We use a stable sigmoid computation using exp(-x) with clamping
    # We compute exp(-x) safely
    exp_neg_x = tl.exp(-x)
    # Clamp to avoid overflow: exp(-x) should be in [1e-10, 1e10]
    # For x > 20, exp(-x) < 1e-9 → sigmoid ≈ 1
    # For x < -20, exp(-x) > 1e9 → sigmoid ≈ 0
    # So we clamp exp(-x) to [1e-10, 1e10]
    exp_neg_x = tl.clip(exp_neg_x, 1e-10, 1e10)

    # Compute sigmoid
    sigmoid_val = 1.0 / (1.0 + exp_neg_x)

    # Store result
    tl.store(out_ptr + offsets, sigmoid_val, mask=mask)


def triton_sigmoid(x: torch.Tensor):
    """
    Custom sigmoid kernel using Triton for improved performance.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()

    n_elements = x.numel()
    BLOCK_SIZE = 256  # Optimized for Ampere: 256 is a good balance between occupancy and memory access

    # Grid size: number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    sigmoid_kernel[grid](x, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return x


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid activation to the input tensor using a custom Triton kernel.
        """
        return triton_sigmoid(x)
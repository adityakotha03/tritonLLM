import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# -------------------- Triton kernel for GELU --------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def gelu_kernel(
    x_ptr,          # input
    out_ptr,        # output
    N,              # number of elements
    sqrt_two_over_pi: tl.constexpr,
    one: tl.constexpr,
    coef: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU computation
    x_cubed = x * x * x
    inner = x + coef * x_cubed
    tanh_term = tl.math.tanh(sqrt_two_over_pi * inner)
    y = 0.5 * x * (one + tanh_term)

    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

# -------------------- Wrapper --------------------
def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Element‑wise GELU using a Triton kernel.
    """
    assert x.is_cuda and x.is_bfloat16() or x.is_float() or x.is_half(), \
        "Input must be on CUDA and of a supported type (float16, bfloat16, float32)."

    # Ensure contiguous
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Constants for the GELU formula
    sqrt_two_over_pi = float(math.sqrt(2.0 / math.pi))
    coef = 0.044715
    one = 1.0

    N = x.numel()

    # Grid definition
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    # Launch kernel
    gelu_kernel[grid](
        x,
        out,
        N,
        sqrt_two_over_pi,
        one,
        coef,
        BLOCK_SIZE=128,  # will be overridden by autotuner
    )
    return out

# -------------------- Optimized model --------------------
class ModelNew(nn.Module):
    """
    Same forward semantics as the original Model, but using a custom
    Triton implementation of GELU for better performance on the A100.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return triton_gelu(x)
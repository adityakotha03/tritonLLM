import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["n"],
)
@triton.jit
def fused_ops_kernel(
    input_ptr,          # pointer to conv output
    scale_ptr,          # pointer to scaling_factor
    bias_ptr,           # pointer to bias
    output_ptr,         # pointer to output
    n,                  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Scale
    scale = tl.load(scale_ptr, mask=mask, other=0.0)
    x = x * scale

    # Tanh
    x = tl.math.tanh(x)

    # Bias
    bias = tl.load(bias_ptr, mask=mask, other=0.0)
    x = x * bias

    # Sigmoid
    x = tl.math.sigmoid(x)

    # Store result
    tl.store(output_ptr + offsets, x, mask=mask)


def fused_ops(x: torch.Tensor, scaling_factor: torch.Tensor, bias: torch.Tensor):
    """
    Fuse scaling * tanh * bias * sigmoid into a single Triton kernel.
    Assumes all tensors are on CUDA and contiguous.
    """
    assert x.is_cuda and scaling_factor.is_cuda and bias.is_cuda
    x = x.contiguous()
    scaling_factor = scaling_factor.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)

    n = x.numel()
    BLOCK_SIZE = 1024  # tunable; 1024 works well on A100

    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    fused_ops_kernel[grid](
        x, scaling_factor, bias, out, n, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model that fuses post‑conv operations into a single Triton kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        self.scaling_factor = nn.Parameter(torch.full(bias_shape, scaling_factor, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops(x, self.scaling_factor, self.bias)
        return x
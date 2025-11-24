import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel that divides by a constant and adds a bias
@triton.jit
def div_add_bias_kernel(
    x_ptr,          # Pointer to the input tensor after pooling
    bias_ptr,       # Pointer to the bias tensor (broadcasted over batch)
    out_ptr,        # Pointer to the output tensor
    n_elements,     # Total number of elements (batch * out_channels)
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input and bias values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)

    # Compute x / divisor + bias
    out = x / divisor + bias

    tl.store(out_ptr + offsets, out, mask=mask)


def div_add_bias_torch(x: torch.Tensor, bias: torch.Tensor, divisor: float):
    """
    Wrapper that calls the Triton kernel to perform division and bias addition
    on the 5‑D tensor x (shape: [B, C, 1, 1, 1]) and bias (shape: [C, 1, 1, 1]).
    """
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    bias = bias.contiguous()

    out = torch.empty_like(x)

    n_elements = x.numel()          # B * C
    BLOCK_SIZE = 1024

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    div_add_bias_kernel[grid](
        x, bias, out,
        n_elements,
        divisor=divisor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original Model.
    All operations are preserved except that the final division by a constant
    and the bias addition are fused into a single Triton kernel to reduce
    memory traffic and kernel launch overhead.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        # 3D convolution
        x = self.conv(x)
        # Max pooling
        x = self.max_pool(x)
        # Global average pooling -> shape (B, C, 1, 1, 1)
        x = self.global_avg_pool(x)
        # Division by divisor and bias addition in one Triton kernel
        x = div_add_bias_torch(x, self.bias, self.divisor)
        # Sum over the specified dimension
        x = torch.sum(x, dim=self.sum_dim)
        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: add bias + ReLU fused
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def bias_relu_fused_kernel(
    out_ptr,           # pointer to output tensor (after conv)
    bias_ptr,          # pointer to bias tensor of shape [C, 1, 1]
    N, C, H, W,        # tensor dimensions
    stride_N, stride_C, stride_H, stride_W,  # strides for out_ptr
    stride_bias_C,     # stride for bias (C dimension)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < (N * C * H * W)

    # Compute multi-dimensional indices from linear offset
    idx = offsets // (H * W)
    rem = offsets % (H * W)
    c = idx // (H * W)   # channel index
    rem2 = idx % (H * W)
    # We don't need spatial indices explicitly because we can load using strides

    # Load conv output element
    out_offset = (
        offsets * stride_N
    )
    # Compute address manually
    addr = (
        ((offsets // (C * H * W)) * stride_N) +          # batch offset
        ((offsets // (H * W)) % C) * stride_C +          # channel offset
        ((offsets // W) % H) * stride_H +               # height offset
        (offsets % W) * stride_W                         # width offset
    )
    out_val = tl.load(out_ptr + addr, mask=mask, other=0.0)

    # Load bias for channel c
    bias_offset = (c * stride_bias_C)
    bias_val = tl.load(bias_ptr + bias_offset, mask=mask, other=0.0)

    # Apply ReLU and add bias
    out_val = tl.where(out_val > 0, out_val, 0.0) + bias_val

    # Store result
    tl.store(out_ptr + addr, out_val, mask=mask)

# ----------------------------------------------------------------------
# Wrapper function that calls the Triton kernel
# ----------------------------------------------------------------------
def bias_relu_fused(out: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Applies ReLU and adds per‑channel bias to the output tensor in place.
    Assumes out.shape = [N, C, H, W] and bias.shape = [C, 1, 1]
    """
    assert out.is_cuda and bias.is_cuda
    N, C, H, W = out.shape

    stride_N, stride_C, stride_H, stride_W = out.stride()
    stride_bias_C = bias.stride(0)

    grid = lambda meta: ((N * C * H * W + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    bias_relu_fused_kernel[grid](
        out, bias,
        N, C, H, W,
        stride_N, stride_C, stride_H, stride_W,
        stride_bias_C,
        BLOCK_SIZE=256,
    )
    return out

# ----------------------------------------------------------------------
# Optimized Model with fused bias + ReLU
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs convolution, applies fused ReLU and bias addition
    using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard convolution
        out = self.conv(x)
        # Fuse ReLU and bias addition via Triton
        bias_relu_fused(out, self.bias)
        return out

# ----------------------------------------------------------------------
# Helper functions to create inputs
# ----------------------------------------------------------------------
def get_inputs():
    batch_size = 128
    in_channels = 64
    height = width = 128
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    bias_shape = (out_channels, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]
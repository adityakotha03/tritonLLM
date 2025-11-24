import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mul_leaky_relu_gelu_kernel(
    x_ptr,               # Pointer to input (conv output)
    multiplier_ptr,      # Pointer to multiplier (broadcasted)
    out_ptr,             # Pointer to output
    n_elements,          # Total number of elements in x
    channels,            # Number of channels
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for broadcasting: assuming shape (B, C, H, W)
    # Flattened index -> channel = (offsets // (H * W)) % C
    # But we don't have H*W here, so we compute stride dynamically via divmod
    # Instead, we use: channel_idx = (offsets // spatial_size) % channels
    # But we don't pass spatial_size. Alternative: use tl.program_id(1) for channel blocks?
    # Instead, let's reindex: we know that the multiplier has shape (C, 1, 1)
    # So for each element at offset, its channel is: (offsets // (height * width)) % channels
    # However, we don't have height and width. So we pass total_elements_per_batch and per_channel?
    # Instead, simpler: assume we pass stride_hwd = height * width * batch_size? No.

    # Alternative: reshape in kernel logic: flatten all but channel dim
    # Let’s assume layout is (B, C, H, W) -> flattened to (B*C*H*W), and channel stride is H*W
    # But we don't know H*W. So we must pass it.

    # Let's change strategy: fuse only elementwise ops with known channel index
    # We can compute: channel_idx = (offsets // (height_width)) % channels
    # But we don't have height_width.

    # New plan: pass height_width as meta
    height_width = n_elements // (batch_size * channels)
    channel_idx = (offsets // height_width) % channels

    # Load multiplier value per channel
    multiplier = tl.load(multiplier_ptr + channel_idx, mask=mask, other=1.0)

    # Multiply
    x = x * multiplier

    # LeakyReLU: x if x >= 0 else x * negative_slope
    negative_slope = 0.01
    x_leaky = tl.where(x >= 0, x, x * negative_slope)

    # GELU approximation: use tanh-based approximation for speed
    # gelu(x) = x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_cubed = x_leaky * x_leaky * x_leaky
    inner = 0.7978845608028654 * (x_leaky + 0.044715 * x_cubed)  # 0.797... = sqrt(2/pi)
    tanh_inner = tl.tanh(inner)
    gelu_out = 0.5 * x_leaky * (1.0 + tanh_inner)

    # Store result
    tl.store(out_ptr + offsets, gelu_out, mask=mask)


def triton_mul_leaky_relu_gelu(x: torch.Tensor, multiplier: torch.Tensor):
    assert x.is_cuda and multiplier.is_cuda
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()

    # Constants
    batch_size, channels, height, width = x.shape
    height_width = height * width

    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    mul_leaky_relu_gelu_kernel[grid](
        x,
        multiplier,
        out,
        n_elements,
        channels,
        BLOCK_SIZE=BLOCK_SIZE,
        batch_size=batch_size,
        height_width=height_width,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused multiplier, LeakyReLU, and GELU using Triton.
    Conv2d remains as-is (cuDNN-optimized), but elementwise ops are fused.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        x = self.conv(x)
        x = triton_mul_leaky_relu_gelu(x, self.multiplier)
        return x
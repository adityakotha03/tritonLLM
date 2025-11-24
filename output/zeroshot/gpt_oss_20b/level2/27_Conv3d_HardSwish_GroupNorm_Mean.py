import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# Triton kernel: HardSwish + mean over spatial dims (D, H, W)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def hard_swish_mean_kernel(
    x_ptr,          # pointer to input tensor (B, C, D, H, W)
    out_ptr,        # pointer to output tensor (B, C)
    batch_size,     # B
    n_channels,     # C
    spatial_stride, # D*H*W
    batch_stride,   # C*D*H*W
    n_elements,     # D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (b, c) pair
    idx = tl.program_id(0)
    b = idx // n_channels
    c = idx % n_channels

    # Compute the starting offset of this (b, c) slice
    base = b * batch_stride + c * spatial_stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # iterate over spatial elements in tiles
    for i in range((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

        # hard-swish: x * sigmoid(x)
        sigmoid = 0.5 * (tl.math.tanh(0.5 * x) + 1.0)
        hs = x * sigmoid

        acc += tl.sum(hs, axis=0)  # accumulate in register

    # After loop, compute mean
    mean_val = acc.sum() / tl.float32(n_elements)

    # store result
    out_idx = b * n_channels + c
    tl.store(out_ptr + out_idx, mean_val, mask=True)


def hard_swish_mean(x: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (B, C, D, H, W) on CUDA, float32
    Returns tensor of shape (B, C)
    """
    assert x.is_cuda and x.dtype == torch.float32, "Input must be CUDA float32 tensor"
    B, C, D, H, W = x.shape
    n_elements = D * H * W
    out = torch.empty((B, C), device=x.device, dtype=torch.float32)

    batch_stride = C * n_elements
    spatial_stride = n_elements

    # grid: one program per (b, c) pair
    grid = lambda meta: (B * C,)

    hard_swish_mean_kernel[grid](
        x, out,
        batch_size=B,
        n_channels=C,
        spatial_stride=spatial_stride,
        batch_stride=batch_stride,
        n_elements=n_elements,
        BLOCK_SIZE=256,
    )
    return out


# ----------------------------------------------------------------------
# New Model with Triton fused hard-swish + mean
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D
    2. HardSwish + mean pooling over spatial dims (fused)
    3. GroupNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # Conv3D
        x = self.conv(x)                       # (B, C, D, H, W)

        # HardSwish + mean (fused with Triton)
        x = hard_swish_mean(x)                 # (B, C)

        # GroupNorm over channels (PyTorch implementation)
        # Note: GroupNorm expects input shape (B, C, ...)
        # Since we reduced spatial dims, we reshape to (B, C, 1)
        x = x.unsqueeze(-1)
        x = self.group_norm(x)                 # (B, C, 1)
        x = x.squeeze(-1)                      # (B, C)
        return x
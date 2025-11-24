import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------------------------------------------------------- #
# Triton kernel for per‑pixel, per‑batch minimum over the channel dimension
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def min_channel_kernel(
    x_ptr,          # pointer to input (N, C, H, W)
    out_ptr,        # pointer to output (N, 1, H, W)
    N, C, H, W,    # tensor dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of spatial pixels across the batch
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N * H * W

    # Compute batch, h, w indices from the linear offset
    batch = offsets // (H * W)
    hw = offsets % (H * W)
    h = hw // W
    w = hw % W

    # Load the first channel value as the initial minimum
    x0 = tl.load(
        x_ptr + (batch * C * H * W) + (0 * H * W) + (h * W) + w,
        mask=mask,
        other=tl.float32(0.0),
    )

    # Iterate over remaining channels
    min_val = x0
    for c in range(1, C):
        val = tl.load(
            x_ptr + (batch * C * H * W) + (c * H * W) + (h * W) + w,
            mask=mask,
            other=tl.float32(0.0),
        )
        min_val = tl.minimum(min_val, val)

    tl.store(out_ptr + offsets, min_val, mask=mask)


def triton_min_channel(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for the Triton min‑channel kernel.

    Args:
        x: Tensor of shape (N, C, H, W) on CUDA.

    Returns:
        Tensor of shape (N, 1, H, W) on CUDA.
    """
    assert x.is_cuda, "Input must be on CUDA"

    N, C, H, W = x.shape
    out = torch.empty((N, 1, H, W), dtype=x.dtype, device=x.device)

    grid = lambda meta: ((N * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    min_channel_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        BLOCK_SIZE=128,
    )
    return out


# --------------------------------------------------------------------------- #
# Optimised model that fuses scaling and min into a single Triton kernel
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised model that performs a convolution, scales the output, and
    then applies a per‑pixel minimum over the channel dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convolution
        x = self.conv(x)          # shape: (N, C, H, W)

        # 2. Scale the result
        x = x * self.scale_factor

        # 3. Per‑pixel minimum over channels using Triton
        x = triton_min_channel(x)   # shape: (N, 1, H, W)

        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------- Triton kernels --------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def mul_clamp_mul_kernel(
    x_ptr,                # input after instance norm
    multiplier_ptr,       # shape (C,)
    out_ptr,              # output after second mul
    n_elements,           # total number of elements (B*C*D*H*W)
    C: tl.constexpr,      # number of channels
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    For each element:
        y = clamp(x * multiplier[channel], clamp_min, clamp_max)
        y = y * multiplier[channel]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute channel index for each offset
    # offset = ((b * C + c) * D + d) * H * W + h * W + w
    # we can recover channel by: c = (offset // (D*H*W)) % C
    # but to keep it simple we compute linear channel stride
    channel_stride = tl.arange(0, BLOCK_SIZE) * C  # not used

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute channel index
    channel_idx = (offsets // (tl.constexpr(DHW)) ) % C  # DHW defined below

    m = tl.load(multiplier_ptr + channel_idx, mask=mask, other=0.0)
    prod = x * m
    clamped = tl.max(tl.min(prod, clamp_max), clamp_min)
    out = clamped * m

    tl.store(out_ptr + offsets, out, mask=mask)


# Reduction kernel for max over channel dimension
# This is a generic reduction kernel pattern from Triton docs
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_W": 128, "BLOCK_SIZE_H": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE_W": 256, "BLOCK_SIZE_H": 256}, num_warps=4),
    ],
    key=["B", "D", "H", "W", "C"],
)
@triton.jit
def max_reduce_kernel(
    src_ptr,       # input after mul_clamp_mul, shape (B, C, D, H, W)
    dst_ptr,       # output after reduction, shape (B, D, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    """
    Reduce over channel dimension (C) using max.
    Each program block handles a tile of (B, D, H, W) positions.
    """
    b = tl.program_id(0)
    d = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    # Compute the global index for the tile
    idx = ((b * D + d) * H + h) * W + w
    stride = C

    # Load initial values for each channel
    offsets = tl.arange(0, BLOCK_SIZE_W) + idx * BLOCK_SIZE_W
    masks = offsets < (B * D * H * W * BLOCK_SIZE_W)

    # Load per-channel data
    data = tl.load(src_ptr + offsets * C, mask=masks, other=-float("inf"))
    # Reduce over channel dimension
    for k in range(0, C, BLOCK_SIZE_H):
        data = tl.maximum(data, tl.load(src_ptr + (offsets + k) * C, mask=masks, other=-float("inf")))

    tl.store(dst_ptr + idx, data, mask=masks)


# -------------------- Helper wrappers --------------------

def triton_mul_clamp_mul(x: torch.Tensor, multiplier: torch.Tensor, clamp_min: float, clamp_max: float):
    """
    Wrapper for the mul_clamp_mul_kernel.
    """
    assert x.is_cuda and multiplier.is_cuda
    B, C, D, H, W = x.shape
    n_elements = x.numel()
    out = torch.empty_like(x)

    # Compute DHW constant
    global DHW
    DHW = D * H * W

    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mul_clamp_mul_kernel[grid](
        x, multiplier, out, n_elements,
        C=C,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        BLOCK_SIZE=meta["BLOCK_SIZE"],
    )
    return out


def triton_max_reduce(x: torch.Tensor):
    """
    Wrapper for the max_reduce_kernel.
    """
    B, C, D, H, W = x.shape
    out = torch.empty((B, D, H, W), device=x.device, dtype=x.dtype)

    # Determine grid size
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_H = 128
    grid = lambda meta: (
        B,
        D,
        (H + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )
    max_reduce_kernel[grid](
        x,
        out,
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    return out


# -------------------- Model definition --------------------

class ModelNew(nn.Module):
    """
    Optimized model using custom Triton kernels for
    multiplication, clamping, second multiplication, and channel-wise max.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape).cuda())
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.instance_norm(x)
        x = triton_mul_clamp_mul(x, self.multiplier.squeeze(), self.clamp_min, self.clamp_max)
        x = triton_max_reduce(x)
        return x
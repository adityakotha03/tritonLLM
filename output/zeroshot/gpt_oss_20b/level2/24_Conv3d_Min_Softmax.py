import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------------
#  Triton kernel to compute minimum over the depth dimension
# ------------------------------------------------------------------
@triton.jit
def min_depth_kernel(
    x_ptr,          # (B, C, D, H, W) input
    out_ptr,        # (B, C, H, W) output
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # each program processes a contiguous block of (B*C*H*W) elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * C * H * W)

    # compute indices
    bc_hw = C * H * W
    b = offsets // bc_hw
    rem1 = offsets % bc_hw
    c = rem1 // (H * W)
    rem2 = rem1 % (H * W)
    h = rem2 // W
    w = rem2 % W

    # strides
    stride_b = C * D * H * W
    stride_c = D * H * W
    stride_h = D * W
    stride_w = D

    base_ptr = x_ptr + b * stride_b + c * stride_c + h * stride_h + w * stride_w

    # compute min over depth
    min_val = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32, mask=mask)
    for d in range(D):
        x = tl.load(base_ptr + d * stride_w, mask=mask, other=float('inf'))
        min_val = tl.minimum(min_val, x)

    out_ptr[offsets] = min_val


# ------------------------------------------------------------------
#  Triton kernel to compute softmax over the channel dimension
# ------------------------------------------------------------------
@triton.jit
def softmax_channel_kernel(
    x_ptr,          # (B, C, H, W) input
    out_ptr,        # (B, C, H, W) output
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # number of (b,h,w) handled by each block
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * H * W)

    # compute indices for (b,h,w)
    bw = H * W
    b = offsets // bw
    rem = offsets % bw
    h = rem // W
    w = rem % W

    # strides
    stride_b = C * H * W
    stride_h = C * W
    stride_w = C

    base_ptr = x_ptr + b * stride_b + h * stride_h + w * stride_w

    # load the entire channel vector for this (b,h,w)
    # vectorized load across channels
    vals = tl.load(base_ptr + tl.arange(0, C) * stride_w, mask=mask, other=0.0)

    # numerical stability: subtract max
    max_val = tl.max(vals)
    vals = vals - max_val

    exp_vals = tl.exp(vals)
    sum_exp = tl.sum(exp_vals)

    out = exp_vals / sum_exp

    tl.store(out_ptr + base_ptr, out, mask=mask)


# ------------------------------------------------------------------
#  Python wrappers for the Triton kernels
# ------------------------------------------------------------------
def triton_min_depth(x: torch.Tensor):
    B, C, D, H, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((B * C * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    min_depth_kernel[grid](
        x,
        out,
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_softmax_channel(x: torch.Tensor):
    B, C, H, W = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: ((B * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    softmax_channel_kernel[grid](
        x,
        out,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ------------------------------------------------------------------
#  Optimized model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, then fuses the minimum
    operation along the depth dimension and softmax over the channel dimension
    into custom Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim  # dimension along which to compute minimum (unused here)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        x = self.conv(x)                       # (B, C, D, H, W)
        x = triton_min_depth(x)                # min over depth -> (B, C, H, W)
        x = triton_softmax_channel(x)          # softmax over channels
        return x
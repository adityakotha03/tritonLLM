import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Triton kernel that fuses: logsumexp (over channel), hard‑swish, subtraction,
# and clamp. The kernel operates on the 5‑D tensor produced by ConvTranspose3d.
# --------------------------------------------------------------------------- #

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["N", "D", "H", "W", "C"],
)
@triton.jit
def fused_ops_kernel(
    x_ptr,          # pointer to the 5‑D input tensor (N, C, D, H, W)
    bias_ptr,       # pointer to the bias scalar (broadcast)
    out_ptr,        # pointer to the output tensor (N, 1, D, H, W)
    N: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one spatial location across all batches.
    # Compute a unique global index for each (n, d, h, w) triple.
    # Number of spatial positions per batch: D*H*W
    spatial_idx = tl.program_id(0)
    batch_idx = spatial_idx // (D * H * W)
    spatial_pos = spatial_idx % (D * H * W)

    d = spatial_pos // (H * W)
    h = (spatial_pos // W) % H
    w = spatial_pos % W

    # Base address for the selected (n, d, h, w) across all channels
    base = (batch_idx * C * D * H * W
            + d * H * W * C
            + h * W * C
            + w * C)

    # ----------------------------------------------------- #
    # 1. logsumexp over the channel dimension
    # ----------------------------------------------------- #
    # Compute max over channels first
    max_val = -1e9
    channel_idx = 0
    while channel_idx < C:
        offsets = base + channel_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + C
        vals = tl.load(x_ptr + offsets, mask=mask, other=-1e9)
        max_val = tl.maximum(max_val, tl.max(vals))
        channel_idx += BLOCK_SIZE

    # Broadcast max_val for all threads
    max_val = tl.broadcast_to(max_val, (BLOCK_SIZE,))

    # Compute sum of exp(x - max_val)
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    channel_idx = 0
    while channel_idx < C:
        offsets = base + channel_idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < base + C
        vals = tl.load(x_ptr + offsets, mask=mask, other=-1e9)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += exp_vals
        channel_idx += BLOCK_SIZE

    logsumexp = max_val + tl.log(tl.sum(sum_exp))

    # ----------------------------------------------------- #
    # 2. hard‑swish: x * sigmoid(x + 3) / 6
    # ----------------------------------------------------- #
    # Note: logsumexp is scalar per (n,d,h,w), so use it directly
    hswish = logsumexp * tl.sigmoid(logsumexp + 3.0) / 6.0

    # ----------------------------------------------------- #
    # 3. Subtract bias (broadcast)
    # ----------------------------------------------------- #
    bias = tl.load(bias_ptr)
    hswish = hswish - bias

    # ----------------------------------------------------- #
    # 4. Clamp to [-1, 1]
    # ----------------------------------------------------- #
    hswish = tl.clamp(hswish, min=-1.0, max=1.0)

    # ----------------------------------------------------- #
    # Store the result
    # ----------------------------------------------------- #
    out_offset = (batch_idx * D * H * W
                  + d * H * W
                  + h * W
                  + w)
    tl.store(out_ptr + out_offset, hswish, mask=tl.arange(0, BLOCK_SIZE) == 0)

# --------------------------------------------------------------------------- #
# Wrapper that launches the Triton kernel
# --------------------------------------------------------------------------- #

def fused_ops(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: Tensor of shape (N, C, D, H, W) on CUDA
    bias: Scalar Tensor on CUDA (broadcast)
    returns Tensor of shape (N, 1, D, H, W) on CUDA
    """
    N, C, D, H, W = x.shape
    out = torch.empty((N, 1, D, H, W), dtype=x.dtype, device=x.device)

    grid = lambda meta: ((N * D * H * W + meta["BLOCK_SIZE"] - 1) //
                         meta["BLOCK_SIZE"],)

    fused_ops_kernel[grid](x, bias, out, N, C, D, H, W, BLOCK_SIZE=256)
    return out

# --------------------------------------------------------------------------- #
# Optimized model
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom Triton kernel to fuse logsumexp,
    hard‑swish, bias subtraction, and clamp after a ConvTranspose3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1, device="cuda"))

    def forward(self, x):
        # 3D transposed convolution
        x = self.conv_transpose(x)                 # shape (N, C, D, H, W)

        # fused operations: logsumexp -> hard‑swish -> bias -> clamp
        x = fused_ops(x, self.bias)                # shape (N, 1, D, H, W)
        return x
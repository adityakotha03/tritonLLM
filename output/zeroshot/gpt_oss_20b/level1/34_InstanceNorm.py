import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def inorm_kernel(
    x_ptr,          # Input tensor [B*C, H*W]
    mean_ptr,       # Mean per (b,c) [B*C]
    var_ptr,        # Variance per (b,c) [B*C]
    out_ptr,        # Output tensor [B*C, H*W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalizes each (b, c) channel of the input tensor.

    Each program instance processes one channel (b, c).  
    The channel is flattened into a 1D array of length H*W.
    The kernel loops over this array in blocks of size BLOCK_SIZE.
    """
    # Compute the (b, c) indices for this program instance
    b = tl.program_id(0) // C
    c = tl.program_id(0) % C

    # Offset into the flattened array for this channel
    channel_offset = ((b * C + c) * H * W)
    N = H * W  # Number of elements in the channel

    # Load the pre‑computed mean and variance for this channel
    mean = tl.load(mean_ptr + b * C + c)
    var = tl.load(var_ptr + b * C + c)
    denom = tl.sqrt(var + eps)

    # Process the channel in chunks of BLOCK_SIZE
    for i in range((N + BLOCK_SIZE - 1) // BLOCK_SIZE):
        offset = channel_offset + i * BLOCK_SIZE
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < channel_offset + N

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) / denom
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_triton(x: torch.Tensor, eps: float = 1e-5):
    """
    Instance Normalization implemented with a custom Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    B, C, H, W = x.shape

    # Compute mean and variance per (b, c) using PyTorch (efficient on GPU)
    mean = x.mean(dim=(2, 3), keepdim=True)                 # shape: [B, C, 1, 1]
    var  = x.var(dim=(2, 3), unbiased=False, keepdim=True) # shape: [B, C, 1, 1]

    # Flatten tensors for the Triton kernel
    x_flat    = x.reshape(B * C, H * W).contiguous()
    mean_flat = mean.reshape(B * C).contiguous()
    var_flat  = var.reshape(B * C).contiguous()
    out_flat  = torch.empty_like(x_flat)

    # Kernel launch configuration
    num_channels = B * C
    BLOCK_SIZE   = 256                     # Tunable block size (power of 2)
    grid = lambda meta: (num_channels,)    # One program per channel

    # Launch the Triton kernel
    inorm_kernel[grid](
        x_flat, mean_flat, var_flat, out_flat,
        B, C, H, W,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back to the original 4‑D shape
    return out_flat.reshape(B, C, H, W)


class ModelNew(nn.Module):
    """
    Optimized Instance Normalization model using a custom Triton kernel.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instance_norm_triton(x, eps=self.eps)
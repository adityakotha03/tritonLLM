import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel that scales the feature map and performs global average
# pooling over the spatial dimensions (H, W).
# ----------------------------------------------------------------------
@triton.jit
def convscale_avg_kernel(
    x_ptr,            # Pointer to the conv_transpose output
    out_ptr,          # Pointer to the output (N, C, 1, 1)
    n_batches,        # Number of batches (N)
    n_channels,       # Number of channels (C)
    n_spatial,        # H * W
    multiplier,       # Scalar multiplier
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel) pair.
    batch = tl.program_id(0) // n_channels
    channel = tl.program_id(0) % n_channels

    # Base offset for the current (batch, channel) feature map
    base = (batch * n_channels + channel) * n_spatial

    # Accumulate sum over spatial dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the spatial tiles
    for i in range(0, n_spatial, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_spatial
        # Load a tile of the feature map
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        acc += x

    # Reduce across the block dimension
    acc_sum = tl.sum(acc)

    # Apply scalar multiplier and divide by number of spatial elements
    mean_val = (acc_sum * multiplier) / tl.float32(n_spatial)

    # Store the result at (batch, channel, 0, 0)
    out_ptr[batch * n_channels + channel] = mean_val


def convscale_avg(x: torch.Tensor, multiplier: float) -> torch.Tensor:
    """
    Wraps the Triton kernel for scaling and global average pooling.
    x: Tensor of shape (N, C, H, W) on CUDA.
    multiplier: Scalar multiplier.
    Returns tensor of shape (N, C, 1, 1).
    """
    assert x.is_cuda, "Input must be a CUDA tensor."
    n, c, h, w = x.shape
    n_spatial = h * w
    out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)

    # Each program handles one (batch, channel) pair
    grid = lambda meta: (n * c,)

    convscale_avg_kernel[grid](
        x, out, n, c, n_spatial, multiplier,
        BLOCK_SIZE=256
    )
    return out


# ----------------------------------------------------------------------
# Model with custom Triton kernel for the final scaling and pooling step.
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar,
    and applies global average pooling in a single Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)            # (N, C, H', W')
        x = convscale_avg(x, self.multiplier) # (N, C, 1, 1)
        return x
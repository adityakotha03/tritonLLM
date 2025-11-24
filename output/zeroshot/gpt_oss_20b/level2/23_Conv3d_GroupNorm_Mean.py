import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#                     Triton kernel for mean across all non-batch dims      #
# --------------------------------------------------------------------------- #
@triton.jit
def mean_kernel(
    x_ptr,            # Pointer to the input tensor
    out_ptr,          # Pointer to the output tensor
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # number of channels
    D: tl.constexpr,  # depth
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch element
    batch_idx = tl.program_id(0)
    # Compute the linear offset for this batch element
    batch_offset = batch_idx * C * D * H * W

    # Number of elements per batch
    num_elements = C * D * H * W

    # Sum accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over the elements in tiles of BLOCK_SIZE
    for i in range((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE):
        # Global offset for this tile
        offsets = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offsets < num_elements
        # Load a tile of the batch element
        vals = tl.load(
            x_ptr + batch_offset + offsets,
            mask=mask,
            other=0.0,
            dtype=tl.float32
        )
        # Accumulate
        acc += vals

    # Reduce across the BLOCK_SIZE elements
    sum_all = tl.sum(acc)

    # Normalize by number of elements
    mean = sum_all / tl.float32(num_elements)

    # Store result
    tl.store(out_ptr + batch_idx, mean)


def triton_mean(x: torch.Tensor):
    """
    Computes mean across all dimensions except batch using a Triton kernel.
    """
    assert x.is_cuda and x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 5, "Expected input of shape (B, C, D, H, W)."

    B, C, D, H, W = x.shape
    out = torch.empty((B,), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1024  # Tunable

    grid = lambda meta: (B,)

    mean_kernel[grid](
        x,
        out,
        B=B,
        C=C,
        D=D,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# --------------------------------------------------------------------------- #
#                               Optimized Model                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimized model that replaces the final mean operation with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.group_norm(x)
        # Replace x.mean(dim=[1,2,3,4]) with Triton kernel
        return triton_mean(x)
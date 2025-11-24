import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Triton kernel: fused MaxPool + Clamp
# ----------------------------------------------------------------------
@triton.jit
def maxpool_clamp_kernel(
    x_ptr,          # input pointer (N, C, H, W)
    out_ptr,        # output pointer (N, C, H_out, W_out)
    n_samples,      # batch size
    n_channels,     # number of channels
    H_in, W_in,     # input height / width
    H_out, W_out,   # output height / width
    pool_size,      # pooling kernel size (assumed square)
    stride,         # stride (assumed equal to pool_size)
    clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Compute total number of output elements
    total_out = n_samples * n_channels * H_out * W_out

    mask = offsets < total_out

    # Compute coordinates for each offset
    # index -> (n, c, h_out, w_out)
    idx_n = offsets // (n_channels * H_out * W_out)
    idx_rem = offsets % (n_channels * H_out * W_out)
    idx_c = idx_rem // (H_out * W_out)
    idx_rem2 = idx_rem % (H_out * W_out)
    idx_h = idx_rem2 // W_out
    idx_w = idx_rem2 % W_out

    # Compute input top-left corner for the pooling window
    h_start = idx_h * stride
    w_start = idx_w * stride

    # Load pooling window and compute max
    max_val = -1e30
    for i in range(pool_size):
        for j in range(pool_size):
            h_in = h_start + i
            w_in = w_start + j
            # Compute linear offset in input tensor
            in_offset = (
                idx_n * n_channels * H_in * W_in
                + idx_c * H_in * W_in
                + h_in * W_in
                + w_in
            )
            val = tl.load(x_ptr + in_offset, mask=mask, other=-1e30)
            max_val = tl.maximum(max_val, val)

    # Apply clamp
    max_val = tl.maximum(max_val, clamp_min)
    max_val = tl.minimum(max_val, clamp_max)

    # Store result
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_maxpool_clamp(
    x: torch.Tensor,
    pool_size: int,
    clamp_min: float,
    clamp_max: float,
    stride: int | None = None,
):
    """
    Triton implementation of MaxPool2d (kernel_size=pool_size, stride=stride)
    followed by torch.clamp(min=clamp_min, max=clamp_max).
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    n_samples, n_channels, H_in, W_in = x.shape
    stride = stride or pool_size
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    out = torch.empty((n_samples, n_channels, H_out, W_out), dtype=x.dtype, device=x.device)

    total_out = n_samples * n_channels * H_out * W_out
    BLOCK_SIZE = 256  # Tunable

    grid = lambda meta: ((total_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool_clamp_kernel[grid](
        x, out, n_samples, n_channels, H_in, W_in, H_out, W_out,
        pool_size, stride, clamp_min, clamp_max, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ----------------------------------------------------------------------
# New model definition
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling,
    fused MaxPool + Clamp using custom Triton kernels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_groups,
        scale_shape,
        maxpool_kernel_size,
        clamp_min,
        clamp_max,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = x * self.scale
        x = triton_maxpool_clamp(
            x,
            pool_size=self.maxpool_kernel_size,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            stride=self.maxpool_kernel_size,
        )
        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
#  Custom kernel: mean over spatial dimensions followed by tanh (fused)
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        triton.Config({},  {"BLOCK_SIZE": 128}),
        triton.Config({},  {"BLOCK_SIZE": 256}),
        triton.Config({},  {"BLOCK_SIZE": 512}),
    ],
    key=["N"],
)
@triton.jit
def mean_tanh_kernel(
    x_ptr,        # pointer to input [B, C, H, W]
    out_ptr,      # pointer to output [B, C, 1, 1]
    B, C, H, W,  # dimensions
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one (batch, channel) pair.
    """
    batch_idx = tl.program_id(0) // C
    chan_idx  = tl.program_id(0) %  C

    # Compute linear offset for the start of this channel slice
    # stride order: (B, C, H, W) -> stride_B = C*H*W, stride_C = H*W
    stride_B = C * H * W
    stride_C = H * W
    base_ptr = x_ptr + batch_idx * stride_B + chan_idx * stride_C

    # Number of spatial elements
    N = H * W
    # Iterate over N with block reduction
    sum = 0.0
    # The first thread in the block loads data for the entire N in steps of BLOCK_SIZE
    for i in range(0, N, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        vals = tl.load(base_ptr + offset, mask=mask, other=0.0)
        sum += tl.sum(vals, axis=0)

    # Compute mean and tanh
    mean = sum / tl.full((1,), N, dtype=tl.float32)
    out  = tl.math.tanh(mean)

    # Store result at [B, C, 0, 0]
    out_ptr[batch_idx * C + chan_idx] = out

# --------------------------------------------------------------------------- #
#  Helper function to invoke the kernel
# --------------------------------------------------------------------------- #
def mean_tanh(x: torch.Tensor):
    """
    x: 4-D tensor [B, C, H, W] on CUDA
    Returns: 4-D tensor [B, C, 1, 1]
    """
    assert x.is_cuda and x.is_contiguous()
    B, C, H, W = x.shape
    out = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)

    # Launch kernel
    grid = lambda meta: (B * C,)
    mean_tanh_kernel[grid](x, out, B, C, H, W, BLOCK_SIZE=256)
    return out

# --------------------------------------------------------------------------- #
#  New model definition
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling,
    hardtanh activation, then a fused mean+ tanh operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(
            min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = self.hardtanh(x)
        x = mean_tanh(x)          # fused mean + tanh
        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl

# --------------------- Triton kernel: logsumexp + ReLU --------------------- #
@triton.jit
def logsumexp_relu_kernel(
    x_ptr,          # Pointer to input tensor (B, C, D, H, W)
    out_ptr,        # Pointer to output tensor (B, 1, D, H, W)
    batch_size,
    channels,
    depth,
    height,
    width,
    stride_d,       # Stride for depth dimension in flattened pointer
    stride_h,       # Stride for height dimension
    stride_w,       # Stride for width dimension
    stride_c,       # Stride for channel dimension
    stride_b,       # Stride for batch dimension
    BLOCK_SIZE: tl.constexpr,
):
    # Total number of spatial positions per batch
    total_spatial = depth * height * width
    # Each program handles BLOCK_SIZE spatial positions
    idx = tl.program_id(0) * BLOCK_SIZE
    offsets = idx + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds
    mask = offsets < total_spatial

    # Compute spatial coordinates from flat index
    z = offsets % depth
    y = (offsets // depth) % height
    x = offsets // (depth * height)

    # Prepare pointers for each batch
    for b in range(batch_size):
        base_ptr = x_ptr + b * stride_b

        # Load the first channel to init max and sum
        max_val = tl.load(base_ptr + z * stride_d + y * stride_h + x * stride_w, mask=mask, other=-float("inf"))
        # Initialize sum with exp(0)=1
        sum_exp = tl.full((BLOCK_SIZE,), 1.0, tl.float32)

        # Iterate over remaining channels
        for c in range(1, channels):
            val = tl.load(
                base_ptr
                + c * stride_c
                + z * stride_d
                + y * stride_h
                + x * stride_w,
                mask=mask,
                other=0.0,
            )
            # Update max
            max_val = tl.maximum(max_val, val)
            # Update sum exp
            sum_exp = sum_exp + tl.exp(val - max_val)

        # Final logsumexp
        lse = max_val + tl.log(sum_exp)

        # Apply ReLU
        out = tl.maximum(lse, 0.0)

        # Store result
        out_base = out_ptr + b * stride_b
        tl.store(out_base + z * stride_d + y * stride_h + x * stride_w, out, mask=mask)

# --------------------- Wrapper function --------------------- #
def logsumexp_relu_torch(x: torch.Tensor):
    """
    x: Tensor of shape (B, C, D, H, W) on CUDA
    Returns tensor of shape (B, 1, D, H, W)
    """
    assert x.is_cuda, "Input must be on CUDA"
    B, C, D, H, W = x.shape
    out = torch.empty((B, 1, D, H, W), dtype=x.dtype, device=x.device)

    # Strides in elements
    stride_b = x.stride(0)
    stride_c = x.stride(1)
    stride_d = x.stride(2)
    stride_h = x.stride(3)
    stride_w = x.stride(4)

    # Block size tuning
    BLOCK_SIZE = 1024

    grid = lambda meta: ((D * H * W + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    logsumexp_relu_kernel[grid](
        x,
        out,
        B,
        C,
        D,
        H,
        W,
        stride_d,
        stride_h,
        stride_w,
        stride_c,
        stride_b,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

# --------------------- New Model --------------------- #
class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, max pooling, logsumexp, and ReLU activation
    with the logsumexp+ReLU fused into a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, depth', height', width')
        """
        x = self.conv(x)            # (B, C, D, H, W)
        x = self.max_pool(x)        # (B, C, D/2, H/2, W/2)
        x = logsumexp_relu_torch(x) # (B, 1, D/2, H/2, W/2)
        return x
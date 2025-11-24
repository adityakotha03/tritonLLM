import torch
import torch.nn as nn
import triton
import triton.language as tl

# ----------------------------------------------------------------------
# Elementwise scale kernel
# ----------------------------------------------------------------------
@triton.jit
def _scale_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    scale = tl.load(scale_ptr, mask=mask, other=1.0)  # scale is scalar
    out = x * scale
    tl.store(out_ptr + offset, out, mask=mask)

def scale_tensor(x: torch.Tensor, scale: torch.Tensor):
    """Elementwise scaling using Triton."""
    assert x.is_cuda and scale.is_cuda, "Inputs must be on CUDA."
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _scale_kernel[grid](x, scale, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ----------------------------------------------------------------------
# Bias addition kernel (bias is broadcasted over batch, depth, height, width)
# ----------------------------------------------------------------------
@triton.jit
def _bias_add_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    dim0,  # out_channels
    dim1,  # depth
    dim2,  # height
    dim3,  # width
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    # compute the channel index for each element
    # offset layout: [N, C, D, H, W] -> channel stride = D*H*W
    stride_c = dim1 * dim2 * dim3
    channel_idx = (offset // stride_c) % dim0

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offset, out, mask=mask)

def add_bias(x: torch.Tensor, bias: torch.Tensor):
    """Add bias with broadcasting using Triton."""
    assert x.is_cuda and bias.is_cuda, "Inputs must be on CUDA."
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    # bias shape is (C,1,1,1)
    dim0 = bias.shape[0]
    dim1 = bias.shape[1] if bias.ndim > 1 else 1
    dim2 = bias.shape[2] if bias.ndim > 2 else 1
    dim3 = bias.shape[3] if bias.ndim > 3 else 1
    _bias_add_kernel[grid](
        x, bias, out, n_elements, dim0, dim1, dim2, dim3, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ----------------------------------------------------------------------
# AvgPool3d kernel (kernel size 2, stride 2, no padding)
# ----------------------------------------------------------------------
@triton.jit
def _avg_pool_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    stride_d, stride_h, stride_w,
    pool_d, pool_h, pool_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offset < n_elements

    # Compute output spatial indices
    out_idx = out_offset
    stride_out = stride_d * stride_h * stride_w
    idx_w = out_idx % stride_w
    idx_h = (out_idx // stride_w) % stride_h
    idx_d = (out_idx // (stride_w * stride_h)) % stride_d

    # Load 8 elements (2x2x2) per output
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for zd in range(pool_d):
        for zh in range(pool_h):
            for zw in range(pool_w):
                in_idx = (
                    (idx_d * stride_d + zd) * stride_h * stride_w
                    + (idx_h * stride_h + zh) * stride_w
                    + (idx_w * stride_w + zw)
                )
                val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
                sum_val += val

    avg = sum_val / (pool_d * pool_h * pool_w)
    tl.store(out_ptr + out_offset, avg, mask=mask)

def avg_pool3d(x: torch.Tensor):
    """3D average pooling with kernel=2, stride=2 using Triton."""
    assert x.is_cuda, "Input must be on CUDA."
    # Input shape: [B, C, D, H, W]
    B, C, D, H, W = x.shape
    out_D = D // 2
    out_H = H // 2
    out_W = W // 2
    out = torch.empty((B, C, out_D, out_H, out_W), device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _avg_pool_kernel[grid](
        x, out, n_elements,
        stride_d=2, stride_h=2, stride_w=2,
        pool_d=2, pool_h=2, pool_w=2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# ----------------------------------------------------------------------
# Full model with fused custom kernels
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model: 3D transposed conv + scaling + avgpool + bias + scaling.
    The conv layer is kept as PyTorch kernel; scaling, avgpool, bias addition are
    replaced by Triton kernels to reduce memory traffic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 scale1, scale2, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32, device="cuda"))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32, device="cuda"))
        self.bias = nn.Parameter(torch.randn(bias_shape, device="cuda"))

    def forward(self, x):
        # ConvTranspose3d
        x = self.conv_transpose(x)

        # Scale1
        x = scale_tensor(x, self.scale1)

        # AvgPool3d
        x = avg_pool3d(x)

        # Bias addition
        x = add_bias(x, self.bias)

        # Scale2
        x = scale_tensor(x, self.scale2)
        return x
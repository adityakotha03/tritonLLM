import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X_ptr,  # Pointer to input tensor (batch, in_channels, height, width)
    W_ptr,  # Pointer to kernel weights (out_channels, in_channels, kernel_h, kernel_w)
    Y_ptr,  # Pointer to output tensor (batch, out_channels, out_h, out_w)
    batch_size, in_channels, out_channels, height, width, out_h, out_w,
    kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_C: tl.constexpr,
):
    # Thread block dimensions
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Block offsets
    block_start_h = pid_h * TILE_H
    block_start_w = pid_w * TILE_W
    block_start_c = pid_c * TILE_C

    # Thread offsets within the block
    offs_h = tl.arange(0, TILE_H)
    offs_w = tl.arange(0, TILE_W)
    offs_c = tl.arange(0, TILE_C)

    # Compute output indices
    output_h = block_start_h + offs_h
    output_w = block_start_w + offs_w
    output_c = block_start_c + offs_c

    # Compute valid output bounds
    mask_h = output_h < out_h
    mask_w = output_w < out_w
    mask_c = output_c < out_channels

    # Create combined mask for output
    out_mask = mask_h[:, None] & mask_w[None, :] & mask_c[None, None]

    # Calculate global input coordinates
    # Input spatial coordinates in the original input
    in_h = (output_h // stride_h) + (output_h % stride_h) * dilation_h
    in_w = (output_w // stride_w) + (output_w % stride_w) * dilation_w

    # Apply padding
    in_h = in_h - padding_h
    in_w = in_w - padding_w

    # Check if valid input coordinates
    valid_h = (in_h >= 0) & (in_h < height)
    valid_w = (in_w >= 0) & (in_w < width)
    valid = valid_h[:, None, None] & valid_w[None, :, None]

    # Combine valid and output masks
    valid_mask = out_mask & valid

    # Prepare output accumulator
    acc = tl.zeros((TILE_H, TILE_W, TILE_C), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for ki in range(0, in_channels, 1):
        # Load kernel tile
        k_start_h = 0
        k_start_w = 0
        k_h = tl.arange(0, kernel_h)
        k_w = tl.arange(0, kernel_w)

        # Get kernel offsets
        kernel_h_offset = k_h * (kernel_w * in_channels + 1)
        kernel_w_offset = k_w

        # Load kernel weights: (out_c, in_c, k_h, k_w) -> (out_c, k_h, k_w)
        kernel = tl.load(
            W_ptr + pid_c * (in_channels * kernel_h * kernel_w) +
            ki * (kernel_h * kernel_w) +
            k_h[:, None] * kernel_w +
            k_w[None, :],
            mask=(k_h[:, None] < kernel_h) & (k_w[None, :] < kernel_w),
            other=0.0
        )

        # Load input tile
        # input: (batch, in_c, height, width)
        input_ptr = X_ptr + pid_batch * (in_channels * height * width) + ki * (height * width)
        input = tl.load(
            input_ptr + (in_h[:, None, None] * width + in_w[None, :, None]) * 1,
            mask=valid,
            other=0.0
        )

        # Perform dot product: (TILE_H, TILE_W) x (kernel_h, kernel_w) -> (TILE_H, TILE_W)
        # We're fusing conv with reduction over kernel and input channel
        # Since kernel size is small (3x3), we can use a direct reduction loop
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input position at kernel offset
                kh_offset = kh * dilation_h
                kw_offset = kw * dilation_w
                in_h_k = in_h - kh_offset
                in_w_k = in_w - kw_offset
                # Compute input value
                input_val = tl.load(
                    input_ptr + (in_h_k[:, None, None] * width + in_w_k[None, :, None]) * 1,
                    mask=valid & (in_h_k >= 0) & (in_h_k < height) & (in_w_k >= 0) & (in_w_k < width),
                    other=0.0
                )
                # Multiply by kernel value and accumulate
                acc += input_val * kernel[kh, kw]

    # Store output
    out_ptr = Y_ptr + pid_batch * (out_channels * out_h * out_w) + pid_c * (out_h * out_w)
    tl.store(
        out_ptr + (output_h[:, None, None] * out_w + output_w[None, :, None]) * 1,
        acc,
        mask=valid_mask
    )


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    Custom Triton-based 2D convolution with fusion of all operations.
    Uses block-based tiling and shared memory for input and kernel tiling.
    Optimized for A100 using FP16/BF16 with Tensor Cores when possible.
    """
    assert x.is_cuda and w.is_cuda, "Input tensors must be on CUDA"
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16, "Only FP16/BF16 supported for tensor core optimization"
    
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = w.shape
    assert out_channels % groups == 0, "out_channels must be divisible by groups"
    assert in_channels % groups == 0, "in_channels must be divisible by groups"

    # Compute output dimensions
    out_h = (height + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    out_w = (width + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1

    # Prepare output
    y = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Grid configuration
    num_tiles_c = triton.cdiv(out_channels, 32)  # TILE_C = 32
    num_tiles_h = triton.cdiv(out_h, 16)  # TILE_H = 16
    num_tiles_w = triton.cdiv(out_w, 16)  # TILE_W = 16
    num_tiles_batch = batch_size

    grid = (num_tiles_batch, num_tiles_c, num_tiles_h, num_tiles_w)

    # Launch kernel
    conv2d_kernel[
        grid,
        num_stages=4,
        num_warps=8,
    ](
        x, w, y,
        batch_size, in_channels, out_channels, height, width, out_h, out_w,
        kernel_h, kernel_w, stride, stride, padding, padding, dilation, dilation,
        BLOCK_SIZE=128,
        TILE_H=16,
        TILE_W=16,
        TILE_C=32,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        # Use Xavier initialization
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size).cuda())
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels).cuda())
        else:
            self.bias_param = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for convolution
        y = triton_conv2d(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
        
        # Add bias if needed
        if self.bias_param is not None:
            y = y + self.bias_param.view(1, -1, 1, 1)
            
        return y
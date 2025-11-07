import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr,
    bias_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    bias_stride,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, groups, num_groups,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Block offsets
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_c * BLOCK_C

    # Output dimensions
    out_h = (height - 1) * stride + kernel_size
    out_w = (width - 1) * stride + kernel_size

    # Compute output indices
    out_h_start = block_h
    out_w_start = block_w
    out_c_start = block_c

    # Ensure output bounds
    out_h_end = tl.minimum(out_h_start + BLOCK_H, out_h)
    out_w_end = tl.minimum(out_w_start + BLOCK_W, out_w)
    out_c_end = tl.minimum(out_c_start + BLOCK_C, out_channels)

    # Load output region
    offs_h = tl.arange(0, BLOCK_H)
    offs_w = tl.arange(0, BLOCK_W)
    offs_c = tl.arange(0, BLOCK_C)
    
    # Create masks
    mask_h = (out_h_start + offs_h) < out_h
    mask_w = (out_w_start + offs_w) < out_w
    mask_c = (out_c_start + offs_c) < out_channels

    # Construct output offsets
    out_offsets = (pid_batch * out_stride0 +
                   (out_h_start + offs_h) * out_stride2 +
                   (out_w_start + offs_w) * out_stride3 +
                   (out_c_start + offs_c) * out_stride1)

    # Initialize output with zeros
    out = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Load input and weights
    # Input is (B, C_in, H, W)
    in_h_start = (out_h_start // stride)
    in_w_start = (out_w_start // stride)
    in_h_end = (out_h_end - 1) // stride + 1
    in_w_end = (out_w_end - 1) // stride + 1

    in_h = tl.arange(0, in_h_end - in_h_start)
    in_w = tl.arange(0, in_w_end - in_w_start)
    in_c = tl.arange(0, in_channels)

    # Load input tiles (B, C_in, H, W)
    in_offsets = (pid_batch * x_stride0 +
                  (in_h_start + in_h) * x_stride2 +
                  (in_w_start + in_w) * x_stride3 +
                  in_c * x_stride1)
    in_tiles = tl.load(x_ptr + in_offsets, mask=(in_h[:, None, None] < (in_h_end - in_h_start)) &
                                       (in_w[None, :, None] < (in_w_end - in_w_start)) &
                                       (in_c[None, None, :] < in_channels), other=0.0)

    # Load kernel tiles (out_c, in_c, k_h, k_w)
    k_h_start = (out_h_start - in_h_start * stride)
    k_w_start = (out_w_start - in_w_start * stride)
    k_h_end = k_h_start + kernel_size
    k_w_end = k_w_start + kernel_size

    k_h = tl.arange(0, k_h_end - k_h_start)
    k_w = tl.arange(0, k_w_end - k_w_start)
    k_c = tl.arange(0, out_channels)

    # Load kernel tiles (out_c, in_c, k_h, k_w)
    kernel_offsets = (k_c[:, None, None, None] * w_stride0 +
                      in_c[None, :, None, None] * w_stride1 +
                      k_h[None, None, :, None] * w_stride2 +
                      k_w[None, None, None, :] * w_stride3)
    kernel_tiles = tl.load(w_ptr + kernel_offsets, mask=(k_h[None, None, :, None] < kernel_size) &
                                                          (k_w[None, None, None, :] < kernel_size) &
                                                          (in_c[None, :, None, None] < in_channels) &
                                                          (k_c[:, None, None, None] < out_channels), other=0.0)

    # Broadcast kernel for groups
    group_size = out_channels // groups
    group_id = pid_c // group_size
    group_offset = group_id * in_channels

    # Tile loop: perform conv_transpose
    for i in range(in_channels // BLOCK_K):
        in_c_idx = i * BLOCK_K + tl.arange(0, BLOCK_K)
        in_c_mask = in_c_idx < in_channels

        # Load input sub-tiles
        in_sub = tl.load(x_ptr + in_offsets + (in_c_idx[None, None, :] * x_stride1), 
                         mask=in_c_mask[None, None, :], other=0.0)
        
        # Load kernel sub-tiles
        kernel_sub = kernel_tiles * tl.load(w_ptr + kernel_offsets + (in_c_idx[None, :, None, None] * w_stride1), 
                                            mask=in_c_mask[None, :, None, None], other=0.0)

        # Compute inner product over input channels
        out += tl.dot(in_sub, kernel_sub)

    # Apply bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + (out_c_start + offs_c) * bias_stride, mask=mask_c, other=0.0)
        out += bias[None, None, :]

    # Store output
    tl.store(out_ptr + out_offsets, out, mask=(mask_h[:, None, None] & mask_w[None, :, None] & mask_c[None, None, :]))


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x2 = x * x
    x3 = x * x2
    coeff = 0.044715
    tanh_input = x * (0.7978845608 * (1 + coeff * x2))
    tanh_val = tl.tanh(tanh_input)
    out = x * 0.5 * (1 + tanh_val)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    batch_size, 
    num_channels,
    height,
    width,
    num_groups,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * num_channels * height * width)

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute group index
    c = (offsets // (height * width)) % num_channels
    g = c // (num_channels // num_groups)
    group_start = g * (num_channels // num_groups) + (offsets // (height * width)) % (num_channels // num_groups)

    # Load mean and rstd for the group
    mean = tl.load(mean_ptr + g, mask=g < num_groups, other=0.0)
    rstd = tl.load(rstd_ptr + g, mask=g < num_groups, other=0.0)

    # Normalize
    x = (x - mean) * rstd

    # Store
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_conv_transpose(x, w, bias, stride, groups):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]
    kernel_size = w.shape[2]

    # Compute output shape
    out_h = (height - 1) * stride + kernel_size
    out_w = (width - 1) * stride + kernel_size

    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Calculate strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = w.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()
    bias_stride = bias.stride(0) if bias is not None else 0

    # Define block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32
    BLOCK_K = 16

    # Grid dimensions
    grid = (
        batch_size,
        (out_h + BLOCK_H - 1) // BLOCK_H,
        (out_w + BLOCK_W - 1) // BLOCK_W,
        (out_channels + BLOCK_C - 1) // BLOCK_C
    )

    # Launch kernel
    conv_transpose_kernel[grid](
        x_ptr=x, w_ptr=w, out_ptr=out, bias_ptr=bias,
        x_stride0=x_stride0, x_stride1=x_stride1, x_stride2=x_stride2, x_stride3=x_stride3,
        w_stride0=w_stride0, w_stride1=w_stride1, w_stride2=w_stride2, w_stride3=w_stride3,
        out_stride0=out_stride0, out_stride1=out_stride1, out_stride2=out_stride2, out_stride3=out_stride3,
        bias_stride=bias_stride,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        height=height, width=width, kernel_size=kernel_size, stride=stride, groups=groups,
        num_groups=8,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K
    )
    return out


def triton_gelu(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    gelu_kernel[grid](x_ptr=x, out_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_group_norm(x, mean, rstd, num_groups):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    group_norm_kernel[grid](
        x_ptr=x, out_ptr=out, mean_ptr=mean, rstd_ptr=rstd,
        batch_size=x.shape[0], num_channels=x.shape[1], height=x.shape[2], width=x.shape[3],
        num_groups=num_groups, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        # Apply custom Triton-based ConvTranspose2d
        x = triton_conv_transpose(x, self.conv_transpose.weight, self.conv_transpose.bias, stride=1, groups=8)
        # Apply GELU via Triton
        x = triton_gelu(x)
        # Apply GroupNorm via Triton
        mean = self.group_norm.running_mean
        rstd = self.group_norm.running_var
        rstd = torch.rsqrt(rstd + self.group_norm.eps)
        x = triton_group_norm(x, mean, rstd, num_groups=8)
        return x
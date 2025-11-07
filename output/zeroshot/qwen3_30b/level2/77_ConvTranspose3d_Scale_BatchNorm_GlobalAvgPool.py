import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr, 
    batch_size, 
    in_channels, 
    out_channels, 
    depth, 
    height, 
    width, 
    kernel_size, 
    stride, 
    padding, 
    BLOCK_D: tl.constexpr, 
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr, 
    BLOCK_C_IN: tl.constexpr, 
    BLOCK_C_OUT: tl.constexpr,
):
    # Block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c_out = tl.program_id(3)

    # Compute output indices
    d_start = pid_d * BLOCK_D
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    c_out_start = pid_c_out * BLOCK_C_OUT

    # Create offsets
    offs_d = d_start + tl.arange(0, BLOCK_D)
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    offs_c_in = tl.arange(0, BLOCK_C_IN)
    offs_c_out = c_out_start + tl.arange(0, BLOCK_C_OUT)

    # Mask for bounds checking
    d_mask = offs_d < depth
    h_mask = offs_h < height
    w_mask = offs_w < width
    c_out_mask = offs_c_out < out_channels
    mask = d_mask[:, None, None, None] & h_mask[:, None, None] & w_mask[:, None] & c_out_mask[None, None, None, :]

    # Load input
    x = tl.load(x_ptr + (offs_d[:, None, None] * height * width + offs_h[None, :, None] * width + offs_w[None, None, :]) * in_channels + offs_c_in[None, None, :], mask=mask, other=0.0)

    # Load weights
    w = tl.load(w_ptr + (offs_c_out[None, None, None, :] * in_channels * kernel_size**3 + offs_c_in[None, None, :, None] * kernel_size**3 + (offs_d[:, None, None, None] - 2) * kernel_size**2 + (offs_h[None, :, None, None] - 2) * kernel_size + (offs_w[None, None, :, None] - 2)), mask=mask, other=0.0)

    # Perform convolution (transposed)
    acc = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_C_OUT), dtype=tl.float32)
    for kd in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input offset
                in_d = offs_d[:, None, None] - kd + 2
                in_h = offs_h[None, :, None] - kh + 2
                in_w = offs_w[None, None, :] - kw + 2
                # Check bounds
                valid = (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
                # Load input
                x_val = tl.load(x_ptr + (in_d[:, None, None] * height * width + in_h[None, :, None] * width + in_w[None, None, :]) * in_channels + offs_c_in[None, None, :], mask=valid, other=0.0)
                # Multiply with weight
                w_val = w[offs_c_out[None, None, None, :] * in_channels * kernel_size**3 + offs_c_in[None, None, :, None] * kernel_size**3 + (kd * kernel_size**2 + kh * kernel_size + kw)]
                acc += x_val[:, :, :, None] * w_val[None, None, :, None]
    # Store output
    tl.store(out_ptr + (offs_d[:, None, None] * height * width + offs_h[None, :, None] * width + offs_w[None, None, :]) * out_channels + offs_c_out[None, None, None, :], acc, mask=mask)


@triton.jit
def scale_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
    var = tl.load(var_ptr + offsets, mask=mask, other=0.0)
    out = (x - mean) / (tl.sqrt(var + eps))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_channels,
    depth,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Assume x is (B, C, D, H, W)
    # We want to reduce over D, H, W to get (B, C, 1, 1, 1)
    pid = tl.program_id(0)
    c = pid % out_channels
    b = pid // out_channels

    # Each block computes one output channel for one batch
    c_start = c * BLOCK_SIZE
    offsets = c_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_channels

    # Load all elements for this batch and channel
    x = tl.load(x_ptr + (b * out_channels * depth * height * width + c * depth * height * width + tl.arange(0, depth)[:, None, None] * height * width + tl.arange(0, height)[None, :, None] * width + tl.arange(0, width)[None, None, :]) * out_channels + offsets[None, None, :])  # Adjust indexing

    # Compute mean
    mean = tl.sum(x, axis=(0, 1, 2)) / (depth * height * width)

    # Store result
    tl.store(out_ptr + (b * out_channels + c), mean, mask=mask)


def triton_conv_transpose_3d(x, w, out, batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride=1, padding=0):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty(batch_size, out_channels, depth, height, width, device=x.device, dtype=x.dtype)

    # Define block sizes
    BLOCK_D = 8
    BLOCK_H = 8
    BLOCK_W = 8
    BLOCK_C_IN = 8
    BLOCK_C_OUT = 8

    # Grid dimensions
    grid_d = (depth + BLOCK_D - 1) // BLOCK_D
    grid_h = (height + BLOCK_H - 1) // BLOCK_H
    grid_w = (width + BLOCK_W - 1) // BLOCK_W
    grid_c_out = (out_channels + BLOCK_C_OUT - 1) // BLOCK_C_OUT

    # Launch kernel
    conv_transpose_3d_kernel[
        (grid_d, grid_h, grid_w, grid_c_out)
    ](
        x, w, out, batch_size, in_channels, out_channels, depth, height, width, kernel_size, stride, padding,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C_IN=BLOCK_C_IN, BLOCK_C_OUT=BLOCK_C_OUT
    )
    return out


def triton_scale(x, scale):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    scale_kernel[grid](x, out, n_elements, scale, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_batch_norm(x, mean, var, eps):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    batch_norm_kernel[grid](x, mean, var, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_global_avg_pool(x, batch_size, out_channels, depth, height, width):
    x = x.contiguous()
    out = torch.empty(batch_size, out_channels, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 32
    grid = lambda meta: (batch_size * out_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    global_avg_pool_kernel[grid](x, out, batch_size, out_channels, depth, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # Apply custom Triton kernels
        x = triton_conv_transpose_3d(
            x,
            self.conv_transpose.weight,
            None,
            batch_size=x.shape[0],
            in_channels=self.conv_transpose.in_channels,
            out_channels=self.conv_transpose.out_channels,
            depth=x.shape[2],
            height=x.shape[3],
            width=x.shape[4],
            kernel_size=self.conv_transpose.kernel_size[0],
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
        )

        x = triton_scale(x, self.scale_factor)

        # BatchNorm is trickier: we need mean/var from running stats
        # We'll use triton_batch_norm for the forward pass, using stored stats
        mean = self.batch_norm.running_mean.view(1, -1, 1, 1, 1).to(x.device)
        var = self.batch_norm.running_var.view(1, -1, 1, 1, 1).to(x.device)
        x = triton_batch_norm(x, mean, var, self.batch_norm.eps)

        x = triton_global_avg_pool(x, batch_size=x.shape[0], out_channels=x.shape[1], depth=x.shape[2], height=x.shape[3], width=x.shape[4])

        # Reshape to (B, C, 1, 1, 1)
        return x.view(x.shape[0], x.shape[1], 1, 1, 1)
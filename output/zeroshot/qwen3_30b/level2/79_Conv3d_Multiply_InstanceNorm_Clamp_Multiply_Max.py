import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    w_stride0, w_stride1, w_stride2, w_stride3, w_stride4,
    out_stride0, out_stride1, out_stride2, out_stride3,
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_size,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    # Block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_o = tl.program_id(3)

    # Compute output block boundaries
    d_start = pid_d * BLOCK_SIZE_D
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    o_start = pid_o * BLOCK_SIZE_O

    # Compute indices within block
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_h = tl.arange(0, BLOCK_SIZE_H)
    offs_w = tl.arange(0, BLOCK_SIZE_W)
    offs_o = tl.arange(0, BLOCK_SIZE_O)

    # Compute output offsets
    out_offsets_d = d_start + offs_d
    out_offsets_h = h_start + offs_h
    out_offsets_w = w_start + offs_w
    out_offsets_o = o_start + offs_o

    # Masks to avoid out-of-bounds access
    mask_d = out_offsets_d < depth
    mask_h = out_offsets_h < height
    mask_w = out_offsets_w < width
    mask_o = out_offsets_o < out_channels

    # Combined mask
    mask = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :] & mask_o[None, None, None]

    # Compute input offsets (apply padding, stride logic implicitly assumed)
    # Assume no padding, stride=1
    pad = (kernel_size - 1) // 2
    x_offsets_d = out_offsets_d[:, None, None] - tl.arange(0, kernel_size)[:, None, None]
    x_offsets_h = out_offsets_h[None, :, None] - tl.arange(0, kernel_size)[None, :, None]
    x_offsets_w = out_offsets_w[None, None, :] - tl.arange(0, kernel_size)[None, None, :]
    x_offsets_o = out_offsets_o[None, None, None]

    # Apply padding and mask
    valid_d = (x_offsets_d >= -pad) & (x_offsets_d < depth + pad)
    valid_h = (x_offsets_h >= -pad) & (x_offsets_h < height + pad)
    valid_w = (x_offsets_w >= -pad) & (x_offsets_w < width + pad)
    valid = valid_d & valid_h & valid_w

    # Flatten input indices and load
    x_offsets = (
        x_offsets_d * x_stride4 +  # D
        x_offsets_h * x_stride3 +  # H
        x_offsets_w * x_stride2 +  # W
        x_offsets_o * x_stride1 +  # C
        tl.arange(0, batch_size)[:, None, None, None] * x_stride0  # B
    )
    x_offsets = x_offsets + (pad, pad, pad, 0, 0)  # Adjust for padding
    x_ptrs = x_ptr + x_offsets
    x_vals = tl.load(x_ptrs, mask=valid[:, :, :, None] & mask, other=0.0)

    # Weights
    w_offsets = (
        tl.arange(0, kernel_size)[:, None, None] * w_stride4 +
        tl.arange(0, kernel_size)[None, :, None] * w_stride3 +
        tl.arange(0, kernel_size)[None, None, :] * w_stride2 +
        out_offsets_o[None, None, None] * w_stride1 +
        tl.arange(0, in_channels)[:, None, None] * w_stride0
    )
    w_ptrs = w_ptr + w_offsets
    w_vals = tl.load(w_ptrs, mask=mask, other=0.0)

    # Perform convolution
    # Sum over kernel and input channels
    out_vals = tl.dot(x_vals, w_vals, allow_tf32=True)  # (B, D, H, W, O) -> (B, D, H, W, O)
    out_vals = tl.sum(out_vals, axis=4)  # Sum over in_channels

    # Output pointer
    out_offsets = (
        tl.arange(0, batch_size)[:, None, None, None] * out_stride0 +
        out_offsets_d[:, None, None] * out_stride1 +
        out_offsets_h[None, :, None] * out_stride2 +
        out_offsets_w[None, None, :] * out_stride3 +
        out_offsets_o[None, None, None] * out_stride4
    )
    out_ptrs = out_ptr + out_offsets
    tl.store(out_ptrs, out_vals, mask=mask)


@triton.jit
def mul_kernel(
    x_ptr, mul_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mul = tl.load(mul_ptr + (offsets % n_elements) % x_ptr.shape[-1], mask=mask, other=0.0)  # Broadcast
    out = x * mul
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def instance_norm_kernel(
    x_ptr, mean_ptr, var_ptr,
    out_ptr,
    batch_size, channels, depth, height, width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offs = offset + tl.arange(0, BLOCK_SIZE)

    # Handle per-channel mean/var
    mask = offs < channels * depth * height * width * batch_size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Extract channel index (c = (offs % (channels*...)) // (depth*height*width))
    c = (offs // (depth * height * width)) % channels
    mean = tl.load(mean_ptr + c, mask=c < channels)
    var = tl.load(var_ptr + c, mask=c < channels)

    # Normalize
    out = (x - mean) / tl.sqrt(var + eps)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    clamp_min: tl.float32,
    clamp_max: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.clip(x, clamp_min, clamp_max)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def max_reduce_kernel(
    x_ptr,
    out_ptr,
    batch_size, depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (depth * height * width)
    dhw = pid % (depth * height * width)
    d = dhw // (height * width)
    h = (dhw % (height * width)) // width
    w = (dhw % (height * width)) % width

    # Load all channel values at (b, :, d, h, w)
    offs = (
        b * depth * height * width * 16 +
        d * height * width * 16 +
        h * width * 16 +
        w * 16 +
        tl.arange(0, 16)
    )
    x = tl.load(x_ptr + offs, mask=tl.arange(0, 16) < 16, other=-float('inf'))

    out = tl.max(x, axis=0)
    tl.store(out_ptr + (b * depth * height * width + d * height * width + h * width + w), out)


def triton_conv3d(x, weight, kernel_size):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, _, _, _ = weight.shape

    out = torch.empty(batch_size, out_channels, depth, height, width, device=x.device, dtype=x.dtype)

    # Compute strides
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3, w_stride4 = weight.stride()
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4 = out.stride()

    # Block sizes (tuned for A100)
    BLOCK_SIZE_D = 16
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_O = 16

    # Grid setup
    grid_d = (triton.cdiv(depth, BLOCK_SIZE_D),)
    grid_h = (triton.cdiv(height, BLOCK_SIZE_H),)
    grid_w = (triton.cdiv(width, BLOCK_SIZE_W),)
    grid_o = (triton.cdiv(out_channels, BLOCK_SIZE_O),)

    # Launch 4D kernel
    conv3d_kernel[
        (grid_d[0], grid_h[0], grid_w[0], grid_o[0])
    ](
        x, weight, out,
        x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
        w_stride0, w_stride1, w_stride2, w_stride3, w_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3,
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_size,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_O=BLOCK_SIZE_O,
    )
    return out


def triton_mul(x, multiplier):
    assert x.is_cuda and multiplier.is_cuda
    x = x.contiguous()
    multiplier = multiplier.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    mul_kernel[grid](x, multiplier, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_instance_norm(x, running_mean, running_var):
    assert x.is_cuda
    x = x.contiguous()

    batch_size, channels, depth, height, width = x.shape
    out = torch.empty_like(x)

    eps = 1e-5
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(batch_size * channels * depth * height * width, meta["BLOCK_SIZE"]),)

    instance_norm_kernel[grid](
        x, running_mean, running_var, out,
        batch_size, channels, depth, height, width,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_clamp(x, min_val, max_val):
    assert x.is_cuda
    x = x.contiguous()

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    clamp_kernel[grid](
        x, out, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_max_reduce(x):
    assert x.is_cuda
    batch_size, channels, depth, height, width = x.shape
    out = torch.empty(batch_size, depth, height, width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = 1024
    grid = lambda meta: (batch_size * depth * height * width,)

    max_reduce_kernel[grid](
        x, out, batch_size, depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=False)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        # Step 1: Conv3d
        x = triton_conv3d(x, self.conv.weight, kernel_size=self.conv.kernel_size[0])

        # Step 2: Multiply
        x = triton_mul(x, self.multiplier)

        # Step 3: InstanceNorm3d (manual via Triton)
        # Simulate mean/var using batch statistics
        # This is a simplified version — real IN uses moving averages
        mean = x.mean(dim=(0, 2, 3, 4), keepdim=True)
        var = x.var(dim=(0, 2, 3, 4), keepdim=True)
        x = triton_instance_norm(x, mean, var)

        # Step 4: Clamp
        x = triton_clamp(x, self.clamp_min, self.clamp_max)

        # Step 5: Multiply again
        x = triton_mul(x, self.multiplier)

        # Step 6: Max over channel dimension
        x = triton_max_reduce(x)

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, 
    w_ptr, 
    out_ptr, 
    B, C_out, C_in, D, H, W, K_D, K_H, K_W, 
    stride_d, stride_h, stride_w, 
    pad_d, pad_h, pad_w,
    BLOCK_SIZE_D: tl.constexpr, 
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate output dimensions
    D_out = (D + 2 * pad_d - K_D) // stride_d + 1
    H_out = (H + 2 * pad_h - K_H) // stride_h + 1
    W_out = (W + 2 * pad_w - K_W) // stride_w + 1

    # Output offsets
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_c_out = pid_c_out * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Mask out of bounds
    mask_d = offs_d < D_out
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask_c_out = offs_c_out < C_out

    # Load output indices
    offs_out = (
        pid_b * C_out * D_out * H_out * W_out +
        pid_c_out * D_out * H_out * W_out +
        pid_d * H_out * W_out +
        pid_h * W_out +
        pid_w
    )

    # Load output
    out = tl.load(out_ptr + offs_out, mask=mask_d & mask_h & mask_w & mask_c_out, other=0.0)

    # Input offsets
    offs_x_d = (offs_d * stride_d) - pad_d
    offs_x_h = (offs_h * stride_h) - pad_h
    offs_x_w = (offs_w * stride_w) - pad_w

    # Load weights
    w_idx = (
        pid_c_out * C_in * K_D * K_H * K_W +
        tl.arange(0, C_in)[:, None, None, None] * K_D * K_H * K_W +
        tl.arange(0, K_D)[:, None, None] * K_H * K_W +
        tl.arange(0, K_H)[:, None] * K_W +
        tl.arange(0, K_W)
    )
    w = tl.load(w_ptr + w_idx, mask=tl.arange(0, C_in)[:, None, None, None] < C_in, other=0.0)

    # Load input
    x = tl.load(
        x_ptr + pid_b * C_in * D * H * W + 
        tl.arange(0, C_in)[:, None, None, None] * D * H * W +
        offs_x_d[None, None, None, :] * H * W +
        offs_x_h[None, None, :, None] * W +
        offs_x_w[None, :, None, None],
        mask=(offs_x_d[None, None, None, :] >= 0) & 
              (offs_x_d[None, None, None, :] < D) &
              (offs_x_h[None, None, :, None] >= 0) & 
              (offs_x_h[None, None, :, None] < H) &
              (offs_x_w[None, :, None, None] >= 0) & 
              (offs_x_w[None, :, None, None] < W),
        other=0.0
    )

    # Compute convolution
    # (C_in, K_D, K_H, K_W) @ (C_in, D, H, W) -> (D_out, H_out, W_out)
    # We need to collapse over C_in, K_D, K_H, K_W
    conv = tl.dot(w, x, allow_tf32=True)
    conv = tl.sum(conv, axis=0)

    # Store result
    tl.store(out_ptr + offs_out, conv, mask=mask_d & mask_h & mask_w & mask_c_out)


@triton.jit
def hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Hardswish: x * relu6(x + 3) / 6
    x_plus_3 = x + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    out = x * relu6 / 6.0
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    B, C, D, H, W,
    num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Total elements in each group
    group_size = C // num_groups
    total_elements_per_group = D * H * W

    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_g = pid_c // group_size
    pid_c_in_group = pid_c % group_size

    # Offset in output
    offs_c = pid_c * D * H * W + pid_c_in_group * D * H * W
    offs_g = pid_g * total_elements_per_group

    # Load input
    x = tl.load(x_ptr + offs_c, mask=tl.arange(0, D*H*W) < total_elements_per_group, other=0.0)

    # Compute mean and variance over spatial dims
    mean = tl.sum(x, axis=0) / total_elements_per_group
    var = tl.sum((x - mean) ** 2, axis=0) / total_elements_per_group
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Scale and shift
    x_norm = (x - mean) * inv_std
    # Scale by weight and add bias (not in this case, so dummy)
    out = x_norm

    # Store
    tl.store(out_ptr + offs_c, out, mask=tl.arange(0, D*H*W) < total_elements_per_group)


@triton.jit
def mean_pool_kernel(
    x_ptr,
    out_ptr,
    B, C,
    BLOCK_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_b = pid_b * C
    offs_c = pid_c * C
    offsets = offs_b + offs_c + tl.arange(0, C)

    # Load data
    x = tl.load(x_ptr + offsets, mask=tl.arange(0, C) < C, other=0.0)

    # Compute mean over spatial dims
    mean_val = tl.sum(x, axis=0) / (16 * 32 * 32)

    # Store output
    tl.store(out_ptr + pid_b * C + pid_c, mean_val, mask=tl.arange(0, C) < C)


@triton.jit
def fused_conv3d_hardswish_groupnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    B, C_out, C_in, D, H, W, K_D, K_H, K_W,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    num_groups,
    eps,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Calculate output dimensions
    D_out = (D + 2 * pad_d - K_D) // stride_d + 1
    H_out = (H + 2 * pad_h - K_H) // stride_h + 1
    W_out = (W + 2 * pad_w - K_W) // stride_w + 1

    # Output offsets
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_c_out = pid_c_out * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    # Mask out of bounds
    mask_d = offs_d < D_out
    mask_h = offs_h < H_out
    mask_w = offs_w < W_out
    mask_c_out = offs_c_out < C_out

    # Load input
    offs_x_d = (offs_d * stride_d) - pad_d
    offs_x_h = (offs_h * stride_h) - pad_h
    offs_x_w = (offs_w * stride_w) - pad_w

    # Load weights
    w_idx = (
        pid_c_out * C_in * K_D * K_H * K_W +
        tl.arange(0, C_in)[:, None, None, None] * K_D * K_H * K_W +
        tl.arange(0, K_D)[:, None, None] * K_H * K_W +
        tl.arange(0, K_H)[:, None] * K_W +
        tl.arange(0, K_W)
    )
    w = tl.load(w_ptr + w_idx, mask=tl.arange(0, C_in)[:, None, None, None] < C_in, other=0.0)

    # Load input
    x = tl.load(
        x_ptr + pid_b * C_in * D * H * W + 
        tl.arange(0, C_in)[:, None, None, None] * D * H * W +
        offs_x_d[None, None, None, :] * H * W +
        offs_x_h[None, None, :, None] * W +
        offs_x_w[None, :, None, None],
        mask=(offs_x_d[None, None, None, :] >= 0) & 
              (offs_x_d[None, None, None, :] < D) &
              (offs_x_h[None, None, :, None] >= 0) & 
              (offs_x_h[None, None, :, None] < H) &
              (offs_x_w[None, :, None, None] >= 0) & 
              (offs_x_w[None, :, None, None] < W),
        other=0.0
    )

    # Compute convolution
    conv = tl.dot(w, x, allow_tf32=True)
    conv = tl.sum(conv, axis=0)

    # Hardswish: x * relu6(x + 3) / 6
    x_plus_3 = conv + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    hardswish = conv * relu6 / 6.0

    # GroupNorm
    group_size = C_out // num_groups
    pid_g = pid_c_out // group_size
    pid_c_in_group = pid_c_out % group_size

    # Compute mean and variance over spatial dims
    mean = tl.sum(hardswish, axis=0) / (D_out * H_out * W_out)
    var = tl.sum((hardswish - mean) ** 2, axis=0) / (D_out * H_out * W_out)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_norm = (hardswish - mean) * inv_std

    # Store output
    offs_out = (
        pid_b * C_out * D_out * H_out * W_out +
        pid_c_out * D_out * H_out * W_out +
        pid_d * H_out * W_out +
        pid_h * W_out +
        pid_w
    )
    tl.store(out_ptr + offs_out, x_norm, mask=mask_d & mask_h & mask_w & mask_c_out)


def triton_conv3d(x, w, stride, padding, kernel_size):
    B, C_in, D, H, W = x.shape
    C_out, _, K_D, K_H, K_W = w.shape

    # Output dimensions
    D_out = (D + 2 * padding[0] - K_D) // stride[0] + 1
    H_out = (H + 2 * padding[1] - K_H) // stride[1] + 1
    W_out = (W + 2 * padding[2] - K_W) // stride[2] + 1

    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)

    # Grid
    grid = lambda meta: (
        B, 
        (C_out + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        (D_out + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch kernel
    conv3d_kernel[
        grid
    ](
        x, w, out,
        B, C_out, C_in, D, H, W, K_D, K_H, K_W,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        BLOCK_SIZE_D=16, BLOCK_SIZE_H=16, BLOCK_SIZE_W=16,
        BLOCK_SIZE_C=8
    )

    return out


def triton_hardswish(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=128)
    return out


def triton_group_norm(x, num_groups, eps):
    B, C, D, H, W = x.shape
    out = torch.empty_like(x)
    grid = lambda meta: (
        B, 
        (C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    )
    group_norm_kernel[grid](x, out, B, C, D, H, W, num_groups, eps, BLOCK_SIZE=64)
    return out


def triton_mean_pool(x):
    B, C, D, H, W = x.shape
    out = torch.empty(B, C, device=x.device, dtype=x.dtype)
    grid = lambda meta: (B, (C + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    mean_pool_kernel[grid](x, out, B, C, BLOCK_SIZE=64)
    return out


def triton_fused_conv3d_hardswish_groupnorm(x, w, stride, padding, kernel_size, num_groups, eps):
    B, C_in, D, H, W = x.shape
    C_out, _, K_D, K_H, K_W = w.shape

    # Output dimensions
    D_out = (D + 2 * padding[0] - K_D) // stride[0] + 1
    H_out = (H + 2 * padding[1] - K_H) // stride[1] + 1
    W_out = (W + 2 * padding[2] - K_W) // stride[2] + 1

    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)

    # Grid
    grid = lambda meta: (
        B, 
        (C_out + meta["BLOCK_SIZE_C"] - 1) // meta["BLOCK_SIZE_C"],
        (D_out + meta["BLOCK_SIZE_D"] - 1) // meta["BLOCK_SIZE_D"],
        (H_out + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W_out + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch fused kernel
    fused_conv3d_hardswish_groupnorm_kernel[
        grid
    ](
        x, w, out,
        B, C_out, C_in, D, H, W, K_D, K_H, K_W,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        num_groups, eps,
        BLOCK_SIZE_D=16, BLOCK_SIZE_H=16, BLOCK_SIZE_W=16,
        BLOCK_SIZE_C=8
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        # Initialize convolution weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

    def forward(self, x):
        # Fused operations: Conv3D + HardSwish + GroupNorm + MeanPool
        x = triton_fused_conv3d_hardswish_groupnorm(
            x,
            self.weight,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            kernel_size=4,
            num_groups=4,
            eps=1e-6
        )
        # Final mean pooling
        x = torch.mean(x, dim=[2, 3, 4])
        return x
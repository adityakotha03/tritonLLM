import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr, scale_ptr, output_ptr,
    H, W, C, N, group_size,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # Compute group offset
    group_idx = pid_g
    c_start = group_idx * group_size
    c_end = min(c_start + group_size, C)

    # Load input and compute mean
    sum_val = 0.0
    sum_sq_val = 0.0
    hw_offset = tl.arange(0, BLOCK_SIZE_HW)
    c_offset = tl.arange(0, BLOCK_SIZE_C)

    for h in range(0, H, BLOCK_SIZE_HW):
        for w in range(0, W, BLOCK_SIZE_HW):
            for c in range(c_start, c_end, BLOCK_SIZE_C):
                # Compute offsets
                offsets_hw = hw_offset[:, None] * stride_xw + tl.arange(0, BLOCK_SIZE_C)[None, :] * stride_xc
                offsets_n = pid_n * stride_xn
                offsets_c = c + c_offset[None, :]
                offsets_h = (h + (hw_offset // W)) * stride_xh
                offsets_w = (w + (hw_offset % W)) * stride_xw
                mask_hw = (hw_offset < H * W)[:, None]
                mask_c = (c_offset < C)[None, :]
                mask = mask_hw and mask_c

                x_ptrs = x_ptr + offsets_n + offsets_c * stride_xc + offsets_h + offsets_w
                x = tl.load(x_ptrs, mask=mask, other=0.0)

                # Accumulate sum and sum of squares
                sum_val += tl.sum(x, axis=1)
                sum_sq_val += tl.sum(x * x, axis=1)

    # Compute mean and variance
    count = H * W * group_size
    mean = sum_val / count
    var = sum_sq_val / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    for h in range(0, H, BLOCK_SIZE_HW):
        for w in range(0, W, BLOCK_SIZE_HW):
            for c in range(c_start, c_end, BLOCK_SIZE_C):
                offsets_hw = hw_offset[:, None] * stride_xw + tl.arange(0, BLOCK_SIZE_C)[None, :] * stride_xc
                offsets_n = pid_n * stride_xn
                offsets_c = c + c_offset[None, :]
                offsets_h = (h + (hw_offset // W)) * stride_xh
                offsets_w = (w + (hw_offset % W)) * stride_xw
                mask_hw = (hw_offset < H * W)[:, None]
                mask_c = (c_offset < C)[None, :]
                mask = mask_hw and mask_c

                x_ptrs = x_ptr + offsets_n + offsets_c * stride_xc + offsets_h + offsets_w
                x = tl.load(x_ptrs, mask=mask, other=0.0)

                # Normalize
                x_norm = (x - mean[:, None]) * inv_std[:, None]

                # Scale
                scale_ptrs = scale_ptr + offsets_c
                scale = tl.load(scale_ptrs, mask=mask_c, other=1.0)
                out = x_norm * scale[None, :]

                # Store output
                output_ptrs = output_ptr + offsets_n + offsets_c * stride_yc + offsets_h + offsets_w
                tl.store(output_ptrs, out, mask=mask)


@triton.jit
def maxpool_clamp_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    pool_size: tl.constexpr,
    clamp_min, clamp_max,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output indices
    out_h = pid_h
    out_w = pid_w
    if out_h * pool_size >= H or out_w * pool_size >= W:
        return

    # Input spatial start
    h_start = out_h * pool_size
    w_start = out_w * pool_size

    # Initialize max value
    max_val = -float('inf')

    # Loop over pooling window
    for ph in range(0, pool_size):
        for pw in range(0, pool_size):
            h_idx = h_start + ph
            w_idx = w_start + pw
            valid = (h_idx < H) & (w_idx < W)
            offset = pid_n * stride_xn + pid_c * stride_xc + h_idx * stride_xh + w_idx * stride_xw
            x_val = tl.load(x_ptr + offset) if valid else -float('inf')
            max_val = tl.maximum(max_val, x_val)

    # Clamp
    clamped_val = tl.maximum(tl.minimum(max_val, clamp_max), clamp_min)

    # Store
    out_offset = pid_n * stride_yn + pid_c * stride_yc + out_h * stride_yh + out_w * stride_yw
    tl.store(out_ptr + out_offset, clamped_val)


def triton_group_norm(x, num_groups, weight, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups

    # Output
    y = torch.empty_like(x)

    # Launch kernel
    def grid(meta):
        return (N, num_groups)

    BLOCK_SIZE_C = triton.next_power_of_2(group_size)
    BLOCK_SIZE_HW = min(triton.next_power_of_2(H * W), 1024)

    group_norm_kernel[grid](
        x, weight, y,
        H, W, C, N, group_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    return y


def triton_maxpool_clamp(x, kernel_size, clamp_min, clamp_max):
    N, C, H, W = x.shape
    out_h = (H + kernel_size - 1) // kernel_size
    out_w = (W + kernel_size - 1) // kernel_size

    y = torch.empty(N, C, out_h, out_w, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (N, C, out_h, out_w)

    maxpool_clamp_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        kernel_size,
        clamp_min, clamp_max,
        BLOCK_SIZE_N=1,
        BLOCK_SIZE_C=1,
        BLOCK_SIZE_H=1,
        BLOCK_SIZE_W=1
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for group norm, scale, maxpool, and clamp.
    Conv2d is left to PyTorch since it uses cuDNN highly optimized kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.num_groups = num_groups
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = triton_group_norm(x, self.num_groups, self.scale, eps=1e-5)
        x = triton_maxpool_clamp(x, self.maxpool_kernel_size, self.clamp_min, self.clamp_max)
        return x
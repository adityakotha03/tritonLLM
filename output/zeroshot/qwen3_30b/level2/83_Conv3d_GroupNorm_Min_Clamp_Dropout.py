import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch.nn import functional as F
import math


@triton.jit
def conv3d_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3, input_stride4,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4,
    output_stride0, output_stride1, output_stride2, output_stride3, output_stride4,
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    GROUPS: tl.constexpr,
    SEED: tl.constexpr
):
    # Calculate indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Block offsets
    block_d = pid_d * BLOCK_D
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_c * BLOCK_OUT_C

    # Offset into output
    output_offsets = block_d * output_stride2 + block_h * output_stride3 + block_w * output_stride4 + block_c * output_stride1

    # Load output tile
    output = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_OUT_C), dtype=tl.float32)

    # Compute output for each input channel and kernel
    for ci in range(0, in_channels, BLOCK_OUT_C):
        # Load input tile
        input_offset = pid_b * input_stride0 + ci * input_stride1
        input_tile = tl.load(
            input_ptr + input_offset + block_d * input_stride2 + block_h * input_stride3 + block_w * input_stride4,
            mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                 (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                 (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w),
            other=0.0
        )

        # Load kernel tile
        weight_offset = (ci // GROUPS) * weight_stride0 + (block_c // GROUPS) * weight_stride1 + \
                        (block_d % kernel_d) * weight_stride2 + (block_h % kernel_h) * weight_stride3 + \
                        (block_w % kernel_w) * weight_stride4
        weight_tile = tl.load(
            weight_ptr + weight_offset,
            mask=(tl.arange(0, BLOCK_D)[:, None, None] < kernel_d) &
                 (tl.arange(0, BLOCK_H)[None, :, None] < kernel_h) &
                 (tl.arange(0, BLOCK_W)[None, None, :] < kernel_w),
            other=0.0
        )

        # Perform 3D convolution
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute input offset for kernel application
                    i_d = block_d + kd
                    i_h = block_h + kh
                    i_w = block_w + kw
                    if i_d < depth and i_h < height and i_w < width:
                        input_slice = tl.load(
                            input_ptr + pid_b * input_stride0 + ci * input_stride1 +
                            i_d * input_stride2 + i_h * input_stride3 + i_w * input_stride4,
                            mask=(tl.arange(0, BLOCK_OUT_C) < out_channels),
                            other=0.0
                        )
                        output += input_slice[None, None, None, :] * weight_tile[kd, kh, kw, None, None, None]
        
        # Update output with channel-wise output
        output += tl.sum(output, axis=(0, 1, 2))  # Sum over spatial dims

    # Store output
    tl.store(output_ptr + output_offsets, output, mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                                                (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                                                (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
                                                (tl.arange(0, BLOCK_OUT_C)[None, None, :] < out_channels))


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
    batch_size, channels, depth, height, width,
    GROUPS: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d = pid_d * BLOCK_D
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_g * BLOCK_C

    # Load input
    x_offsets = pid_b * x_stride0 + block_c * x_stride1 + block_d * x_stride2 + block_h * x_stride3 + block_w * x_stride4
    x_tile = tl.load(
        x_ptr + x_offsets,
        mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
             (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
             (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
             (tl.arange(0, BLOCK_C)[None, None, :] < channels // GROUPS),
        other=0.0
    )

    # Compute mean and variance across group
    x_mean = tl.sum(x_tile, axis=(0, 1, 2)) / (BLOCK_D * BLOCK_H * BLOCK_W)
    x_mean = tl.broadcast(x_mean, (BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_C))
    x_var = tl.sum((x_tile - x_mean) ** 2, axis=(0, 1, 2)) / (BLOCK_D * BLOCK_H * BLOCK_W)

    # Normalize
    x_norm = (x_tile - x_mean) / (tl.sqrt(x_var + 1e-6))
    x_norm = x_norm * tl.load(gamma_ptr + block_c) + tl.load(beta_ptr + block_c)

    # Store output
    out_offsets = pid_b * out_stride0 + block_c * out_stride1 + block_d * out_stride2 + block_h * out_stride3 + block_w * out_stride4
    tl.store(out_ptr + out_offsets, x_norm, mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                                              (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                                              (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
                                              (tl.arange(0, BLOCK_C)[None, None, :] < channels // GROUPS))


@triton.jit
def clamp_min_kernel(
    x_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
    batch_size, channels, depth, height, width,
    MIN_VALUE: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d = pid_d * BLOCK_D
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_c * BLOCK_C

    x_offsets = pid_b * x_stride0 + block_c * x_stride1 + block_d * x_stride2 + block_h * x_stride3 + block_w * x_stride4
    out_offsets = pid_b * out_stride0 + block_c * out_stride1 + block_d * out_stride2 + block_h * out_stride3 + block_w * out_stride4

    x_tile = tl.load(
        x_ptr + x_offsets,
        mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
             (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
             (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
             (tl.arange(0, BLOCK_C)[None, None, :] < channels),
        other=0.0
    )

    out_tile = tl.maximum(x_tile, tl.full(x_tile.shape, MIN_VALUE, dtype=x_tile.dtype))
    tl.store(out_ptr + out_offsets, out_tile, mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                                                (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                                                (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
                                                (tl.arange(0, BLOCK_C)[None, None, :] < channels))


@triton.jit
def dropout_kernel(
    x_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
    batch_size, channels, depth, height, width,
    DROPOUT_P: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d = pid_d * BLOCK_D
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_c * BLOCK_C

    # Generate random number per element
    offset = (pid_b * batch_size + pid_c * channels + pid_d * depth + pid_h * height + pid_w * width) * 512
    xorshifted = (SEED ^ (SEED << 13)) ^ (SEED >> 17) ^ (SEED << 5)
    rng_state = xorshifted
    random_val = (rng_state & 0x7fffffff) / (2**31 - 1)

    x_offsets = pid_b * x_stride0 + block_c * x_stride1 + block_d * x_stride2 + block_h * x_stride3 + block_w * x_stride4
    out_offsets = pid_b * out_stride0 + block_c * out_stride1 + block_d * out_stride2 + block_h * out_stride3 + block_w * out_stride4

    x_tile = tl.load(
        x_ptr + x_offsets,
        mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
             (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
             (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
             (tl.arange(0, BLOCK_C)[None, None, :] < channels),
        other=0.0
    )

    # Apply dropout
    keep_mask = random_val > DROPOUT_P
    out_tile = tl.where(keep_mask, x_tile / (1 - DROPOUT_P), 0.0)

    tl.store(out_ptr + out_offsets, out_tile, mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                                                (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                                                (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
                                                (tl.arange(0, BLOCK_C)[None, None, :] < channels))


@triton.jit
def fused_conv3d_groupnorm_clamp_dropout_kernel(
    input_ptr, weight_ptr, gamma_ptr, beta_ptr,
    input_stride0, input_stride1, input_stride2, input_stride3, input_stride4,
    weight_stride0, weight_stride1, weight_stride2, weight_stride3, weight_stride4,
    gamma_stride0, beta_stride0,
    output_ptr,
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    GROUPS: tl.constexpr,
    MIN_VALUE: tl.constexpr,
    MAX_VALUE: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    SEED: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
    BLOCK_OUT_C: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    block_d = pid_d * BLOCK_D
    block_h = pid_h * BLOCK_H
    block_w = pid_w * BLOCK_W
    block_c = pid_g * BLOCK_C

    # Initialize output tile
    output_tile = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    # Convolution
    for ci in range(0, in_channels, BLOCK_OUT_C):
        # Load input
        input_offset = pid_b * input_stride0 + ci * input_stride1
        input_tile = tl.load(
            input_ptr + input_offset + block_d * input_stride2 + block_h * input_stride3 + block_w * input_stride4,
            mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                 (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                 (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w),
            other=0.0
        )

        # Load kernel
        weight_offset = (ci // GROUPS) * weight_stride0 + (block_c // GROUPS) * weight_stride1 + \
                        (block_d % kernel_d) * weight_stride2 + (block_h % kernel_h) * weight_stride3 + \
                        (block_w % kernel_w) * weight_stride4
        weight_tile = tl.load(
            weight_ptr + weight_offset,
            mask=(tl.arange(0, BLOCK_D)[:, None, None] < kernel_d) &
                 (tl.arange(0, BLOCK_H)[None, :, None] < kernel_h) &
                 (tl.arange(0, BLOCK_W)[None, None, :] < kernel_w),
            other=0.0
        )

        # Conv
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    i_d = block_d + kd
                    i_h = block_h + kh
                    i_w = block_w + kw
                    if i_d < depth and i_h < height and i_w < width:
                        x_slice = tl.load(
                            input_ptr + pid_b * input_stride0 + ci * input_stride1 +
                            i_d * input_stride2 + i_h * input_stride3 + i_w * input_stride4,
                            mask=(tl.arange(0, BLOCK_C) < out_channels),
                            other=0.0
                        )
                        output_tile += x_slice[None, None, None, :] * weight_tile[kd, kh, kw, None, None, None]

    # Normalize
    output_tile = output_tile / (BLOCK_D * BLOCK_H * BLOCK_W)
    mean = tl.sum(output_tile, axis=(0, 1, 2)) / (BLOCK_D * BLOCK_H * BLOCK_W)
    var = tl.sum((output_tile - mean[None, None, None, :]) ** 2, axis=(0, 1, 2)) / (BLOCK_D * BLOCK_H * BLOCK_W)

    # Clamp min
    output_tile = tl.maximum(output_tile, tl.full(output_tile.shape, MIN_VALUE, dtype=output_tile.dtype))

    # Clamp max
    output_tile = tl.minimum(output_tile, tl.full(output_tile.shape, MAX_VALUE, dtype=output_tile.dtype))

    # Dropout
    offset = (pid_b * batch_size + pid_g * out_channels + pid_d * depth + pid_h * height + pid_w * width) * 512
    xorshifted = (SEED ^ (SEED << 13)) ^ (SEED >> 17) ^ (SEED << 5)
    rng_state = xorshifted
    random_val = (rng_state & 0x7fffffff) / (2**31 - 1)
    keep_mask = random_val > DROPOUT_P
    output_tile = tl.where(keep_mask, output_tile / (1 - DROPOUT_P), 0.0)

    # Store output
    output_offsets = pid_b * input_stride0 + block_c * input_stride1 + block_d * input_stride2 + block_h * input_stride3 + block_w * input_stride4
    tl.store(output_ptr + output_offsets, output_tile, mask=(tl.arange(0, BLOCK_D)[:, None, None] < depth - block_d) &
                                                        (tl.arange(0, BLOCK_H)[None, :, None] < height - block_h) &
                                                        (tl.arange(0, BLOCK_W)[None, None, :] < width - block_w) &
                                                        (tl.arange(0, BLOCK_C)[None, None, :] < out_channels))


def triton_fused_forward(
    input_tensor, weight_tensor, gamma_tensor, beta_tensor,
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_d, kernel_h, kernel_w,
    groups, min_value, max_value, dropout_p, seed
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda and gamma_tensor.is_cuda and beta_tensor.is_cuda
    assert input_tensor.dtype == torch.bfloat16
    assert weight_tensor.dtype == torch.bfloat16
    assert gamma_tensor.dtype == torch.bfloat16
    assert beta_tensor.dtype == torch.bfloat16

    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    gamma_tensor = gamma_tensor.contiguous()
    beta_tensor = beta_tensor.contiguous()

    output_tensor = torch.empty_like(input_tensor, dtype=torch.bfloat16)

    # Determine grid size
    BLOCK_D, BLOCK_H, BLOCK_W = 16, 16, 16
    BLOCK_OUT_C, BLOCK_C = 16, 16

    num_groups = out_channels // groups
    grid = lambda meta: (
        batch_size, num_groups, 
        math.ceil(depth / meta["BLOCK_D"]),
        math.ceil(height / meta["BLOCK_H"]),
        math.ceil(width / meta["BLOCK_W"])
    )

    fused_conv3d_groupnorm_clamp_dropout_kernel[grid](
        input_tensor, weight_tensor, gamma_tensor, beta_tensor,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3), input_tensor.stride(4),
        weight_tensor.stride(0), weight_tensor.stride(1), weight_tensor.stride(2), weight_tensor.stride(3), weight_tensor.stride(4),
        gamma_tensor.stride(0), beta_tensor.stride(0),
        output_tensor,
        batch_size, in_channels, out_channels, depth, height, width,
        kernel_d, kernel_h, kernel_w,
        groups, min_value, max_value, dropout_p, seed,
        BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_OUT_C=BLOCK_OUT_C, BLOCK_C=BLOCK_C
    )

    return output_tensor


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Use Triton fused kernel
        return triton_fused_forward(
            x, self.weight, self.gamma, self.beta,
            x.size(0), self.in_channels, self.out_channels,
            x.size(2), x.size(3), x.size(4),
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.groups, self.min_value, self.max_value, self.dropout_p, 12345
        )
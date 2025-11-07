import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    depth, height, width,
    out_depth, out_height, out_width,
    kernel_size, stride, padding,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr, BLOCK_SIZE_IC: tl.constexpr,
    TILE_D: tl.constexpr, TILE_H: tl.constexpr, TILE_W: tl.constexpr,
    TILE_OC: tl.constexpr, TILE_IC: tl.constexpr,
):
    # Thread block indices
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Compute output indices
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_c = pid_c * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)

    # Mask for output bounds
    mask_d = out_d < out_depth
    mask_h = out_h < out_height
    mask_w = out_w < out_width
    mask_c = out_c < out_channels

    # Compute input indices (via deconvolution: output -> input)
    inp_d = (out_d - padding) * stride
    inp_h = (out_h - padding) * stride
    inp_w = (out_w - padding) * stride

    # Create offset for input (in the input space)
    inp_d = inp_d[:, None, None]  # [BLOCK_SIZE_D, 1, 1]
    inp_h = inp_h[None, :, None]  # [1, BLOCK_SIZE_H, 1]
    inp_w = inp_w[None, None, :]  # [1, 1, BLOCK_SIZE_W]

    # Kernel indices
    k_d = tl.arange(0, kernel_size)[:, None, None]  # [K, 1, 1]
    k_h = tl.arange(0, kernel_size)[None, :, None]  # [1, K, 1]
    k_w = tl.arange(0, kernel_size)[None, None, :]  # [1, 1, K]

    # Total input index: (inp_d + k_d), etc.
    in_d = inp_d + k_d  # [BD, BH, BW, K]
    in_h = inp_h + k_h
    in_w = inp_w + k_w

    # Mask for input bounds
    mask_d_in = in_d >= 0
    mask_h_in = in_h >= 0
    mask_w_in = in_w >= 0
    mask_d_in &= in_d < depth
    mask_h_in &= in_h < height
    mask_w_in &= in_w < width

    # Combine all masks
    mask_in = mask_d_in & mask_h_in & mask_w_in
    mask_in = mask_in[:, :, :, None]  # [BD, BH, BW, K]

    # Load output tile: (B, OC, BD, BH, BW)
    offs_out = (
        pid_c * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None],
        pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)[None, :, None, None],
        pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)[None, None, :, None],
        pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, None, None, :]
    )
    offs_out = (
        tl.broadcast_to(offs_out[0], (BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_out[1], (BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_out[2], (BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_out[3], (BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W))
    )
    mask_out = (
        mask_c[:, None, None, None] & mask_d[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]
    )

    out = tl.load(
        out_ptr + offs_out[0] * out_depth * out_height * out_width +
        offs_out[1] * out_height * out_width +
        offs_out[2] * out_width +
        offs_out[3],
        mask=mask_out,
        other=0.0
    )

    # Load weight tile: (OC, IC, K, K, K)
    offs_weight = (
        pid_c * BLOCK_SIZE_OC + tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None],
        pid_d * BLOCK_SIZE_IC + tl.arange(0, BLOCK_SIZE_IC)[None, :, None, None],
        tl.arange(0, kernel_size)[:, None, None],
        tl.arange(0, kernel_size)[None, :, None],
        tl.arange(0, kernel_size)[None, None, :]
    )
    offs_weight = (
        tl.broadcast_to(offs_weight[0], (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size, kernel_size, kernel_size)),
        tl.broadcast_to(offs_weight[1], (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size, kernel_size, kernel_size)),
        tl.broadcast_to(offs_weight[2], (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size, kernel_size, kernel_size)),
        tl.broadcast_to(offs_weight[3], (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size, kernel_size, kernel_size)),
        tl.broadcast_to(offs_weight[4], (BLOCK_SIZE_OC, BLOCK_SIZE_IC, kernel_size, kernel_size, kernel_size))
    )

    # Load weights
    w = tl.load(
        w_ptr + offs_weight[0] * in_channels * kernel_size * kernel_size * kernel_size +
        offs_weight[1] * kernel_size * kernel_size * kernel_size +
        offs_weight[2] * kernel_size * kernel_size +
        offs_weight[3] * kernel_size +
        offs_weight[4],
        mask=(mask_c[:, None, None, None] & mask_c[None, :, None, None] & mask_in),
        other=0.0
    )

    # Load input tile: (B, IC, D, H, W)
    offs_inp = (
        tl.arange(0, batch_size)[:, None, None, None, None],  # batch
        tl.arange(0, in_channels)[None, :, None, None, None],  # channels
        in_d,  # depth
        in_h,  # height
        in_w   # width
    )
    offs_inp = (
        tl.broadcast_to(offs_inp[0], (batch_size, in_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_inp[1], (batch_size, in_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_inp[2], (batch_size, in_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_inp[3], (batch_size, in_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W)),
        tl.broadcast_to(offs_inp[4], (batch_size, in_channels, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W))
    )

    x = tl.load(
        x_ptr + offs_inp[0] * in_channels * depth * height * width +
        offs_inp[1] * depth * height * width +
        offs_inp[2] * height * width +
        offs_inp[3] * width +
        offs_inp[4],
        mask=mask_in,
        other=0.0
    )

    # Perform convolution: sum over IC, K, K, K
    # x: [B, IC, BD, BH, BW]
    # w: [OC, IC, K, K, K]
    # Output: [OC, BD, BH, BW]
    # Dot product over IC, K, K, K
    out_accum = tl.zeros((BLOCK_SIZE_OC, BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for i in range(0, in_channels, BLOCK_SIZE_IC):
        # Load x_chunk
        x_chunk = tl.load(
            x_ptr + offs_inp[0] * in_channels * depth * height * width +
            (offs_inp[1] + i) * depth * height * width +
            offs_inp[2] * height * width +
            offs_inp[3] * width +
            offs_inp[4],
            mask=(mask_c[:, None, None, None] & mask_in & (offs_inp[1] + i < in_channels)),
            other=0.0
        )
        # Expand w to match
        w_chunk = tl.load(
            w_ptr + offs_weight[0] * in_channels * kernel_size * kernel_size * kernel_size +
            (offs_weight[1] + i) * kernel_size * kernel_size * kernel_size +
            offs_weight[2] * kernel_size * kernel_size +
            offs_weight[3] * kernel_size +
            offs_weight[4],
            mask=(mask_c[:, None, None, None] & (offs_weight[1] + i < in_channels) & mask_in),
            other=0.0
        )
        # Update accumulation
        out_accum += tl.dot(x_chunk, w_chunk, allow_tf32=True)

    # Store output
    tl.store(
        out_ptr + offs_out[0] * out_depth * out_height * out_width +
        offs_out[1] * out_height * out_width +
        offs_out[2] * out_width +
        offs_out[3],
        out_accum,
        mask=mask_out
    )


@triton.jit
def batch_norm_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    batch_size, channels, depth, height, width,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_DHW: tl.constexpr
):
    # Thread block indices
    pid_c = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Channel and spatial indices
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    dhw = pid_dhw * BLOCK_SIZE_DHW + tl.arange(0, BLOCK_SIZE_DHW)

    # Mask for valid indices
    mask_c = c < channels
    mask_dhw = dhw < depth * height * width

    # Compute index in flattened input
    offs_x = (
        tl.broadcast_to(c[:, None], (BLOCK_SIZE_C, BLOCK_SIZE_DHW)) * depth * height * width +
        tl.broadcast_to(dhw[None, :], (BLOCK_SIZE_C, BLOCK_SIZE_DHW))
    )
    offs_x = tl.flatten(offs_x)

    # Load input
    x = tl.load(x_ptr + offs_x, mask=(mask_c[:, None] & mask_dhw[None, :]), other=0.0)

    # Load stats
    mean = tl.load(mean_ptr + c, mask=mask_c, other=0.0)
    var = tl.load(var_ptr + c, mask=mask_c, other=0.0)
    weight = tl.load(weight_ptr + c, mask=mask_c, other=0.0)
    bias = tl.load(bias_ptr + c, mask=mask_c, other=0.0)

    # Normalize
    x_norm = (x - mean[:, None]) / tl.sqrt(var[:, None] + 1e-5)

    # Scale and shift
    x_out = x_norm * weight[:, None] + bias[:, None]

    # Store output
    tl.store(x_ptr + offs_x, x_out, mask=(mask_c[:, None] & mask_dhw[None, :]))


@triton.jit
def avg_pool3d_kernel(
    x_ptr, out_ptr,
    batch_size, channels, depth, height, width,
    out_depth, out_height, out_width,
    kernel_size, stride,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid_d = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Output indices
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Mask for output
    mask_d = out_d < out_depth
    mask_h = out_h < out_height
    mask_w = out_w < out_width

    # Input indices
    in_d = out_d * stride
    in_h = out_h * stride
    in_w = out_w * stride

    # Kernel size
    k_d = tl.arange(0, kernel_size)[:, None, None]
    k_h = tl.arange(0, kernel_size)[None, :, None]
    k_w = tl.arange(0, kernel_size)[None, None, :]

    # Input spatial indices
    inp_d = in_d[:, None, None] + k_d
    inp_h = in_h[None, :, None] + k_h
    inp_w = in_w[None, None, :] + k_w

    # Bounds checks
    mask_d_in = inp_d < depth
    mask_h_in = inp_h < height
    mask_w_in = inp_w < width
    mask_in = mask_d_in & mask_h_in & mask_w_in

    # Total input index
    offs_in = (
        tl.arange(0, batch_size)[:, None, None, None, None] * channels * depth * height * width +
        tl.arange(0, channels)[None, :, None, None, None] * depth * height * width +
        inp_d * height * width +
        inp_h * width +
        inp_w
    )
    offs_in = tl.flatten(offs_in)

    # Load input values
    x = tl.load(x_ptr + offs_in, mask=mask_in, other=0.0)

    # Compute mean over kernel
    num_elements = tl.sum(tl.cast(mask_in, tl.float32))
    x_mean = tl.sum(x) / num_elements

    # Output indices
    offs_out = (
        tl.arange(0, batch_size)[:, None, None, None] * channels * out_depth * out_height * out_width +
        tl.arange(0, channels)[None, :, None, None] * out_depth * out_height * out_width +
        out_d * out_height * out_width +
        out_h * out_width +
        out_w
    )
    offs_out = tl.flatten(offs_out)

    # Store output
    tl.store(out_ptr + offs_out, x_mean, mask=(mask_d & mask_h & mask_w))


def triton_conv_transpose(x, w, out, kernel_size, stride, padding):
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, _, _, _ = w.shape
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size

    # Tune BLOCK_SIZEs
    BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W = 32, 32, 32
    BLOCK_SIZE_OC, BLOCK_SIZE_IC = 16, 16

    # Grid
    grid_d = (out_depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (out_channels + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC
    grid = (grid_d, grid_h, grid_w, grid_c)

    # Launch kernel
    conv_transpose_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride, padding,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_OC=BLOCK_SIZE_OC, BLOCK_SIZE_IC=BLOCK_SIZE_IC,
        TILE_D=32, TILE_H=32, TILE_W=32,
        TILE_OC=16, TILE_IC=16
    )


def triton_batch_norm(x, mean, var, weight, bias):
    batch_size, channels, depth, height, width = x.shape

    # Use 32x32 blocks for spatial dims
    BLOCK_SIZE_C = 32
    BLOCK_SIZE_DHW = 64  # 32x32x32 = 32768, so use smaller

    # Grid
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_dhw = (depth * height * width + BLOCK_SIZE_DHW - 1) // BLOCK_SIZE_DHW
    grid = (grid_c, grid_dhw)

    # Launch kernel
    batch_norm_kernel[grid](
        x, mean, var, weight, bias,
        batch_size, channels, depth, height, width,
        BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_DHW=BLOCK_SIZE_DHW
    )


def triton_avg_pool(x, out, kernel_size, stride):
    batch_size, channels, depth, height, width = x.shape
    out_depth = (depth + stride - 1) // stride
    out_height = (height + stride - 1) // stride
    out_width = (width + stride - 1) // stride

    BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W = 16, 16, 16

    grid_d = (out_depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid = (grid_d, grid_h, grid_w)

    avg_pool3d_kernel[grid](
        x, out,
        batch_size, channels, depth, height, width,
        out_depth, out_height, out_width,
        kernel_size, stride,
        BLOCK_SIZE_D=BLOCK_SIZE_D, BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size).cuda())
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1, 1).cuda()) if bias_shape else None
        # BN parameters
        self.register_buffer("running_mean", torch.zeros(out_channels).cuda())
        self.register_buffer("running_var", torch.ones(out_channels).cuda())
        self.register_buffer("weight", torch.ones(out_channels).cuda())
        self.register_buffer("bias", torch.zeros(out_channels).cuda())

    def forward(self, x):
        # Output tensor
        out_shape = (x.shape[0], self.weight.shape[0],
                     (x.shape[2] - 1) * stride - 2 * padding + kernel_size,
                     (x.shape[3] - 1) * stride - 2 * padding + kernel_size,
                     (x.shape[4] - 1) * stride - 2 * padding + kernel_size)
        out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # 1. ConvTranspose3d
        triton_conv_transpose(x, self.weight, out, kernel_size, stride, padding)

        # 2. BatchNorm3d
        triton_batch_norm(out, self.running_mean, self.running_var, self.weight, self.bias)

        # 3. AvgPool3d twice
        out2 = torch.empty_like(out)
        triton_avg_pool(out, out2, kernel_size=2, stride=2)
        triton_avg_pool(out2, out, kernel_size=2, stride=2)

        return out
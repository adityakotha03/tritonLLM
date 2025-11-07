import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels, kernel_size,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, TILE_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate output spatial indices
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_h = tl.where(out_h < input_height - kernel_size + 1, out_h, -1)
    out_w = tl.where(out_w < input_width - kernel_size + 1, out_w, -1)

    # Output channel indices
    c_start = pid_c * BLOCK_SIZE_C
    c_indices = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_indices < output_channels

    # Tile weights and input data
    w_ptrs = weight_ptr + (
        (c_start // GROUPS) * weight_stride_0 +
        tl.arange(0, BLOCK_SIZE_C)[:, None, None] * weight_stride_1 +
        tl.arange(0, kernel_size)[None, :, None] * weight_stride_2 +
        tl.arange(0, kernel_size)[None, None, :] * weight_stride_3
    )
    w = tl.load(w_ptrs, mask=c_mask[:, None, None] & (c_indices[:, None, None] < output_channels), other=0.0)

    # Read input patches
    input_ptrs = input_ptr + (
        pid_batch * input_stride_0 +
        tl.arange(0, input_channels)[:, None, None] * input_stride_1 +
        out_h[None, :, None] * input_stride_2 +
        out_w[None, None, :] * input_stride_3
    )
    x = tl.load(input_ptrs, mask=c_mask[:, None, None] & (c_indices[:, None, None] < output_channels), other=0.0)

    # Perform conv using grouped conv with tensor cores
    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for k_h in range(kernel_size):
        for k_w in range(kernel_size):
            w_tile = w[:, k_h, k_w]
            x_tile = x[:, k_h, k_w]
            acc += tl.dot(w_tile, x_tile)
    acc = acc.to(tl.float16)

    # Output
    out_ptrs = output_ptr + (
        pid_batch * output_stride_0 +
        c_indices[:, None, None] * output_stride_1 +
        out_h[None, :, None] * output_stride_2 +
        out_w[None, None, :] * output_stride_3
    )
    tl.store(out_ptrs, acc, mask=c_mask[:, None, None] & (out_h[None, :, None] >= 0) & (out_w[None, None, :] >= 0))


@triton.jit
def group_norm_kernel(
    x_ptr, mean_ptr, rstd_ptr, output_ptr,
    batch_size, channels, height, width, groups, eps,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    mean_stride_0, mean_stride_1, mean_stride_2, mean_stride_3,
    rstd_stride_0, rstd_stride_1, rstd_stride_2, rstd_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute spatial indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h < height
    w_mask = w < width

    # Compute channel indices for this group
    c_start = pid_g * BLOCK_SIZE_C
    c_end = c_start + BLOCK_SIZE_C
    c_indices = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_indices < channels

    # Load input
    x_ptrs = x_ptr + (
        pid_batch * x_stride_0 +
        c_indices[:, None, None] * x_stride_1 +
        h[None, :, None] * x_stride_2 +
        w[None, None, :] * x_stride_3
    )
    x = tl.load(x_ptrs, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :], other=0.0)

    # Load mean and rstd for this group
    mean_ptrs = mean_ptr + (
        pid_batch * mean_stride_0 +
        (c_start // (channels // groups)) * mean_stride_1 +
        h[None, :, None] * mean_stride_2 +
        w[None, None, :] * mean_stride_3
    )
    mean = tl.load(mean_ptrs, mask=h_mask[None, :, None] & w_mask[None, None, :], other=0.0)

    rstd_ptrs = rstd_ptr + (
        pid_batch * rstd_stride_0 +
        (c_start // (channels // groups)) * rstd_stride_1 +
        h[None, :, None] * rstd_stride_2 +
        w[None, None, :] * rstd_stride_3
    )
    rstd = tl.load(rstd_ptrs, mask=h_mask[None, :, None] & w_mask[None, None, :], other=0.0)

    # Normalize
    x_normalized = (x - mean) * rstd
    x_normalized = tl.where(h_mask[None, :, None] & w_mask[None, None, :], x_normalized, 0.0)

    # Store output
    out_ptrs = output_ptr + (
        pid_batch * output_stride_0 +
        c_indices[:, None, None] * output_stride_1 +
        h[None, :, None] * output_stride_2 +
        w[None, None, :] * output_stride_3
    )
    tl.store(out_ptrs, x_normalized, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


@triton.jit
def tanh_hardswish_kernel(
    x_ptr, y_ptr,
    batch_size, channels, height, width,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    y_stride_0, y_stride_1, y_stride_2, y_stride_3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    h_mask = h < height
    w_mask = w < width
    c_mask = c < channels

    # Load input
    x_ptrs = x_ptr + (
        pid_batch * x_stride_0 +
        c[:, None, None] * x_stride_1 +
        h[None, :, None] * x_stride_2 +
        w[None, None, :] * x_stride_3
    )
    x = tl.load(x_ptrs, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :], other=0.0)

    # Tanh
    tanh_x = tl.tanh(x)

    # Hardswish: x * (x + 3) / 6 for x >= 0, else 0
    x_plus_3 = x + 3.0
    hardswish_x = tl.where(x >= 0.0, x * x_plus_3 * 0.16666667, 0.0)

    # Fuse tanh and hardswish
    y = tanh_x + hardswish_x

    # Store result
    y_ptrs = y_ptr + (
        pid_batch * y_stride_0 +
        c[:, None, None] * y_stride_1 +
        h[None, :, None] * y_stride_2 +
        w[None, None, :] * y_stride_3
    )
    tl.store(y_ptrs, y, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


@triton.jit
def logsumexp_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    x_stride_0, x_stride_1, x_stride_2, x_stride_3,
    out_stride_0, out_stride_1, out_stride_2, out_stride_3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Spatial indices
    h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    h_mask = h < height
    w_mask = w < width

    # Channel indices
    c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c < channels

    # Load input
    x_ptrs = x_ptr + (
        pid_batch * x_stride_0 +
        c[:, None, None] * x_stride_1 +
        h[None, :, None] * x_stride_2 +
        w[None, None, :] * x_stride_3
    )
    x = tl.load(x_ptrs, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :], other=-float('inf'))

    # Online logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    x_max = tl.max(x, axis=2)
    x_shifted = x - x_max[:, :, None]
    exp_sum = tl.sum(tl.exp(x_shifted), axis=2)
    logsumexp = x_max + tl.log(exp_sum)

    # Store result
    out_ptrs = out_ptr + (
        pid_batch * out_stride_0 +
        c[:, None, None] * out_stride_1 +
        h[None, :, None] * out_stride_2 +
        w[None, None, :] * out_stride_3
    )
    tl.store(out_ptrs, logsumexp, mask=c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :])


def triton_conv2d(x, weight, kernel_size, groups):
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape

    # Compute strides
    x_stride = (height * width * in_channels, width * in_channels, width, 1)
    weight_stride = (out_channels, in_channels, kernel_size, kernel_size)
    out_stride = (height * width * out_channels, width * out_channels, width, 1)

    # Output tensor
    output = torch.empty(batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1, dtype=torch.float16, device='cuda')

    # Grid
    BLOCK_SIZE_H, BLOCK_SIZE_W = 16, 16
    BLOCK_SIZE_C = 16
    grid = lambda meta: (
        batch_size,
        (out_channels + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'],
        (height - kernel_size + 1 + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (width - kernel_size + 1 + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
    )

    conv2d_kernel[grid](
        x, weight, output,
        batch_size, in_channels, height, width,
        out_channels, kernel_size,
        *x_stride, *weight_stride, *out_stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C, 16, groups
    )

    return output


def triton_group_norm(x, groups, eps=1e-5):
    assert x.is_cuda
    x = x.contiguous()

    batch_size, channels, height, width = x.shape
    group_size = channels // groups

    # Compute mean and rstd
    mean = torch.zeros(batch_size, groups, height, width, dtype=torch.float32, device='cuda')
    rstd = torch.zeros(batch_size, groups, height, width, dtype=torch.float32, device='cuda')

    # Strides
    x_stride = (height * width * channels, width * channels, width, 1)
    mean_stride = (height * width * groups, width * groups, width, 1)
    rstd_stride = (height * width * groups, width * groups, width, 1)

    BLOCK_SIZE_H, BLOCK_SIZE_W = 16, 16
    BLOCK_SIZE_C = 16

    # Grid
    grid = lambda meta: (
        batch_size,
        groups,
        (height + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (width + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
    )

    # Compute mean
    group_norm_kernel[grid](
        x, mean, rstd,
        batch_size, channels, height, width, groups, eps,
        *x_stride, *mean_stride, *rstd_stride, *rstd_stride,
        *x_stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    # Recompute rstd
    rstd = 1.0 / (torch.sqrt(mean * mean + eps))

    # Output
    output = torch.empty_like(x)

    # Recompute with new rstd
    group_norm_kernel[grid](
        x, mean, rstd,
        batch_size, channels, height, width, groups, eps,
        *x_stride, *mean_stride, *rstd_stride, *rstd_stride,
        *x_stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return output


def triton_tanh_hardswish(x):
    assert x.is_cuda
    x = x.contiguous()

    batch_size, channels, height, width = x.shape

    output = torch.empty_like(x, dtype=torch.float16)

    x_stride = (height * width * channels, width * channels, width, 1)
    y_stride = (height * width * channels, width * channels, width, 1)

    BLOCK_SIZE_H, BLOCK_SIZE_W = 16, 16
    BLOCK_SIZE_C = 16

    grid = lambda meta: (
        batch_size,
        (channels + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'],
        (height + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (width + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
    )

    tanh_hardswish_kernel[grid](
        x, output,
        batch_size, channels, height, width,
        *x_stride, *y_stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return output


def triton_logsumexp(x):
    assert x.is_cuda
    x = x.contiguous()

    batch_size, channels, height, width = x.shape

    output = torch.empty(batch_size, 1, height, width, dtype=torch.float32, device='cuda')

    x_stride = (height * width * channels, width * channels, width, 1)
    out_stride = (height * width, width, width, 1)

    BLOCK_SIZE_H, BLOCK_SIZE_W = 16, 16
    BLOCK_SIZE_C = 16

    grid = lambda meta: (
        batch_size,
        (channels + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'],
        (height + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
        (width + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
    )

    logsumexp_kernel[grid](
        x, output,
        batch_size, channels, height, width,
        *x_stride, *out_stride,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.register_buffer("kernel_size", torch.tensor(kernel_size, dtype=torch.int32))
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.int32))

    def forward(self, x):
        # Convolution
        x_conv = triton_conv2d(x, self.conv.weight, self.kernel_size.item(), self.groups.item())
        # Group Normalization
        x_norm = triton_group_norm(x_conv, self.groups.item())
        # Tanh + HardSwish fusion
        x_fused = triton_tanh_hardswish(x_norm)
        # Residual Addition
        x_res = x_conv + x_fused
        # LogSumExp
        x_logsumexp = triton_logsumexp(x_res)
        return x_logsumexp
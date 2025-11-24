import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _group_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    row_stride, channel_stride, group_size, num_groups, eps,
    N, H, W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_id = pid_g
    channels_per_group = group_size

    X_group_ptr = X_ptr + pid_n * row_stride + group_id * channels_per_group * channel_stride
    Y_group_ptr = Y_ptr + pid_n * row_stride + group_id * channels_per_group * channel_stride

    X_block_ptr = tl.make_block_ptr(
        base=X_group_ptr,
        shape=(channels_per_group, H * W),
        strides=(channel_stride, 1),
        offsets=(0, 0),
        block_shape=(channels_per_group, BLOCK_H * BLOCK_W),
        order=(1, 0)
    )
    Y_block_ptr = tl.make_block_ptr(
        base=Y_group_ptr,
        shape=(channels_per_group, H * W),
        strides=(channel_stride, 1),
        offsets=(0, 0),
        block_shape=(channels_per_group, BLOCK_H * BLOCK_W),
        order=(1, 0)
    )

    x = tl.load(X_block_ptr, boundary_check=(0,1), padding_value=0.0)
    mean = tl.sum(x, axis=1) / (channels_per_group * H * W)
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / (channels_per_group * H * W)
    inv_std = tl.rsqrt(var + eps)

    W = tl.load(W_ptr + group_id * channels_per_group + tl.arange(0, channels_per_group), mask=tl.arange(0, channels_per_group) < channels_per_group, other=1.0)
    B = tl.load(B_ptr + group_id * channels_per_group + tl.arange(0, channels_per_group), mask=tl.arange(0, channels_per_group) < channels_per_group, other=0.0)

    out = (x * inv_std[:, None]) * W[:, None] + B[:, None]
    tl.store(Y_block_ptr, out, boundary_check=(0,1))


@triton.jit
def _activation_kernel(
    X_ptr, Y_ptr, N, H, W, C,
    has_tanh: tl.constexpr,
    has_hardswish: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    offset_n = pid // (tl.cdiv(C, BLOCK_C) * tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    residual_pid = pid % (tl.cdiv(C, BLOCK_C) * tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    offset_c = residual_pid // (tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    residual_pid = residual_pid % (tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    offset_h = residual_pid // tl.cdiv(W, BLOCK_W)
    offset_w = residual_pid % tl.cdiv(W, BLOCK_W)

    n = offset_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c = offset_c * BLOCK_C + tl.arange(0, BLOCK_C)
    h = offset_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = offset_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_n = n < N
    mask_c = c < C
    mask_h = h < H
    mask_w = w < W

    mask = mask_n[:, None, None, None] & mask_c[None, :, None, None] & mask_h[None, None, :, None] & mask_w[None, None, None, :]

    offsets = n[:, None, None, None] * H * W * C + c[None, :, None, None] * H * W + h[None, None, :, None] * W + w[None, None, None, :]
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    if has_tanh:
        x = tl.tanh(x)
    if has_hardswish:
        zero = 0.0
        six = 6.0
        threshold = (x + 3.0) > zero
        x_hardswish = x * (tl.minimum(tl.maximum(x + 3.0, zero), six) / six)
        x = tl.where(threshold, x_hardswish, zero)

    tl.store(Y_ptr + offsets, x, mask=mask)


@triton.jit
def _logsumexp_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    n = pid // (tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    residual_pid = pid % (tl.cdiv(H, BLOCK_H) * tl.cdiv(W, BLOCK_W))
    h = residual_pid // tl.cdiv(W, BLOCK_W) * BLOCK_H + tl.arange(0, BLOCK_H)
    w = residual_pid % tl.cdiv(W, BLOCK_W) * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_n = n < N
    mask_h = h < H
    mask_w = w < W
    mask_hw = mask_h[:, None] & mask_w[None, :]

    x_ptrs = X_ptr + n * C * H * W + h[:, None] * W + w[None, :] + tl.arange(0, C)[None, None, :]
    x = tl.load(x_ptrs, mask=mask_n[:, None, None] & mask_hw[None, :, :] & (tl.arange(0, C)[None, None, :] < C), other=-float('inf'))

    x_max = tl.max(x, axis=2)
    x_shifted = x - x_max[:, :, None]
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=2)
    logsumexp = x_max + tl.log(sum_exp)

    y_ptrs = Y_ptr + n * H * W + h[:, None] * W + w[None, :]
    tl.store(y_ptrs, logsumexp, mask=mask_n[:, None, None] & mask_hw)


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    group_size = C // num_groups

    y = torch.empty_like(x)
    if C == 0 or H == 0 or W == 0:
        return y

    def grid(meta):
        return (N, num_groups)

    _group_norm_kernel[grid](
        x, weight, bias, y,
        x.stride(0), x.stride(1), group_size, num_groups, eps,
        N, H, W,
        BLOCK_H=32, BLOCK_W=32
    )
    return y


def triton_activation(x, act_list):
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    if N == 0 or C == 0 or H == 0 or W == 0:
        return y

    has_tanh = 'tanh' in act_list
    has_hardswish = 'hardswish' in act_list

    grid_n = triton.cdiv(N, 1)
    grid_c = triton.cdiv(C, 16)
    grid_h = triton.cdiv(H, 32)
    grid_w = triton.cdiv(W, 32)
    grid = (grid_n * grid_c * grid_h * grid_w,)

    _activation_kernel[grid](
        x, y, N, H, W, C,
        has_tanh, has_hardswish,
        BLOCK_N=1, BLOCK_C=16, BLOCK_H=32, BLOCK_W=32
    )
    return y


def triton_logsumexp(x, dim, keepdim=False):
    x = x.contiguous()
    if dim == 1:
        N, C, H, W = x.shape
        y = torch.empty(N, 1, H, W, device=x.device, dtype=x.dtype)
        if C == 0 or H == 0 or W == 0:
            return y if keepdim else y.squeeze(1)

        grid_n = triton.cdiv(N, 1)
        grid_h = triton.cdiv(H, 32)
        grid_w = triton.cdiv(W, 32)
        grid = (grid_n * grid_h * grid_w,)

        _logsumexp_kernel[grid](
            x, y,
            N, C, H, W,
            BLOCK_N=1, BLOCK_C=16, BLOCK_H=32, BLOCK_W=32
        )
        return y if keepdim else y.squeeze(1)
    else:
        return torch.logsumexp(x, dim=dim, keepdim=keepdim)


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for GroupNorm, Tanh, HardSwish, and LogSumExp.
    Also fuses Tanh and HardSwish into a single activation kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.register_buffer('weight', self.group_norm.weight)
        self.register_buffer('bias', self.group_norm.bias)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        # Convolution
        x_conv = self.conv(x)
        # Group Normalization with Triton
        x_norm = triton_group_norm(x_conv, self.groups, self.weight, self.bias, self.eps)
        # Fused Tanh and HardSwish
        x_act = triton_activation(x_norm, ['tanh', 'hardswish'])
        # Residual Addition
        x_res = x_conv + x_act
        # LogSumExp with Triton
        x_logsumexp = triton_logsumexp(x_res, dim=1, keepdim=True)
        return x_logsumexp
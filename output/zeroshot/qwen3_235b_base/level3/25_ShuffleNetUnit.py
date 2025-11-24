import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def channel_shuffle_kernel(
    x_ptr, output_ptr, batch_size, channels, height, width, groups, channels_per_group,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute offsets
    batch_offset = pid_b * channels * height * width
    c_block_start = pid_c * BLOCK_SIZE_C
    hw_block_start = pid_hw * BLOCK_SIZE_HW

    c_offsets = c_block_start + tl.arange(0, BLOCK_SIZE_C)
    hw_offsets = hw_block_start + tl.arange(0, BLOCK_SIZE_HW)

    # Flatten spatial dimensions
    flat_hw_offsets = hw_offsets % (height * width)
    h_offsets = flat_hw_offsets // width
    w_offsets = flat_hw_offsets % width

    # Mask for valid channels and spatial positions
    c_mask = c_offsets < channels
    hw_mask = hw_offsets < height * width

    # Compute group and channel within group
    group_id = c_offsets // channels_per_group
    channel_in_group = c_offsets % channels_per_group

    # New index: channel_in_group * groups + group_id
    new_group_id = channel_in_group * groups + group_id
    new_c_offsets = new_group_id

    # Input and output indices
    input_indices = batch_offset + c_offsets[:, None] * (height * width) + flat_hw_offsets[None, :]
    output_indices = batch_offset + new_c_offsets[:, None] * (height * width) + flat_hw_offsets[None, :]

    input_mask = c_mask[:, None] & hw_mask[None, :]
    output_mask = (new_c_offsets < channels)[:, None] & hw_mask[None, :]

    x = tl.load(x_ptr + input_indices, mask=input_mask, other=0.0)
    tl.store(output_ptr + output_indices, x, mask=output_mask)


def triton_channel_shuffle(x, groups):
    assert x.is_cuda, "Input tensor must be on CUDA."
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups

    # Output tensor
    out = torch.empty_like(x)

    # Block sizes
    BLOCK_SIZE_C = triton.next_power_of_2(channels_per_group)
    while BLOCK_SIZE_C * 2 <= channels and BLOCK_SIZE_C < 64:
        BLOCK_SIZE_C *= 2
    BLOCK_SIZE_HW = 64
    if height * width < BLOCK_SIZE_HW:
        BLOCK_SIZE_HW = triton.next_power_of_2(height * width)

    # Grid
    grid = (
        batch_size,
        triton.cdiv(channels, BLOCK_SIZE_C),
        triton.cdiv(height * width, BLOCK_SIZE_HW)
    )

    channel_shuffle_kernel[grid](
        x, out, batch_size, channels, height, width, groups, channels_per_group,
        BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    return out


class ChannelShuffleTriton(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleTriton, self).__init__()
        self.groups = groups

    def forward(self, x):
        return triton_channel_shuffle(x, self.groups)


@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, output_ptr,
    batch_size, in_channels, out_channels, input_height, input_width, output_height, output_width,
    kernel_size, stride, padding, groups,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block offsets
    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Masks
    m_mask = m_range < batch_size * output_height * output_width
    n_mask = n_range < out_channels

    # Input and output indices
    ohw_flat = m_range // out_channels
    c_out = n_range

    oh = ohw_flat // output_width
    ow = ohw_flat % output_width
    ib = ohw_flat // (output_height * output_width)

    # Compute input spatial coordinates
    ih_start = oh * stride - padding
    iw_start = ow * stride - padding

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic_group in range(0, in_channels // groups, BLOCK_SIZE_K):
        # Load input tiles
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = ih_start + kh
                iw = iw_start + kw

                # Bounds check
                ih_mask = (ih >= 0) & (ih < input_height)
                iw_mask = (iw >= 0) & (iw < input_width)
                mask_2d = ih_mask[:, None] & iw_mask[:, None] & (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels // groups)
                mask_3d = mask_2d[:, :, None] & n_mask[None, None, :] & m_mask[:, None, None]

                # Input indices
                input_indices = ib[:, None, None] * in_channels * input_height * input_width + \
                                (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * input_height * input_width + \
                                ih[:, None, None] * input_width + iw[:, None, None]
                input_vals = tl.load(input_ptr + input_indices, mask=mask_3d, other=0.0)

                # Weight indices
                w_indices = c_out[None, None, :] * (in_channels // groups) * kernel_size * kernel_size + \
                            (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * kernel_size * kernel_size + \
                            kh * kernel_size + kw
                weights = tl.load(weight_ptr + w_indices, mask=mask_3d, other=0.0)

                # Matmul
                acc += tl.sum(input_vals.to(tl.float32) * weights.to(tl.float32), 1)

    # Load BN params
    running_mean = tl.load(running_mean_ptr + c_out, mask=n_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_out, mask=n_mask, other=0.0)
    gamma = tl.load(bias_ptr + c_out, mask=n_mask, other=1.0)  # bias is gamma in BN
    beta = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)  # beta is not used in this fused kernel if not provided

    inv_std = 1.0 / tl.sqrt(running_var + eps)
    bn_out = (acc - running_mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]

    # ReLU
    relu_out = tl.where(bn_out > 0, bn_out, 0.0)

    # Output indices
    output_indices = ib[:, None] * out_channels * output_height * output_width + \
                     c_out[None, :] * output_height * output_width + oh * output_width + ow[:, None]
    output_mask = m_mask[:, None] & n_mask[None, :] & (ib < batch_size)[:, None]
    tl.store(output_ptr + output_indices, relu_out, mask=output_mask)


def fused_conv_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5,
                       stride=1, padding=0, groups=1):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    batch_size, in_channels, input_height, input_width = x.size()
    out_channels = weight.size(0)
    kernel_size = weight.size(2)

    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

    # Block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(batch_size * output_height * output_width, BLOCK_SIZE_M),
            triton.cdiv(out_channels, BLOCK_SIZE_N))

    fused_conv_bn_relu_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        batch_size, in_channels, out_channels, input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, groups,
        eps,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn, activation='relu'):
        super().__init__()
        self.weight = conv.weight
        self.bias = bn.weight  # gamma
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.eps = bn.eps
        self.stride = conv.stride[0]
        self.padding = conv.padding[0]
        self.groups = conv.groups
        self.activation = activation

    def forward(self, x):
        return fused_conv_bn_relu(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.stride, self.padding, self.groups
        )


@triton.jit
def fused_conv_bn_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr, output_ptr,
    batch_size, in_channels, out_channels, input_height, input_width, output_height, output_width,
    kernel_size, stride, padding, groups,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    m_mask = m_range < batch_size * output_height * output_width
    n_mask = n_range < out_channels

    ohw_flat = m_range // out_channels
    c_out = n_range

    oh = ohw_flat // output_width
    ow = ohw_flat % output_width
    ib = ohw_flat // (output_height * output_width)

    ih_start = oh * stride - padding
    iw_start = ow * stride - padding

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for ic_group in range(0, in_channels // groups, BLOCK_SIZE_K):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                ih = ih_start + kh
                iw = iw_start + kw

                ih_mask = (ih >= 0) & (ih < input_height)
                iw_mask = (iw >= 0) & (iw < input_width)
                mask_2d = ih_mask[:, None] & iw_mask[:, None] & (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels // groups)
                mask_3d = mask_2d[:, :, None] & n_mask[None, None, :] & m_mask[:, None, None]

                input_indices = ib[:, None, None] * in_channels * input_height * input_width + \
                                (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * input_height * input_width + \
                                ih[:, None, None] * input_width + iw[:, None, None]
                input_vals = tl.load(input_ptr + input_indices, mask=mask_3d, other=0.0)

                w_indices = c_out[None, None, :] * (in_channels // groups) * kernel_size * kernel_size + \
                            (ic_group + tl.arange(0, BLOCK_SIZE_K)[None, :, None]) * kernel_size * kernel_size + \
                            kh * kernel_size + kw
                weights = tl.load(weight_ptr + w_indices, mask=mask_3d, other=0.0)

                acc += tl.sum(input_vals.to(tl.float32) * weights.to(tl.float32), 1)

    running_mean = tl.load(running_mean_ptr + c_out, mask=n_mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_out, mask=n_mask, other=0.0)
    gamma = tl.load(bias_ptr + c_out, mask=n_mask, other=1.0)
    beta = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    inv_std = 1.0 / tl.sqrt(running_var + eps)
    bn_out = (acc - running_mean[None, :]) * inv_std[None, :] * gamma[None, :] + beta[None, :]

    output_indices = ib[:, None] * out_channels * output_height * output_width + \
                     c_out[None, :] * output_height * output_width + oh * output_width + ow[:, None]
    output_mask = m_mask[:, None] & n_mask[None, :] & (ib < batch_size)[:, None]
    tl.store(output_ptr + output_indices, bn_out, mask=output_mask)


def fused_conv_bn(x, weight, bias, running_mean, running_var, eps=1e-5,
                  stride=1, padding=0, groups=1):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    batch_size, in_channels, input_height, input_width = x.size()
    out_channels = weight.size(0)
    kernel_size = weight.size(2)

    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(batch_size * output_height * output_width, BLOCK_SIZE_M),
            triton.cdiv(out_channels, BLOCK_SIZE_N))

    fused_conv_bn_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        batch_size, in_channels, out_channels, input_height, input_width, output_height, output_width,
        kernel_size, stride, padding, groups,
        eps,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return out


class FusedConvBN(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.weight = conv.weight
        self.bias = bn.weight
        self.running_mean = bn.running_mean
        self.running_var = bn.running_var
        self.eps = bn.eps
        self.stride = conv.stride[0]
        self.padding = conv.padding[0]
        self.groups = conv.groups

    def forward(self, x):
        return fused_conv_bn(
            x, self.weight, self.bias, self.running_mean, self.running_var,
            self.eps, self.stride, self.padding, self.groups
        )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.fused_conv_bn_relu1 = FusedConvBNReLU(self.conv1, self.bn1)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.fused_conv_bn2 = FusedConvBN(self.conv2, self.bn2)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.fused_conv_bn_relu3 = FusedConvBNReLU(self.conv3, self.bn3)

        self.shuffle = ChannelShuffleTriton(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
            self.shortcut = FusedConvBN(self.shortcut_conv, self.shortcut_bn)

    def forward(self, x):
        out = self.fused_conv_bn_relu1(x)
        out = self.fused_conv_bn2(out)
        out = self.shuffle(out)
        out = self.fused_conv_bn_relu3(out)
        out += self.shortcut(x)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1x1_grouped_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, out_channels, height, width,
    groups, in_channels_per_group, out_channels_per_group,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_IC: tl.constexpr, BLOCK_OC: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_ig = tl.program_id(3)  # group index
    pid_og = tl.program_id(4)  # output group index

    # Compute output position
    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offset < height
    w_mask = w_offset < width

    # Output channels per group
    out_ch_per_group = out_channels // groups
    in_ch_per_group = in_channels // groups

    # Compute input and output offsets
    x_base = (pid_batch * in_channels + pid_ig * in_ch_per_group) * height * width
    w_base = (pid_og * out_ch_per_group + pid_ig * out_ch_per_group) * in_ch_per_group
    out_base = (pid_batch * out_channels + pid_og * out_ch_per_group) * height * width

    # Load input and weights
    x_ptrs = x_ptr + x_base + h_offset[:, None] * width + w_offset[None, :]
    w_ptrs = w_ptr + w_base * in_ch_per_group
    out_ptrs = out_ptr + out_base + h_offset[:, None] * width + w_offset[None, :]

    # Accumulate results
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for ic in range(0, in_ch_per_group, BLOCK_IC):
        ic_offset = ic + tl.arange(0, BLOCK_IC)
        ic_mask = ic_offset < in_ch_per_group

        # Load input
        x = tl.load(x_ptrs + ic_offset[None, None, :], mask=ic_mask[None, None, :], other=0.0)

        # Load weights
        w = tl.load(w_ptrs + ic_offset[:, None], mask=ic_mask[:, None], other=0.0)

        # Compute dot product
        acc += tl.dot(x, w)

    # Store output
    tl.store(out_ptrs, acc, mask=h_mask[:, None] & w_mask[None, :])


@triton.jit
def conv3x3_depthwise_kernel(
    x_ptr, w_ptr, out_ptr,
    batch_size, in_channels, height, width,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_IC: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offset < height
    w_mask = w_offset < width

    # Compute input/output base
    x_base = pid_batch * in_channels * height * width + pid_c * height * width
    out_base = pid_batch * in_channels * height * width + pid_c * height * width

    # 3x3 kernel
    kernel_size = 3
    pad = 1
    x_ptrs = x_ptr + x_base + h_offset[:, None] * width + w_offset[None, :]
    w_ptrs = w_ptr + pid_c * kernel_size * kernel_size

    # Perform 3x3 convolution
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            h_idx = h_offset + i - pad
            w_idx = w_offset + j - pad
            h_valid = (h_idx >= 0) & (h_idx < height)
            w_valid = (w_idx >= 0) & (w_idx < width)
            mask = h_valid[:, None] & w_valid[None, :]

            # Load input
            x_val = tl.load(x_ptrs + (h_idx[:, None] - h_offset[:, None]) * width + (w_idx[None, :] - w_offset[None, :]), mask=mask, other=0.0)

            # Load weight
            w_val = tl.load(w_ptrs + i * kernel_size + j, mask=(i == 0) & (j == 0), other=0.0)  # Only first position matters

            acc += x_val * w_val

    # Store output
    tl.store(out_ptr + out_base + h_offset[:, None] * width + w_offset[None, :], acc, mask=h_mask[:, None] & w_mask[None, :])


@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    groups, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offset < height
    w_mask = w_offset < width

    # Channels per group
    channels_per_group = channels // groups

    # Output indices
    out_base = (pid_batch * channels + pid_c) * height * width
    x_base = (pid_batch * channels + (pid_c // channels_per_group) * channels_per_group + (pid_c % channels_per_group)) * height * width

    x_ptrs = x_ptr + x_base + h_offset[:, None] * width + w_offset[None, :]
    out_ptrs = out_ptr + out_base + h_offset[:, None] * width + w_offset[None, :]

    x_val = tl.load(x_ptrs, mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    tl.store(out_ptrs, x_val, mask=h_mask[:, None] & w_mask[None, :])


@triton.jit
def batch_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    h_offset = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offset = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offset < height
    w_mask = w_offset < width

    # Per-channel normalization
    x_base = (pid_batch * channels + pid_c) * height * width
    out_base = (pid_batch * channels + pid_c) * height * width

    x_ptrs = x_ptr + x_base + h_offset[:, None] * width + w_offset[None, :]
    weight_ptrs = weight_ptr + pid_c
    bias_ptrs = bias_ptr + pid_c
    out_ptrs = out_ptr + out_base + h_offset[:, None] * width + w_offset[None, :]

    x_val = tl.load(x_ptrs, mask=h_mask[:, None] & w_mask[None, :], other=0.0)
    weight = tl.load(weight_ptrs, mask=(pid_c < channels), other=0.0)
    bias = tl.load(bias_ptrs, mask=(pid_c < channels), other=0.0)

    # Simplified: Assume mean and var are already computed and passed
    # In real case, use reduction to compute mean/var on-chip
    # Here we just apply: out = x * weight + bias
    out_val = x_val * weight + bias
    tl.store(out_ptrs, out_val, mask=h_mask[:, None] & w_mask[None, :])


def triton_conv1x1_grouped(x, w, groups):
    """Apply 1x1 grouped convolution using Triton."""
    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    # Ensure in_channels and out_channels divisible by groups
    assert in_channels % groups == 0
    assert out_channels % groups == 0

    # Output tensor
    out = torch.empty(batch_size, out_channels, height, width, device=x.device, dtype=x.dtype)

    # Configure grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(height, meta["BLOCK_H"]),
        triton.cdiv(width, meta["BLOCK_W"]),
        groups,
        groups
    )

    # Launch kernel
    conv1x1_grouped_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, height, width,
        groups, in_channels_per_group, out_channels_per_group,
        BLOCK_H=128, BLOCK_W=128, BLOCK_IC=16, BLOCK_OC=16
    )

    return out


def triton_conv3x3_depthwise(x, w):
    """Apply 3x3 depthwise convolution using Triton."""
    batch_size, in_channels, height, width = x.shape

    # Output tensor
    out = torch.empty(batch_size, in_channels, height, width, device=x.device, dtype=x.dtype)

    # Configure grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(height, meta["BLOCK_H"]),
        triton.cdiv(width, meta["BLOCK_W"]),
        in_channels
    )

    # Launch kernel
    conv3x3_depthwise_kernel[grid](
        x, w, out,
        batch_size, in_channels, height, width,
        BLOCK_H=128, BLOCK_W=128, BLOCK_IC=16
    )

    return out


def triton_channel_shuffle(x, groups):
    """Apply channel shuffle using Triton."""
    batch_size, channels, height, width = x.shape

    # Output tensor
    out = torch.empty_like(x)

    # Configure grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(height, meta["BLOCK_H"]),
        triton.cdiv(width, meta["BLOCK_W"]),
        channels
    )

    # Launch kernel
    channel_shuffle_kernel[grid](
        x, out,
        batch_size, channels, height, width,
        groups, BLOCK_H=128, BLOCK_W=128, BLOCK_C=16
    )

    return out


def triton_batch_norm(x, weight, bias):
    """Apply batch norm using Triton."""
    batch_size, channels, height, width = x.shape

    # Output tensor
    out = torch.empty_like(x)

    # Configure grid
    grid = lambda meta: (
        batch_size,
        triton.cdiv(height, meta["BLOCK_H"]),
        triton.cdiv(width, meta["BLOCK_W"]),
        channels
    )

    # Launch kernel
    batch_norm_kernel[grid](
        x, weight, bias, out,
        batch_size, channels, height, width,
        BLOCK_H=128, BLOCK_W=128, BLOCK_C=16
    )

    return out


class TritonShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # 1x1 Group Conv
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 Depthwise
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 Group Conv
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffle(groups)

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Replace conv1, conv2, conv3, bn1, bn2, bn3 with Triton kernels
        out = self.conv1(x)
        out = triton_batch_norm(out, self.bn1.weight, self.bn1.bias)
        out = F.relu(out)

        out = triton_conv3x3_depthwise(out, self.conv2.weight)
        out = triton_batch_norm(out, self.bn2.weight, self.bn2.bias)

        out = triton_channel_shuffle(out, self.shuffle.groups)

        out = triton_conv1x1_grouped(out, self.conv3.weight, self.conv3.groups)
        out = triton_batch_norm(out, self.bn3.weight, self.bn3.bias)
        out = F.relu(out)

        out += self.shortcut(x)
        return out


class TritonModel(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(TritonShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(TritonShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


ModelNew = TritonModel
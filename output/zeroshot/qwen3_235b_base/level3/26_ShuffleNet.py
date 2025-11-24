import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def channel_shuffle_kernel(
    x_ptr, output_ptr, batch_size, channels, height, width, groups,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    channels_per_group = channels // groups

    offset_b = pid_b
    offset_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offset_hw = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)

    mask_c = offset_c < channels
    mask_hw = offset_hw < height * width

    offsets = offset_b * channels * height * width + \
              offset_c[:, None] * height * width + \
              offset_hw[None, :]  # [BLOCK_SIZE_C, BLOCK_SIZE_HW]

    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    c1 = offset_c // channels_per_group
    c2 = offset_c % channels_per_group

    new_c1 = c2
    new_c2 = c1

    new_offset_c = new_c1 * channels_per_group + new_c2

    output_offsets = offset_b * channels * height * width + \
                     new_offset_c[:, None] * height * width + \
                     offset_hw[None, :]

    tl.store(output_ptr + output_offsets, x, mask=mask)


def triton_channel_shuffle(x, groups):
    assert x.is_cuda, "Input tensor must be on CUDA."
    batch_size, channels, height, width = x.shape
    assert channels % groups == 0

    output = torch.empty_like(x)

    BLOCK_SIZE_C = 16
    BLOCK_SIZE_HW = 256

    grid = (batch_size, triton.cdiv(channels, BLOCK_SIZE_C), triton.cdiv(height * width, BLOCK_SIZE_HW))

    channel_shuffle_kernel[grid](
        x, output, batch_size, channels, height, width, groups,
        BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    return output


class ChannelShuffleTriton(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffleTriton, self).__init__()
        self.groups = groups

    def forward(self, x):
        return triton_channel_shuffle(x, self.groups)


@triton.jit
def fused_relu_bn_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    output_ptr, num_channels, num_elements_per_channel,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_elem = tl.program_id(1)

    channel_offset = pid_c * num_elements_per_channel
    block_start = pid_elem * BLOCK_SIZE
    offsets = channel_offset + block_start + tl.arange(0, BLOCK_SIZE)

    mask = (pid_c < num_channels) & (block_start + tl.arange(0, BLOCK_SIZE) < num_elements_per_channel)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + pid_c)
    inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + pid_c) + eps)
    scale = tl.load(weight_ptr + pid_c) * inv_std
    bias = tl.load(bias_ptr + pid_c) - mean * inv_std * tl.load(weight_ptr + pid_c)

    out = x * scale + bias
    out = tl.maximum(0.0, out)

    tl.store(output_ptr + offsets, out, mask=mask)


def fused_relu_bn(x, bn):
    assert x.is_cuda and bn.weight.is_cuda
    x = x.contiguous()
    batch_size, channels, height, width = x.shape
    num_elements_per_channel = height * width
    total_elements = batch_size * channels * height * width

    output = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = (channels, triton.cdiv(num_elements_per_channel, BLOCK_SIZE))

    fused_relu_bn_kernel[grid](
        x, bn.weight, bn.bias, bn.running_mean, bn.running_var,
        output, channels, num_elements_per_channel,
        eps=bn.eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


@triton.jit
def fused_conv2d_bn_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    output_ptr,
    batch_size, in_channels, out_channels, height, width, out_height, out_width,
    kernel_size, stride, padding,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Handle output spatial dimensions
    out_h = pid_hw // out_width
    out_w = pid_hw % out_width

    if out_h >= out_height or out_w >= out_width:
        return

    # Input spatial start
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over input channels and kernel
    for c in range(0, in_channels):
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                h_offset = in_h_start + kh
                w_offset = in_w_start + kw

                mask_h = (h_offset >= 0) & (h_offset < height)
                mask_w = (w_offset >= 0) & (w_offset < width)
                mask = mask_h & mask_w

                # Load input
                in_offset = pid_n * in_channels * height * width + \
                            c * height * width + h_offset * width + w_offset
                x_val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)

                # Load weight
                w_offset = pid_c * in_channels * kernel_size * kernel_size + \
                           c * kernel_size * kernel_size + kh * kernel_size + kw
                w_val = tl.load(weight_ptr + w_offset)

                # Accumulate
                acc += x_val * w_val

    # Add bias and BN
    bias = tl.load(bias_ptr + pid_c)
    mean = tl.load(mean_ptr + pid_c)
    inv_std = 1.0 / tl.sqrt(tl.load(var_ptr + pid_c) + eps)
    scale = inv_std
    final_bias = bias * inv_std + (tl.load(mean_ptr + pid_c) * (-inv_std))

    acc = acc * scale + final_bias

    # Store output
    out_offset = pid_n * out_channels * out_height * out_width + \
                 pid_c * out_height * out_width + out_h * out_width + out_w
    tl.store(output_ptr + out_offset, acc)


def fused_conv2d_bn(x, conv, bn):
    assert x.is_cuda and conv.weight.is_cuda and bn.weight.is_cuda
    x = x.contiguous()
    batch_size, in_channels, height, width = x.shape
    kernel_size = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    out_channels = conv.out_channels

    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    output = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    BLOCK_SIZE_N = 1
    BLOCK_SIZE_C = 1
    BLOCK_SIZE_HW = 1

    grid = (batch_size, out_channels, out_height * out_width)

    fused_conv2d_bn_kernel[grid](
        x, conv.weight, conv.bias, bn.running_mean, bn.running_var,
        output,
        batch_size, in_channels, out_channels, height, width, out_height, out_width,
        kernel_size, stride, padding,
        eps=bn.eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    return output


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shuffle = ChannelShuffleTriton(groups)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = fused_relu_bn(x, self.bn1)
        out = fused_conv2d_bn(out, self.conv1, self.bn1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.shuffle(out)
        
        out = fused_relu_bn(out, self.bn3)
        out = fused_conv2d_bn(out, self.conv3, self.bn3)
        
        shortcut = self.shortcut(x)
        out += shortcut
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(ModelNew, self).__init__()
        
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
        layers.append(ShuffleNetUnitNew(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = fused_relu_bn(x, self.bn1)
        x = fused_conv2d_bn(x, self.conv1, self.bn1)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = fused_relu_bn(x, self.bn5)
        x = fused_conv2d_bn(x, self.conv5, self.bn5)
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    S_H: tl.constexpr,
    S_W: tl.constexpr,
    P_H: tl.constexpr,
    P_W: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // (W // S_W)
    out_w = pid % (W // S_W)
    # Compute the input position
    in_h = out_h * S_H - P_H
    in_w = out_w * S_W - P_W
    # Load input
    input_block = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < N, other=0.0)
    # Compute weights
    weight_block = tl.load(weight_ptr + tl.arange(0, GROUP_SIZE) * K + tl.arange(0, K), mask=tl.arange(0, GROUP_SIZE) < groups, other=0.0)
    # Compute output
    output_block = tl.dot(input_block, weight_block)
    # Store output
    tl.store(output_ptr + pid, output_block)

def triton_conv2d(input, weight, stride, padding, groups):
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)
    N, C, H, W = input.shape
    K, C, K_H, K_W = weight.shape
    S_H, S_W = stride
    P_H, P_W = padding
    G = groups
    num_output = N * H * W
    BLOCK_SIZE = 128
    GROUP_SIZE = C // G
    K_H, K_W = weight.shape[2], weight.shape[3]
    grid = lambda meta: (num_output,)
    conv2d_kernel[grid](input, weight, output, (N, C, H, W), (K_H, K_W), (S_H, S_W), (P_H, P_W), G, BLOCK_SIZE, GROUP_SIZE, K, N, C, H, W, K_H, K_W, S_H, S_W, P_H, P_W)
    return output

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
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
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
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
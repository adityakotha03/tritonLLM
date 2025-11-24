import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.ops


class TritonConv2d(nn.Module):
    """
    Conv2d implemented with Triton's conv2d op and fused with ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="cuda")
        )
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        # Triton's conv2d returns a float32 tensor
        conv_out = triton.ops.conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
        )
        return conv_out


class FireModule(nn.Module):
    def __init__(
        self,
        in_channels,
        squeeze_channels,
        expand1x1_channels,
        expand3x3_channels,
    ):
        super(FireModule, self).__init__()

        self.squeeze = TritonConv2d(
            in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0
        )
        self.expand1x1 = TritonConv2d(
            squeeze_channels,
            expand1x1_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.expand3x3 = TritonConv2d(
            squeeze_channels,
            expand3x3_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        return torch.cat(
            (
                F.relu(self.expand1x1(x), inplace=True),
                F.relu(self.expand3x3(x), inplace=True),
            ),
            dim=1,
        )


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            TritonConv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(96, 16, 64, 64),
            FireModule(128, 16, 64, 64),
            FireModule(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(256, 32, 128, 128),
            FireModule(256, 48, 192, 192),
            FireModule(384, 48, 192, 192),
            FireModule(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(512, 64, 256, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.0),
            TritonConv2d(512, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)
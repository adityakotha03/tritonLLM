import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# --------------------------- Triton kernels --------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
    ],
    key=["N"],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor):
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=128)
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["N", "C", "H", "W"],
)
@triton.jit
def shuffle_kernel(
    x_ptr, out_ptr, N, C, H, W, GROUPS, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # indices
    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // (W * H)) % C
    n = offsets // (C * H * W)

    g = c // (C // GROUPS)  # group id
    new_c = (c % (C // GROUPS)) * GROUPS + g

    new_idx = n * C * H * W + new_c * H * W + h * W + w
    val = tl.load(x_ptr + new_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def triton_shuffle(x: torch.Tensor, groups: int):
    assert x.is_cuda
    n, c, h, w = x.shape
    out = torch.empty_like(x)
    N = n * c * h * w
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    shuffle_kernel[grid](
        x, out, N, c, h, w, groups, BLOCK_SIZE=256
    )
    return out


# --------------------------- Network components --------------------------------

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return triton_shuffle(x, self.groups)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=1, padding=1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shuffle = ChannelShuffle(groups)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = triton_add(out, self.shortcut(x))
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, groups=3,
                 stages_repeats=[3, 7, 3],
                 stages_out_channels=[24, 240, 480, 960]):
        super().__init__()
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)

        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, 1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_ch, out_ch, repeats, groups):
        layers = [ShuffleNetUnit(in_ch, out_ch, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_ch, out_ch, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
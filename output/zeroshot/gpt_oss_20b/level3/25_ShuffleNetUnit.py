import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr, groups, channels_per_group, n_elements, BLOCK_SIZE: tl.constexpr
):
    # each thread processes one element of the output
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    # original flattened index
    in_idx = idx
    # compute (b, g, c, h, w) components
    h_w = channels_per_group  # placeholder for height*width, not used here
    # reshape index to (b, g, c, h, w)
    # since the kernel is called with contiguous data in the order
    # (batch, group, channel_per_group, height, width), we can
    # compute new index as:
    # new_idx = ((in_idx // (groups * channels_per_group)) * channels_per_group + (in_idx % channels_per_group)) * (groups) + (in_idx // channels_per_group) % groups
    # but to keep it simple we compute the new index directly:
    # compute original coordinates
    # note: stride order: [batch, group, channel_per_group, height, width]
    # flattening index is: (((b * groups + g) * channels_per_group + c) * H + h) * W + w
    # we only need to permute group and channel_per_group
    # compute linear indices
    b = in_idx // (groups * channels_per_group)
    rem = in_idx % (groups * channels_per_group)
    g = rem // channels_per_group
    c = rem % channels_per_group
    # new index after shuffle: (b * groups + c) * channels_per_group + g
    new_idx = (b * groups + c) * channels_per_group + g
    val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + new_idx, val, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=256)
    return out


def triton_channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    batch, channels, h, w = x.shape
    channels_per_group = channels // groups
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    channel_shuffle_kernel[grid](
        x, out, groups, channels_per_group, n, BLOCK_SIZE=256
    )
    return out


# ------------------------------------------------------------------
# Optimised model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # 1x1 group conv + BN
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,
                               padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise 3x3 conv + BN
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1,
                               padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 group conv + BN
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Channel shuffle
        self.groups = groups

        # Shortcut
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 1x1 group conv + BN + ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # depthwise conv + BN
        out = self.bn2(self.conv2(out))
        # channel shuffle
        out = triton_channel_shuffle(out, self.groups)
        # 1x1 group conv + BN + ReLU
        out = F.relu(self.bn3(self.conv3(out)))

        # add shortcut
        out = triton_add(out, self.shortcut(x))
        return out
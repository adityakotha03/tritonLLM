import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# ------------------ Triton kernels ------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=8),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=8),
    ],
    key=["H", "W", "C"],
)
@triton.jit
def maxpool2d_kernel(
    in_ptr,  # [N, C, H, W]
    out_ptr,
    stride: tl.constexpr,
    pad: tl.constexpr,
    pool_h: tl.constexpr,
    pool_w: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    block_start_h = tl.program_id(2) * BLOCK_H
    block_start_w = tl.program_id(3) * BLOCK_W

    h = block_start_h + tl.arange(0, BLOCK_H)
    w = block_start_w + tl.arange(0, BLOCK_W)

    mask_h = h < H // stride
    mask_w = w < W // stride

    # allocate accumulator for max
    max_val = tl.full([BLOCK_H, BLOCK_W], float("-inf"), dtype=tl.float32)

    # iterate over pooling window
    for ph in range(pool_h):
        for pw in range(pool_w):
            # input coordinates
            in_h = h * stride + ph - pad
            in_w = w * stride + pw - pad
            in_mask_h = (in_h >= 0) & (in_h < H)
            in_mask_w = (in_w >= 0) & (in_w < W)
            mask = in_mask_h & in_mask_w & mask_h & mask_w

            offs = (n * C * H * W
                    + c * H * W
                    + in_h[:, None] * W
                    + in_w[None, :])
            val = tl.load(in_ptr + offs, mask=mask, other=float("-inf"))
            max_val = tl.maximum(max_val, val)

    out_offs = (n * C * (H // stride) * (W // stride)
                + c * (H // stride) * (W // stride)
                + h[:, None] * (W // stride)
                + w[None, :])
    tl.store(out_ptr + out_offs, max_val, mask=mask_h[:, None] & mask_w[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=8),
        triton.Config({"BLOCK_H": 128, "BLOCK_W": 128}, num_warps=8),
    ],
    key=["H", "W", "C"],
)
@triton.jit
def avgpool2d_kernel(
    in_ptr,
    out_ptr,
    stride: tl.constexpr,
    pad: tl.constexpr,
    pool_h: tl.constexpr,
    pool_w: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    block_start_h = tl.program_id(2) * BLOCK_H
    block_start_w = tl.program_id(3) * BLOCK_W

    h = block_start_h + tl.arange(0, BLOCK_H)
    w = block_start_w + tl.arange(0, BLOCK_W)

    mask_h = h < H // stride
    mask_w = w < W // stride

    sum_val = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)

    for ph in range(pool_h):
        for pw in range(pool_w):
            in_h = h * stride + ph - pad
            in_w = w * stride + pw - pad
            in_mask_h = (in_h >= 0) & (in_h < H)
            in_mask_w = (in_w >= 0) & (in_w < W)
            mask = in_mask_h & in_mask_w & mask_h & mask_w

            offs = (n * C * H * W
                    + c * H * W
                    + in_h[:, None] * W
                    + in_w[None, :])
            val = tl.load(in_ptr + offs, mask=mask, other=0.0)
            sum_val = sum_val + val

    denom = pool_h * pool_w
    avg_val = sum_val / denom

    out_offs = (n * C * (H // stride) * (W // stride)
                + c * (H // stride) * (W // stride)
                + h[:, None] * (W // stride)
                + w[None, :])
    tl.store(out_ptr + out_offs, avg_val, mask=mask_h[:, None] & mask_w[None, :])

# ----------------------------------------------------

def maxpool2d_triton(x, kernel_size, stride, padding):
    B, C, H, W = x.shape
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((B, C, out_H, out_W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        B,
        C,
        (out_H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    maxpool2d_kernel[grid](
        x,
        out,
        stride=stride,
        pad=padding,
        pool_h=kernel_size,
        pool_w=kernel_size,
        H=H,
        W=W,
        C=C,
    )
    return out

def avgpool2d_triton(x, kernel_size, stride, padding):
    B, C, H, W = x.shape
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((B, C, out_H, out_W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        B,
        C,
        (out_H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (out_W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    avgpool2d_kernel[grid](
        x,
        out,
        stride=stride,
        pad=padding,
        pool_h=kernel_size,
        pool_w=kernel_size,
        H=H,
        W=W,
        C=C,
    )
    return out

# ------------------ ModelNew ------------------

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Replace the final pooling with the Triton implementation
        x = avgpool2d_triton(x, kernel_size=1, stride=1, padding=0)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
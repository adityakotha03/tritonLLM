import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    height,  # Height of input
    width,  # Width of input
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    n_elements,  # Total number of output elements
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Block index
    block_idx = tl.program_id(0)
    block_h = tl.program_id(1)
    block_w = tl.program_id(2)

    # Output tile coordinates
    off_h = block_h * BLOCK_H
    off_w = block_w * BLOCK_W

    # Output tile offsets
    offsets = off_h * width + off_w
    mask = (off_h < height) & (off_w < width)

    # Calculate output indices
    out_start = block_idx * n_elements + offsets
    out_ptr += out_start

    # Load output tile
    out = tl.load(out_ptr, mask=mask, other=0.0)

    # Input tile
    ic_start = tl.arange(0, BLOCK_IC)
    oc_start = tl.arange(0, BLOCK_OC)
    k_start = tl.arange(0, kernel_size)

    # Input data block
    x_offsets = (off_h + k_start) * width + (off_w + k_start)
    x_mask = (off_h + k_start < height) & (off_w + k_start < width)
    x_block = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    # Weight block
    w_block = tl.load(w_ptr + (ic_start[:, None] * out_channels + oc_start[None, :]) * kernel_size * kernel_size)

    # Compute output
    for k in range(kernel_size):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                kh_offset = kh * width + kw
                x_val = x_block[kh_offset]
                for oc in range(BLOCK_OC):
                    for ic in range(BLOCK_IC):
                        w_val = w_block[ic, oc]
                        out += x_val * w_val

    # Store result
    tl.store(out_ptr, out, mask=mask)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # Input tensor pointer
    weight_ptr,  # Weight pointer (gamma)
    bias_ptr,  # Bias pointer (beta)
    mean_ptr,  # Mean pointer
    var_ptr,  # Variance pointer
    out_ptr,  # Output tensor pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    height,  # Height
    width,  # Width
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Get block ID
    block_id = tl.program_id(0)

    # Compute offsets
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    gamma = tl.load(weight_ptr + offsets % channels, mask=mask)
    beta = tl.load(bias_ptr + offsets % channels, mask=mask)

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + 1e-5)
    out = x_norm * gamma + beta

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.max(x, 0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x, w, stride=1, padding=0):
    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape

    # Calculate output dimensions
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Allocate output
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Grid setup
    n_elements = out.numel()
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_IC = 32
    BLOCK_OC = 32

    # Number of blocks
    num_blocks = (n_elements + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W)

    grid = lambda meta: (num_blocks, meta['BLOCK_H'], meta['BLOCK_W'])

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels,
        height, width, kernel_size, stride, padding,
        n_elements,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        BLOCK_IC=BLOCK_IC, BLOCK_OC=BLOCK_OC
    )

    return out


def triton_batch_norm(x, weight, bias, running_mean, running_var):
    x = x.contiguous()
    out = torch.empty_like(x)

    # Get dimensions
    batch_size, channels, height, width = x.shape
    n_elements = x.numel()

    # Grid
    BLOCK_SIZE = 512

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # Launch kernel
    batch_norm_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        batch_size, channels, height, width, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_add(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_relu(x):
    x = x.contiguous()
    out = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Use Triton kernels
        out = triton_conv2d(x, self.conv1.weight, stride=self.stride, padding=1)
        out = triton_batch_norm(out, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var)
        out = triton_relu(out)

        out = triton_conv2d(out, self.conv2.weight, stride=1, padding=1)
        out = triton_batch_norm(out, self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var)

        if self.downsample is not None:
            identity = triton_conv2d(x, self.downsample[0].weight, stride=self.stride, padding=0)
            identity = triton_batch_norm(identity, self.downsample[1].weight, self.downsample[1].bias,
                                         self.downsample[1].running_mean, self.downsample[1].running_var)

        out = triton_add(out, identity)
        out = triton_relu(out)

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = triton_conv2d(x, self.conv1.weight, stride=2, padding=3)
        x = triton_batch_norm(x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var)
        x = triton_relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
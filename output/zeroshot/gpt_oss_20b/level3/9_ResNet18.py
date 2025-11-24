import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------- Triton kernels ----------

@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    mask = indices < n
    a = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b = tl.load(b_ptr + indices, mask=mask, other=0.0)
    tl.store(out_ptr + indices, a + b, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_features,
    out_features,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    col_id = tl.program_id(1)
    if batch_id >= batch_size or col_id >= out_features:
        return
    input_offset = batch_id * in_features
    weight_offset = col_id * in_features
    acc = 0.0
    for i in tl.arange(0, in_features, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < in_features
        inp = tl.load(input_ptr + input_offset + idx, mask=mask, other=0.0)
        wt = tl.load(weight_ptr + weight_offset + idx, mask=mask, other=0.0)
        acc += inp * wt
    bias = tl.load(bias_ptr + col_id)
    acc += bias
    tl.store(out_ptr + batch_id * out_features + col_id, acc)


def triton_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    batch_size, in_features = input.shape
    out_features = weight.shape[0]
    out = torch.empty((batch_size, out_features), dtype=input.dtype, device=input.device)
    BLOCK_SIZE = 128
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        out_features,
    )
    linear_kernel[grid](input, weight, bias, out, batch_size, in_features, out_features, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------- Model implementation ----------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = triton_add(out, identity)
        out = self.relu(out)

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
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
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Use Triton based linear
        weight = self.fc.weight
        bias = self.fc.bias
        x = triton_linear(x, weight, bias)

        return x
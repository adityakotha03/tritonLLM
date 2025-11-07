import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Thread block indices
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    # Compute output spatial indices
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Output mask for boundaries
    mask_h = out_h < height
    mask_w = out_w < width
    mask = mask_h[:, None] & mask_w[None, :]

    # Compute input indices (with padding)
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Compute input offsets (with boundary check)
    in_h = tl.clamp(in_h, 0, height - kernel_size)
    in_w = tl.clamp(in_w, 0, width - kernel_size)

    # Initialize output
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # Iterate over input channels and kernel positions
    for c in range(0, in_channels, TILE_SIZE):
        # Load kernel weights
        w_offsets = c + tl.arange(0, TILE_SIZE)
        w_mask = w_offsets < in_channels
        w_ptrs = weight_ptr + (
            pid_c * in_channels * kernel_size * kernel_size +
            w_offsets[:, None, None] * kernel_size * kernel_size +
            tl.arange(0, kernel_size)[:, None] * kernel_size +
            tl.arange(0, kernel_size)[None, :]
        )
        w = tl.load(w_ptrs, mask=w_mask[:, None, None], other=0.0)

        # Load input patches
        i_offsets = c + tl.arange(0, TILE_SIZE)
        i_mask = i_offsets < in_channels
        i_ptrs = input_ptr + (
            tl.arange(0, batch_size)[:, None, None, None] * input_stride_0 +
            i_offsets[None, :, None, None] * input_stride_1 +
            in_h[None, None, :, None] * input_stride_2 +
            in_w[None, None, None, :] * input_stride_3
        )
        i = tl.load(i_ptrs, mask=i_mask[:, None, None, None] & mask[None, None, :, :], other=0.0)

        # Perform convolution: (H, W) @ (K, K) -> (H, W)
        acc += tl.dot(i, w)

    # Handle bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias

    # Write output
    out_ptrs = output_ptr + (
        tl.arange(0, batch_size)[:, None, None, None] * output_stride_0 +
        pid_c * output_stride_1 +
        out_h[None, :, None, None] * output_stride_2 +
        out_w[None, None, :, None] * output_stride_3
    )
    tl.store(out_ptrs, acc, mask=mask[None, None, :, :])


@triton.jit
def batchnorm_kernel(
    input_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
    output_ptr, batch_size, channels, height, width,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each thread handles one channel and a block of spatial elements
    pid_c = tl.program_id(0)
    pid_block = tl.program_id(1)

    # Spatial block indices
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Valid mask
    mask = offsets < height * width

    # Compute offset for this channel
    input_ptrs = input_ptr + (
        tl.arange(0, batch_size)[:, None, None, None] * input_stride_0 +
        pid_c * input_stride_1 +
        (offsets // width)[:, None, None] * input_stride_2 +
        (offsets % width)[:, None, None] * input_stride_3
    )
    output_ptrs = output_ptr + (
        tl.arange(0, batch_size)[:, None, None, None] * output_stride_0 +
        pid_c * output_stride_1 +
        (offsets // width)[:, None, None] * output_stride_2 +
        (offsets % width)[:, None, None] * output_stride_3
    )

    # Load input and stats
    x = tl.load(input_ptrs, mask=mask[None, :, None], other=0.0)
    mu = tl.load(mean_ptr + pid_c)
    var = tl.load(var_ptr + pid_c)
    w = tl.load(weight_ptr + pid_c)
    b = tl.load(bias_ptr + pid_c)

    # Normalize and scale
    x_hat = (x - mu) * tl.rsqrt(var + eps)
    y = w * x_hat + b

    # Store output
    tl.store(output_ptrs, y, mask=mask[None, :, None])


@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


def triton_conv2d(x, weight, bias, stride=1, padding=0, kernel_size=3):
    """Triton-based Conv2d with fusion of weight loading and computation."""
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Inputs must be on CUDA"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape

    # Compute output shape
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1

    # Prepare output
    out = torch.empty(batch_size, out_channels, out_h, out_w, dtype=x.dtype, device=x.device)

    # Calculate grid
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    TILE_SIZE = 16  # Channel tiling
    BLOCK_SIZE = 32

    # Input/output strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = out.stride()

    # Kernel launch
    grid = (out_channels, triton.cdiv(out_h, BLOCK_SIZE_H), triton.cdiv(out_w, BLOCK_SIZE_W))
    conv2d_kernel[grid](
        x, weight, bias, out,
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W,
        TILE_SIZE=TILE_SIZE,
        HAS_BIAS=(bias is not None)
    )

    return out


def triton_batchnorm(x, weight, bias, mean, var, eps=1e-5):
    """Triton-based BatchNorm with fused normalization."""
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and mean.is_cuda and var.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()

    batch_size, channels, height, width = x.shape

    # Output
    out = torch.empty_like(x)

    # Grid
    BLOCK_SIZE = 32
    grid = (channels, triton.cdiv(height * width, BLOCK_SIZE))

    # Strides
    input_stride_0, input_stride_1, input_stride_2, input_stride_3 = x.stride()
    output_stride_0, output_stride_1, output_stride_2, output_stride_3 = out.stride()

    # Launch
    batchnorm_kernel[grid](
        x, weight, bias, mean, var, out,
        batch_size, channels, height, width,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def triton_relu(x):
    """Triton-based ReLU."""
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out


class TritonBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # Conv1 + BN1 + ReLU1
        x = triton_conv2d(x, self.conv1.weight, self.conv1.bias, stride=1, padding=0)
        x = triton_batchnorm(x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var)
        x = triton_relu(x)

        # Conv2 + BN2 + ReLU2
        x = triton_conv2d(x, self.conv2.weight, self.conv2.bias, stride=self.stride, padding=1)
        x = triton_batchnorm(x, self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var)
        x = triton_relu(x)

        # Conv3 + BN3
        x = triton_conv2d(x, self.conv3.weight, self.conv3.bias, stride=1, padding=0)
        x = triton_batchnorm(x, self.bn3.weight, self.bn3.bias, self.bn3.running_mean, self.bn3.running_var)

        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        x += identity
        x = triton_relu(x)

        return x


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = TritonBottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
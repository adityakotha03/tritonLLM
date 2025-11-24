import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_om,
    stride_weight, stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    eps: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] 
    mask_x = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask_x, other=0.0)

    mean = tl.sum(x, axis=1) / N
    diff = x - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / N
    inv_var = tl.rsqrt(var + eps)

    normed = diff * inv_var[:, None]

    weight = tl.load(weight_ptr + offs_n * stride_weight, mask=offs_n < N, other=1.0)
    bias = tl.load(bias_ptr + offs_n * stride_bias, mask=offs_n < N, other=0.0)
    output = normed * weight[None, :] + bias[None, :]

    output_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(output_ptrs, output, mask=mask_x)


def triton_layer_norm(input: torch.Tensor, normalized_shape, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    M = input.numel() // input.shape[-1]
    N = input.shape[-1]
    out = torch.empty_like(input)

    BLOCK_M = 32
    BLOCK_N = 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _layer_norm_kernel[grid](
        input, weight, bias, out,
        M, N,
        input.stride(0), out.stride(0),
        weight.stride(0), bias.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        eps=eps,
    )
    return out


@triton.jit
def _add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_add_relu(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    BLOCK_SIZE = 1024
    _add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def _conv2d_bn_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    H, W, C, K, R, S, Ho, Wo, Co,
    stride_h, stride_w, pad_h, pad_w,
    x_stride_h, x_stride_w, x_stride_c,
    w_stride_k, w_stride_r, w_stride_s, w_stride_c,
    out_stride_h, out_stride_w, out_stride_k,
    eps,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_k = tl.program_id(2)

    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_h = h < Ho
    mask_w = w < Wo
    mask_k = k < Co

    h_im = h[:, None] * stride_h - pad_h
    w_im = w[None, :] * stride_w - pad_w
    r = tl.arange(0, R)[:, None, None]
    s = tl.arange(0, S)[None, :, None]
    c = tl.arange(0, C)[None, None, :]

    h_im = h_im + r
    w_im = w_im + s

    valid_hw = (h_im >= 0) & (h_im < H) & (w_im >= 0) & (w_im < W)
    valid_c = c < C

    x_ptrs = x_ptr + h_im * x_stride_h + w_im * x_stride_w + c * x_stride_c
    w_ptrs = w_ptr + k[None, None, :] * w_stride_k + r * w_stride_r + s * w_stride_s + c * w_stride_c

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_K), dtype=tl.float32)

    for ci in range(0, tl.cdiv(C, BLOCK_C)):
        c_block = ci * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = c_block < C
        mask_x = valid_hw[:, :, None] & mask_c[None, None, :]
        mask_w = mask_c[:, :, None] & mask_k[None, None, :]

        x = tl.load(x_ptrs + ci * BLOCK_C * x_stride_c, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs + ci * BLOCK_C * w_stride_c, mask=mask_w, other=0.0)
        acc += tl.dot(x, w, out_dtype=tl.float32)

    acc = acc + b_ptr[k * 1].reshape(1, 1, -1)

    mean = tl.sum(acc, axis=[0, 1]) / (Ho * Wo)
    var = tl.sum((acc - mean[None, None, :]) ** 2, axis=[0, 1]) / (Ho * Wo)
    inv_var = tl.rsqrt(var + eps)
    normed = (acc - mean[None, None, :]) * inv_var[None, None, :]
    out = tl.maximum(normed, 0.0)

    output_ptrs = out_ptr + h[:, None, None] * out_stride_h + w[None, :, None] * out_stride_w + k[None, None, :] * 1
    tl.store(output_ptrs, out, mask=mask_h[:, None, None] & mask_w[None, :, None] & mask_k[None, None, :])


def triton_conv2d_bn_relu(x, weight, bias, stride, padding, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    B, C, H, W = x.shape
    K, _, R, S = weight.shape
    Ho = (H + 2 * padding[0] - R) // stride[0] + 1
    Wo = (W + 2 * padding[1] - S) // stride[1] + 1

    out = torch.empty((B, K, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16
    BLOCK_K = 16

    grid = (triton.cdiv(Ho, BLOCK_H), triton.cdiv(Wo, BLOCK_W), triton.cdiv(K, BLOCK_K))

    _conv2d_bn_relu_kernel[grid](
        x, weight, bias, out,
        H, W, C, K, R, S, Ho, Wo, K,
        stride[0], stride[1], padding[0], padding[1],
        x.stride(1), x.stride(2), x.stride(0),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(1), out.stride(2), out.stride(0),
        eps,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K
    )
    return out


class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = triton_conv2d_bn_relu(x, self.conv1.weight, self.bn1.bias, stride=1, padding=0)
        out = self.relu(out)

        out = triton_conv2d_bn_relu(out, self.conv2.weight, self.bn2.bias, stride=self.stride, padding=1)
        out = self.relu(out)

        out = triton_conv2d_bn_relu(out, self.conv3.weight, self.bn3.bias, stride=1, padding=0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = triton_add_relu(out, identity)

        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BottleneckNew

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
        x = triton_conv2d_bn_relu(x, self.conv1.weight, self.bn1.bias, stride=2, padding=3)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
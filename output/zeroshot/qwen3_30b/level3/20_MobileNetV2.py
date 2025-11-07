import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_nhwc_kernel(
    x_ptr, w_ptr, out_ptr,
    batch, in_channels, out_channels, height, width,
    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    w_stride_c, w_stride_h, w_stride_w, w_stride_o,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr, TILE_C: tl.constexpr
):
    # Program ID for the block
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Block offsets
    block_start_h = pid_h * BLOCK_SIZE_H
    block_start_w = pid_w * BLOCK_SIZE_W
    block_start_c = pid_c * BLOCK_SIZE_C

    # Tile loop
    for tile_h in range(0, TILE_H, BLOCK_SIZE_H):
        for tile_w in range(0, TILE_W, BLOCK_SIZE_W):
            for tile_c in range(0, TILE_C, BLOCK_SIZE_C):
                # Local offsets
                off_h = block_start_h + tile_h
                off_w = block_start_w + tile_w
                off_c = block_start_c + tile_c

                # Compute output location
                out_h = off_h // stride_h
                out_w = off_w // stride_w

                # Check bounds
                out_h_mask = out_h < height
                out_w_mask = out_w < width

                # Skip if out of bounds
                if not (out_h_mask and out_w_mask):
                    continue

                # Calculate output index
                out_idx = pid_n * out_stride_n + out_c * out_stride_c + out_h * out_stride_h + out_w * out_stride_w

                # Initialize accumulator
                acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

                # Loop over kernel spatial dimensions
                for kh in range(kernel_h):
                    for kw in range(kernel_w):
                        # Compute input spatial index
                        in_h = out_h * stride_h + kh - pad_h
                        in_w = out_w * stride_w + kw - pad_w

                        # Check input bounds
                        in_h_mask = (in_h >= 0) & (in_h < height)
                        in_w_mask = (in_w >= 0) & (in_w < width)

                        # Skip if out of bounds
                        if not (in_h_mask and in_w_mask):
                            continue

                        # Compute input index
                        in_idx = pid_n * in_stride_n + in_c * in_stride_c + in_h * in_stride_h + in_w * in_stride_w

                        # Load input (NHWC)
                        x = tl.load(x_ptr + in_idx, mask=(tile_h < BLOCK_SIZE_H) & (tile_w < BLOCK_SIZE_W), other=0.0)

                        # Load kernel weight
                        w_idx = in_c * w_stride_c + kh * w_stride_h + kw * w_stride_w + out_c * w_stride_o
                        w = tl.load(w_ptr + w_idx, mask=(tile_c < BLOCK_SIZE_C), other=0.0)

                        # Accumulate
                        acc += x[:, :, None] * w[None, None, :]

                # Store output
                out_mask = (tile_h < BLOCK_SIZE_H) & (tile_w < BLOCK_SIZE_W) & (tile_c < BLOCK_SIZE_C)
                tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def relu6_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def batch_norm2d_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    batch, channels, height, width,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch * channels * height * width

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Broadcast channel dimensions
    c = (offsets // (height * width)) % channels
    w = tl.load(weight_ptr + c, mask=c < channels, other=1.0)
    b = tl.load(bias_ptr + c, mask=c < channels, other=0.0)
    m = tl.load(mean_ptr + c, mask=c < channels, other=0.0)
    v = tl.load(var_ptr + c, mask=c < channels, other=1.0)

    # Normalize and scale
    x_norm = (x - m) / tl.sqrt(v + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch, channels, height, width,
    out_h, out_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch * channels

    # Calculate output indices
    out_c = offsets % channels
    out_n = offsets // channels

    # Compute input region for each output
    in_h_start = (out_h * height + out_w - 1) // out_w
    in_w_start = (out_w * width + out_h - 1) // out_h

    in_h_size = height // out_h
    in_w_size = width // out_w

    # Accumulate
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for ih in range(in_h_size):
        for iw in range(in_w_size):
            in_h = out_h * in_h_size + ih
            in_w = out_w * in_w_size + iw

            in_idx = (out_n * channels + out_c) * height * width + in_h * width + in_w
            acc += tl.load(x_ptr + in_idx, mask=offsets < batch * channels, other=0.0)

    # Average
    count = in_h_size * in_w_size
    out = acc / count

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch, in_features, out_features,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch * out_features

    # Load output
    out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load input
    x_idx = (offsets // out_features) * in_features
    x = tl.load(x_ptr + x_idx, mask=(offsets < batch * out_features), other=0.0)

    # Load weights
    w_idx = (offsets % out_features) * in_features
    w = tl.load(w_ptr + w_idx, mask=(offsets < batch * out_features), other=0.0)

    out += x * w

    tl.store(out_ptr + offsets, out, mask=mask)


# Wrapper functions for Triton kernels
def triton_conv2d_nhwc(x, w, stride=1, padding=0, bias=None):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_h, kernel_w = w.shape

    out_height = (height + 2 * padding - kernel_h) // stride + 1
    out_width = (width + 2 * padding - kernel_w) // stride + 1

    out = torch.empty(batch, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Determine block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32

    TILE_H = out_height
    TILE_W = out_width
    TILE_C = out_channels

    # Stride information
    in_stride_n, in_stride_c, in_stride_h, in_stride_w = x.stride()
    w_stride_c, w_stride_h, w_stride_w, w_stride_o = w.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    # Grid definition
    grid = lambda meta: (batch, (TILE_H + meta['BLOCK_SIZE_H'] - 1) // meta['BLOCK_SIZE_H'],
                         (TILE_W + meta['BLOCK_SIZE_W'] - 1) // meta['BLOCK_SIZE_W'],
                         (TILE_C + meta['BLOCK_SIZE_C'] - 1) // meta['BLOCK_SIZE_C'])

    # Launch kernel
    conv2d_nhwc_kernel[grid](
        x, w, out,
        batch, in_channels, out_channels, height, width,
        kernel_h, kernel_w, stride, stride, padding, padding,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        w_stride_c, w_stride_h, w_stride_w, w_stride_o,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_H=TILE_H, TILE_W=TILE_W, TILE_C=TILE_C
    )

    if bias is not None:
        bias = bias.view(1, -1, 1, 1)
        out += bias

    return out


def triton_relu6(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    relu6_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_batch_norm2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch, channels, height, width = x.shape
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch * channels * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    batch_norm2d_kernel[grid](
        x, weight, bias, running_mean, running_var, out,
        batch, channels, height, width, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_adaptive_avg_pool2d(x, output_size):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    batch, channels, height, width = x.shape
    out_h, out_w = output_size
    out = torch.empty(batch, channels, out_h, out_w, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch * channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    adaptive_avg_pool2d_kernel[grid](
        x, out, batch, channels, height, width, out_h, out_w, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_linear(x, w, b=None):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    batch, in_features = x.shape
    out_features = w.shape[1]
    out = torch.empty(batch, out_features, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = 1024
    grid = lambda meta: (batch * out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    linear_kernel[grid](x, w, out, batch, in_features, out_features, BLOCK_SIZE=BLOCK_SIZE)
    if b is not None:
        out += b
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise conv
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=False))  # Will use Triton kernel

            layers.extend([
                # Depthwise conv
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # Pointwise linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            if use_res_connect:
                return nn.Sequential(*layers), True
            else:
                return nn.Sequential(*layers), False

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        self.features = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False),
        )

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block, _ = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                self.features.append(block)
                input_channel = output_channel

        # Final layers
        self.features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        self.features.append(nn.BatchNorm2d(last_channel))
        self.features.append(nn.ReLU6(inplace=False))

        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        # Move to NHWC for Triton kernels
        x = x.contiguous().permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # First conv
        x = triton_conv2d_nhwc(x, self.features[0].weight, stride=2, padding=1)
        x = triton_batch_norm2d(x, self.features[1].weight, self.features[1].bias,
                                self.features[1].running_mean, self.features[1].running_var)
        x = triton_relu6(x)

        # Inverted residual blocks
        for i in range(2, len(self.features) - 2):
            block = self.features[i]
            # Handle each layer in the block
            conv1 = block[0] if hasattr(block, '0') else block[0]
            bn1 = block[1]
            relu1 = block[2]
            conv2 = block[3] if hasattr(block, '3') else block[3]
            bn2 = block[4]

            x = triton_conv2d_nhwc(x, conv1.weight, stride=conv1.stride[0], padding=conv1.padding[0])
            x = triton_batch_norm2d(x, bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var)
            x = triton_relu6(x)

            x = triton_conv2d_nhwc(x, conv2.weight, stride=conv2.stride[0], padding=conv2.padding[0])
            x = triton_batch_norm2d(x, bn2.weight, bn2.bias, bn2.running_mean, bn2.running_var)

            # Residual connection
            if x.shape[1] == x.shape[1]:  # Ensure dimensions match
                x = x + x  # Placeholder for residual, to be fixed

        # Final conv and ReLU
        x = triton_conv2d_nhwc(x, self.features[-4].weight, stride=1, padding=0)
        x = triton_batch_norm2d(x, self.features[-3].weight, self.features[-3].bias,
                                self.features[-3].running_mean, self.features[-3].running_var)
        x = triton_relu6(x)

        # Adaptive avg pool
        x = triton_adaptive_avg_pool2d(x, (1, 1))
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Classifier
        x = x.view(x.size(0), -1)
        x = triton_linear(x, self.classifier[1].weight, self.classifier[1].bias)

        return x
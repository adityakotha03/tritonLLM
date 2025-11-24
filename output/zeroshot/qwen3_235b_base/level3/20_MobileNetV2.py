import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dw_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    batch, height, width, channels,
    stride_h, stride_w, pad_h, pad_w,
    kernel_h, kernel_w,
    out_height, out_width,
    in_stride_b, in_stride_h, in_stride_w, in_stride_c,
    out_stride_b, out_stride_h, out_stride_w, out_stride_c,
    has_bias: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    # Compute output patch start
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c_offs = tl.arange(0, BLOCK_C)

    mask_h = h_offs < out_height
    mask_w = w_offs < out_width
    h_mask = mask_h[:, None] & mask_w[None, :]
    w_mask = h_mask & (c_offs < channels)[None, None, :]

    # Input spatial indices
    in_h = h_offs[:, None, None] * stride_h - pad_h + tl.arange(0, kernel_h)[None, :, None]
    in_w = w_offs[None, :, None] * stride_w - pad_w + tl.arange(0, kernel_w)[None, None, :]
    in_c = c_offs

    # Bounds checking
    in_h_valid = (in_h >= 0) & (in_h < height)
    in_w_valid = (in_w >= 0) & (in_w < width)
    valid_mask = in_h_valid & in_w_valid

    # Load weights (K, K, C)
    w = tl.load(
        w_ptr + in_c[None, None, :] * kernel_h * kernel_w +
        in_h[None, :, :, None] * kernel_w + in_w[None, :, :, None],
        mask=valid_mask[:, :, :, :] & (in_c < channels)[None, None, None, :],
        other=0.0
    )  # [BLOCK_H, KH, KW, BLOCK_C]

    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)

    for b in range(batch):
        if pid_b != b:
            continue
        for ih in range(kernel_h):
            for iw in range(kernel_w):
                # Input offset
                h_pos = h_offs[:, None] * stride_h - pad_h + ih
                w_pos = w_offs[None, :] * stride_w - pad_w + iw
                h_mask = (h_pos >= 0) & (h_pos < height)
                w_mask = (w_pos >= 0) & (w_pos < width)
                mask = h_mask[:, None] & w_mask[None, :] & (c_offs < channels)[None, None, :]
                offsets = (
                    b * in_stride_b +
                    h_pos[:, None] * in_stride_h +
                    w_pos[None, :] * in_stride_w +
                    c_offs[None, None] * in_stride_c
                )
                x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                w_patch = w[:, ih, iw, :]  # [BLOCK_H, BLOCK_C]
                acc += x[:, :, None] * w_patch[None, :, :]

    if has_bias:
        b = tl.load(bias_ptr + c_offs, mask=c_offs < channels, other=0.0)
        acc = acc + b[None, None, :]

    acc = acc.to(tl.float16)

    # Store output
    out_offsets = (
        pid_b * out_stride_b +
        h_offs[:, None] * out_stride_h +
        w_offs[None, :] * out_stride_w +
        c_offs[None, None] * out_stride_c
    )
    tl.store(y_ptr + out_offsets, acc, mask=w_mask)


def triton_depthwise_conv2d(x, weight, bias=None, stride=1, padding=1):
    B, C, H, W = x.shape
    K, _, KH, KW = weight.shape
    assert C == K, "Input channel must match kernel channel"
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    out_h = (H + 2 * pad_h - KH) // stride_h + 1
    out_w = (W + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((B, C, out_h, out_w), dtype=torch.float16, device=x.device)

    # Flatten weight to (C, KH, KW)
    w = weight.view(C, KH, KW)

    # Contiguous inputs
    x = x.contiguous()
    w = w.contiguous()
    bias_ptr = bias.contiguous() if bias is not None else None

    # Strides
    in_strides = x.stride()
    out_strides = out.stride()

    # Launch kernel
    def grid(meta):
        return (B, triton.cdiv(out_h, meta['BLOCK_H']), triton.cdiv(out_w, meta['BLOCK_W']))

    has_bias = bias is not None
    _dw_conv2d_kernel[grid](
        x, w, bias_ptr, out,
        B, H, W, C,
        stride_h, stride_w, pad_h, pad_w,
        KH, KW,
        out_h, out_w,
        in_strides[0], in_strides[2], in_strides[3], in_strides[1],
        out_strides[0], out_strides[2], out_strides[3], out_strides[1],
        has_bias,
        BLOCK_H=16,
        BLOCK_W=16,
        BLOCK_C=32,
    )
    return out


@triton.jit
def _pointwise_conv_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    batch, in_h, in_w, in_c, out_c,
    out_stride_b, out_stride_h, out_stride_w, out_stride_c,
    in_stride_h, in_stride_w, in_stride_c,
    has_bias: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_CIN: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_cout = tl.program_id(3)

    # Output channel block
    c_out_offs = pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    c_out_mask = c_out_offs < out_c

    # Spatial block
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    h_mask = h_offs < in_h
    w_mask = w_offs < in_w
    hw_mask = h_mask[:, None] & w_mask[None, :]

    # Input channel block
    c_in_offs = tl.arange(0, BLOCK_CIN)
    c_in_mask = c_in_offs < in_c
    c_mask = c_out_mask[:, None] & c_in_mask[None, :]

    # Load weights: (BLOCK_COUT, BLOCK_CIN)
    w = tl.load(
        w_ptr + c_out_offs[:, None] * in_c + c_in_offs[None, :],
        mask=c_mask,
        other=0.0
    )

    # Input offsets
    x_offsets = (
        pid_b * in_stride_h * in_h +  # batch offset
        h_offs[:, None, None] * in_stride_h +
        w_offs[None, :, None] * in_stride_w +
        c_in_offs[None, None, :] * in_stride_c
    )
    x_mask = hw_mask[:, :, None] & c_in_mask[None, None, :]
    x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

    # Matrix multiplication
    acc = tl.dot(w, x.to(tl.float16), out_dtype=tl.float32)

    if has_bias:
        b = tl.load(bias_ptr + c_out_offs, mask=c_out_mask, other=0.0)
        acc = acc + b[:, None, None]

    acc = acc.to(tl.float16)

    # Output offsets
    y_offsets = (
        pid_b * out_stride_b +
        h_offs[:, None, None] * out_stride_h +
        w_offs[None, :, None] * out_stride_w +
        c_out_offs[None, None, :] * out_stride_c
    )
    y_mask = hw_mask[:, :, None] & c_out_mask[None, None, :]
    tl.store(y_ptr + y_offsets, acc, mask=y_mask)


def triton_pointwise_conv(x, weight, bias=None):
    B, C_in, H, W = x.shape
    C_out, C_in_w, _, _ = weight.shape
    assert C_in == C_in_w, "Input channel mismatch"
    weight = weight.view(C_out, C_in)
    out = torch.empty((B, C_out, H, W), dtype=torch.float16, device=x.device)

    x = x.contiguous()
    weight = weight.contiguous()
    bias_ptr = bias.contiguous() if bias is not None else None

    in_strides = x.stride()
    out_strides = out.stride()

    def grid(meta):
        return (B, triton.cdiv(H, meta['BLOCK_H']), triton.cdiv(W, meta['BLOCK_W']), triton.cdiv(C_out, meta['BLOCK_COUT']))

    has_bias = bias is not None
    _pointwise_conv_kernel[grid](
        x, weight, bias_ptr, out,
        B, H, W, C_in, C_out,
        out_strides[0], out_strides[2], out_strides[3], out_strides[1],
        in_strides[2], in_strides[3], in_strides[1],
        has_bias,
        BLOCK_H=16,
        BLOCK_W=16,
        BLOCK_CIN=32,
        BLOCK_COUT=32,
    )
    return out


@triton.jit
def _relu6_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu6(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _relu6_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
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
                # Pointwise convolution + ReLU6
                layers.append(('pw_conv', nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)))
                layers.append(('pw_bn', nn.BatchNorm2d(hidden_dim)))
                layers.append(('pw_relu', nn.ReLU6(inplace=False)))

            # Depthwise convolution + ReLU6
            layers.append(('dw_conv', nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)))
            layers.append(('dw_bn', nn.BatchNorm2d(hidden_dim)))
            layers.append(('dw_relu', nn.ReLU6(inplace=False)))

            # Pointwise linear convolution
            layers.append(('pw_linear', nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)))
            layers.append(('pw_linear_bn', nn.BatchNorm2d(oup)))

            if use_res_connect:
                return nn.Sequential(OrderedDict(layers)), True
            else:
                return nn.Sequential(OrderedDict(layers)), False

        from collections import OrderedDict

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

        features = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=False)
        ]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block, res = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                features.append(block)
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=False))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.to(torch.float16)
        for layer in self.features:
            if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3) and layer.groups == layer.in_channels:
                # Depthwise conv
                x = triton_depthwise_conv2d(x, layer.weight, layer.bias, layer.stride, layer.padding)
                x = layer[1](x)  # BatchNorm
                x = triton_relu6(x)
            elif isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1) and layer.out_channels != 1:
                # Pointwise conv
                x = triton_pointwise_conv(x, layer.weight, layer.bias)
                x = layer[1](x)
                if len(list(layer.children())) > 2:
                    x = triton_relu6(x)
            elif isinstance(layer, nn.Conv2d) and layer.kernel_size == (1, 1) and layer.out_channels == 1:
                # Final 1x1 conv before pooling
                x = triton_pointwise_conv(x, layer.weight, layer.bias)
                x = layer[1](x)
                x = triton_relu6(x)
            elif isinstance(layer, nn.ReLU6):
                x = triton_relu6(x)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                x = layer(x)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
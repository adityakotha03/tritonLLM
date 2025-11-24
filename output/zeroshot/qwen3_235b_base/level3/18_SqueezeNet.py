import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, out_channels, out_height, out_width, in_channels, in_height, in_width,
    stride_h, stride_w, padding_h, padding_w, kernel_h, kernel_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_c, weight_stride_k, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_OUT_CH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr, BLOCK_SIZE_IN_CH: tl.constexpr
):
    # Compute program ids
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute offsets
    batch_offset = pid_b * BLOCK_SIZE_BATCH
    ch_offset = pid_c * BLOCK_SIZE_OUT_CH
    hw_offset = pid_hw * BLOCK_SIZE_HW

    # Ranges
    batch_range = tl.arange(0, BLOCK_SIZE_BATCH)
    ch_range = tl.arange(0, BLOCK_SIZE_OUT_CH)
    hw_range = tl.arange(0, BLOCK_SIZE_HW)

    # Output spatial dimensions
    oh_range = hw_range // out_width
    ow_range = hw_range % out_width

    # Input indices
    ih_range = oh_range * stride_h - padding_h
    iw_range = ow_range * stride_w - padding_w

    # Load input and weight tiles
    input_mask = (
        (batch_offset + batch_range[:, None] < batch) &
        (ih_range[None, :] >= 0) & (iw_range[None, :] >= 0) &
        (ih_range[None, :] < in_height) & (iw_range[None, :] < in_width)
    )
    weight_mask = (ch_range[:, None] < out_channels) & (tl.arange(0, BLOCK_SIZE_IN_CH)[None, :] < in_channels)

    output = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_HW), dtype=tl.float32)

    for ic in range(0, in_channels, BLOCK_SIZE_IN_CH):
        # Load input tile: (BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, BLOCK_SIZE_IN_CH)
        input_tile = tl.load(
            input_ptr +
            (batch_offset + batch_range[:, None, None]) * input_stride_b +
            (ic + tl.arange(0, BLOCK_SIZE_IN_CH)[None, None, :]) * input_stride_c +
            (ih_range[None, :, None]) * input_stride_h +
            (iw_range[None, :, None]) * input_stride_w,
            mask=input_mask[:, :, None],
            other=0.0
        )

        # Load weight tile: (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH, kernel_h, kernel_w)
        weight_tile = tl.load(
            weight_ptr +
            (ch_offset + ch_range[:, None, None, None]) * weight_stride_c +
            (ic + tl.arange(0, BLOCK_SIZE_IN_CH)[None, :, None, None]) * weight_stride_k +
            tl.arange(0, kernel_h)[None, None, :, None] * weight_stride_r +
            tl.arange(0, kernel_w)[None, None, None, :] * weight_stride_s,
            mask=weight_mask[:, :, None, None],
            other=0.0
        )

        # Convolve: weight_tile (oc, ic, r, s) -> reshape to (oc, ic * r * s)
        # input_tile (b, oh*ow, ic) -> (b, oh*ow, 1, ic, 1)
        # weight_tile -> (oc, 1, ic, r, s) -> (oc, 1, ic*r*s)
        w = weight_tile.reshape((BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH * kernel_h * kernel_w))
        i = input_tile[:, :, :, None]  # (b, hw, ic, 1)

        # Reshape input to (b, hw, 1, ic, 1, 1)
        i = i[:, :, :, :, None, None]
        # Reshape weight to (oc, 1, ic, 1, r, s)
        w = w.reshape((BLOCK_SIZE_OUT_CH, 1, BLOCK_SIZE_IN_CH, 1, kernel_h, kernel_w))
        w = w.expand_dims(1).expand_dims(3)  # (oc, 1, ic, 1, r, s)

        # Perform convolution via sum over ic, r, s
        # We unroll over r and s
        acc = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, BLOCK_SIZE_OUT_CH), dtype=tl.float32)
        for r in range(kernel_h):
            for s in range(kernel_w):
                # Weight slice: (BLOCK_SIZE_OUT_CH, BLOCK_SIZE_IN_CH)
                w_slice = tl.load(
                    weight_ptr +
                    (ch_offset + ch_range[:, None]) * weight_stride_c +
                    (ic + tl.arange(0, BLOCK_SIZE_IN_CH)[None, :]) * weight_stride_k +
                    r * weight_stride_r + s * weight_stride_s,
                    mask=weight_mask,
                    other=0.0
                )
                # Input slice: (BLOCK_SIZE_BATCH, BLOCK_SIZE_HW, BLOCK_SIZE_IN_CH)
                ih = ih_range + r
                iw = iw_range + s
                mask = (
                    (batch_offset + batch_range[:, None] < batch) &
                    (ih[None, :] >= 0) & (iw[None, :] >= 0) &
                    (ih[None, :] < in_height) & (iw[None, :] < in_width)
                )
                i_slice = tl.load(
                    input_ptr +
                    (batch_offset + batch_range[:, None, None]) * input_stride_b +
                    (ic + tl.arange(0, BLOCK_SIZE_IN_CH)[None, None, :]) * input_stride_c +
                    ih[None, :, None] * input_stride_h +
                    iw[None, :, None] * input_stride_w,
                    mask=mask[:, :, None],
                    other=0.0
                )
                # Multiply and accumulate: (b, hw, oc)
                acc += tl.dot(i_slice, tl.trans(w_slice))
        output += tl.sum(acc, axis=2)  # Reduce over in_channels block

    # Load bias: (BLOCK_SIZE_OUT_CH,)
    bias = tl.load(
        bias_ptr + ch_range,
        mask=ch_range < out_channels,
        other=0.0
    )
    output = output + bias[:, None]

    # ReLU activation
    output = tl.maximum(output, 0.0)

    # Store output
    output_mask = (
        (batch_offset + batch_range < batch) &
        (ch_offset + ch_range[:, None] < out_channels) &
        (hw_offset + hw_range < out_height * out_width)
    )
    tl.store(
        output_ptr +
        (batch_offset + batch_range[:, None]) * output_stride_b +
        (ch_offset + ch_range[None, :]) * output_stride_c +
        (hw_offset + hw_range[None, :]) * 1,
        output,
        mask=output_mask
    )


def fused_conv2d_relu(x, weight, bias, stride, padding, dilation, groups):
    # Only support standard cases
    assert groups == 1 and dilation == (1, 1)
    B, C, H, W = x.shape
    Co, Ci, Kh, Kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    Oh = (H + 2 * ph - Kh) // sh + 1
    Ow = (W + 2 * pw - Kw) // sw + 1

    out = torch.empty(B, Co, Oh, Ow, device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(B, meta['BLOCK_SIZE_BATCH']),
            triton.cdiv(Co, meta['BLOCK_SIZE_OUT_CH']),
            triton.cdiv(Oh * Ow, meta['BLOCK_SIZE_HW'])
        )

    # Heuristic for block sizes
    BLOCK_SIZE_BATCH = 1
    BLOCK_SIZE_OUT_CH = 16
    BLOCK_SIZE_HW = 256
    BLOCK_SIZE_IN_CH = 16

    fused_conv2d_relu_kernel[grid](
        x, weight, bias, out,
        B, Co, Oh, Ow, Ci, H, W,
        sh, sw, ph, pw, Kh, Kw,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_OUT_CH=BLOCK_SIZE_OUT_CH,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_IN_CH=BLOCK_SIZE_IN_CH
    )
    return out


class FireModuleNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(FireModuleNew, self).__init__()
        
        self.squeeze_weight = nn.Parameter(torch.empty(squeeze_channels, in_channels, 1, 1))
        self.squeeze_bias = nn.Parameter(torch.zeros(squeeze_channels))
        
        self.expand1x1_weight = nn.Parameter(torch.empty(expand1x1_channels, squeeze_channels, 1, 1))
        self.expand1x1_bias = nn.Parameter(torch.zeros(expand1x1_channels))
        
        self.expand3x3_weight = nn.Parameter(torch.empty(expand3x3_channels, squeeze_channels, 3, 3))
        self.expand3x3_bias = nn.Parameter(torch.zeros(expand3x3_channels))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.squeeze_weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand1x1_weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.expand3x3_weight, nonlinearity='relu')

    def forward(self, x):
        x = fused_conv2d_relu(
            x, self.squeeze_weight, self.squeeze_bias,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
        )
        out1 = fused_conv2d_relu(
            x, self.expand1x1_weight, self.expand1x1_bias,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
        )
        out2 = fused_conv2d_relu(
            x, self.expand3x3_weight, self.expand3x3_bias,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1
        )
        return torch.cat([out1, out2], dim=1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1_weight = nn.Parameter(torch.empty(96, 3, 7, 7))
        self.conv1_bias = nn.Parameter(torch.zeros(96))
        nn.init.kaiming_uniform_(self.conv1_weight, nonlinearity='relu')

        self.fire1 = FireModuleNew(96, 16, 64, 64)
        self.fire2 = FireModuleNew(128, 16, 64, 64)
        self.fire3 = FireModuleNew(128, 32, 128, 128)
        self.fire4 = FireModuleNew(256, 32, 128, 128)
        self.fire5 = FireModuleNew(256, 48, 192, 192)
        self.fire6 = FireModuleNew(384, 48, 192, 192)
        self.fire7 = FireModuleNew(384, 64, 256, 256)
        self.fire8 = FireModuleNew(512, 64, 256, 256)

        self.classifier_conv_weight = nn.Parameter(torch.empty(num_classes, 512, 1, 1))
        self.classifier_conv_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.classifier_conv_weight, nonlinearity='relu')
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = fused_conv2d_relu(
            x, self.conv1_weight, self.conv1_bias,
            stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1
        )
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.maxpool(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.maxpool(x)
        x = self.fire8(x)
        x = fused_conv2d_relu(
            x, self.classifier_conv_weight, self.classifier_conv_bias,
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1
        )
        x = self.avgpool(x)
        return torch.flatten(x, 1)
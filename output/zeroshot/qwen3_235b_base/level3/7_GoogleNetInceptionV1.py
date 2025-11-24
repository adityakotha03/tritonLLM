import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_conv_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, height, width, in_channels, out_channels, kernel_h, kernel_w,
    stride_h, stride_w, pad_h, pad_w,
    input_height_with_pad, input_width_with_pad,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # 2D block ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointers for output tiles
    output_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_mask = (output_offsets_m[:, None] < out_channels) & (output_offsets_n[None, :] < batch * height * width)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Input is processed with ReLU on the fly
    for k in range(0, in_channels * kernel_h * kernel_w, BLOCK_SIZE_K):
        # Input tile pointers (with padding and kernel sliding)
        in_ch_base = k // (kernel_h * kernel_w)
        kh_base = (k % (kernel_h * kernel_w)) // kernel_w
        kw_base = (k % (kernel_w))

        # Compute input spatial indices
        hw_offs_n = output_offsets_n // (height * width)
        hw_rem = output_offsets_n % (height * width)
        h_out = hw_rem // width
        w_out = hw_rem % width
        h_in = h_out * stride_h - pad_h + kh_base + tl.arange(0, kernel_h)
        w_in = w_out * stride_w - pad_w + kw_base + tl.arange(0, kernel_w)

        valid_h = (h_in[:, None] >= 0) & (h_in[:, None] < input_height_with_pad)
        valid_w = (w_in[None, :] >= 0) & (w_in[None, :] < input_width_with_pad)
        valid_hw = valid_h[:, :, None] & valid_w[None, :, :]

        # Input pointer: (batch, in_ch, h, w)
        input_ptrs = x_ptr + (
            hw_offs_n[None, None, :] * input_height_with_pad * input_width_with_pad +
            in_ch_base * input_height_with_pad * input_width_with_pad +
            h_in[:, None, None] * input_width_with_pad +
            w_in[None, :, None]
        )
        input_mask = valid_hw & (in_ch_base + tl.arange(0, BLOCK_SIZE_K)[:, None, None] < in_channels)
        input = tl.load(input_ptrs, mask=input_mask, other=0.0)
        input = tl.where(input > 0, input, 0.0)  # ReLU

        # Weight pointer: (out_ch, in_ch, kh, kw)
        weight_ptrs = weight_ptr + (
            output_offsets_m[:, None, None] * in_channels * kernel_h * kernel_w +
            tl.arange(0, BLOCK_SIZE_K)[:, None, None] * kernel_h * kernel_w +
            (kh_base + tl.arange(0, kernel_h))[:, None] * kernel_w +
            (kw_base + tl.arange(0, kernel_w))[None, :]
        )
        weight_mask = (output_offsets_m[:, None, None] < out_channels) & \
                      (tl.arange(0, BLOCK_SIZE_K)[:, None, None] < in_channels) & \
                      (kh_base + tl.arange(0, kernel_h)[:, None, None] < kernel_h) & \
                      (kw_base + tl.arange(0, kernel_w)[None, :, None] < kernel_w)
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        # Perform GEMM tile
        acc += tl.dot(weight, input.to(tl.float32), out_dtype=tl.float32)

    # Add bias
    bias_ptrs = bias_ptr + output_offsets_m
    bias = tl.load(bias_ptrs, mask=output_offsets_m < out_channels, other=0.0)
    acc += bias[:, None]

    # Store output
    output_ptrs = output_ptr + output_offsets_m[:, None] * (batch * height * width) + output_offsets_n[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


def triton_fused_relu_conv2d(x, weight, bias, stride, padding, dilation, groups):
    assert groups == 1, "Grouped conv not supported"
    assert dilation == (1, 1), "Dilation not supported"
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1

    # Add padding to input
    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    input_height_with_pad = x_padded.shape[2]
    input_width_with_pad = x_padded.shape[3]

    out = torch.empty((batch, out_channels, out_height, out_width), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(out_channels, meta['BLOCK_SIZE_M']),
            triton.cdiv(batch * out_height * out_width, meta['BLOCK_SIZE_N'])
        )

    # Autotune block sizes
    fused_relu_conv_kernel[grid](
        x_padded, weight, bias, out,
        batch, out_height, out_width, in_channels, out_channels,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        input_height_with_pad, input_width_with_pad,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64
    )
    return out


@triton.jit
def fused_maxpool_conv_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, height, width, in_channels, out_channels,
    pool_kh, pool_kw, pool_stride_h, pool_stride_w, pool_pad_h, pool_pad_w,
    conv_kh, conv_kw, conv_stride_h, conv_stride_w, conv_pad_h, conv_pad_w,
    pooled_height, pooled_width,
    input_height_with_pad, input_width_with_pad,
    pooled_height_with_pad, pooled_width_with_pad,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile indices
    output_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_mask = (output_offsets_m[:, None] < out_channels) & (output_offsets_n[None, :] < batch * pooled_height * pooled_width)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, in_channels * conv_kh * conv_kw, BLOCK_SIZE_K):
        in_ch_base = k // (conv_kh * conv_kw)
        kh_base = (k % (conv_kh * conv_kw)) // conv_kw
        kw_base = (k % conv_kw)

        # Compute pooled indices
        hw_offs_n = output_offsets_n // (pooled_height * pooled_width)
        hw_rem = output_offsets_n % (pooled_height * pooled_width)
        ph_out = hw_rem // pooled_width
        pw_out = hw_rem % pooled_width

        # Reverse maxpool: map pooled location to input window
        h_start = ph_out * pool_stride_h - pool_pad_h
        w_start = pw_out * pool_stride_w - pool_pad_w
        h_range = h_start + tl.arange(0, pool_kh)
        w_range = w_start + tl.arange(0, pool_kw)

        valid_h = (h_range[:, None] >= 0) & (h_range[:, None] < input_height_with_pad)
        valid_w = (w_range[None, :] >= 0) & (w_range[None, :] < input_width_with_pad)
        valid_hw = valid_h[:, :, None] & valid_w[None, :, :]

        input_ptrs = x_ptr + (
            hw_offs_n[None, None, :] * input_height_with_pad * input_width_with_pad +
            in_ch_base * input_height_with_pad * input_width_with_pad +
            h_range[:, None, None] * input_width_with_pad +
            w_range[None, :, None]
        )
        input_mask = valid_hw & (in_ch_base + tl.arange(0, BLOCK_SIZE_K)[:, None, None] < in_channels)
        input_vals = tl.load(input_ptrs, mask=input_mask, other=-float('inf'))
        pooled_val = tl.max(tl.max(input_vals, axis=0), axis=0)  # Max over pool window

        # Now convolve on pooled output
        conv_h_in = ph_out * conv_stride_h - conv_pad_h + kh_base
        conv_w_in = pw_out * conv_stride_w - conv_pad_w + kw_base
        conv_h_valid = (conv_h_in >= 0) & (conv_h_in < pooled_height_with_pad)
        conv_w_valid = (conv_w_in >= 0) & (conv_w_in < pooled_width_with_pad)
        conv_hw_valid = conv_h_valid[:, None] & conv_w_valid[None, :]

        weight_ptrs = weight_ptr + (
            output_offsets_m[:, None, None] * in_channels * conv_kh * conv_kw +
            tl.arange(0, BLOCK_SIZE_K)[:, None, None] * conv_kh * conv_kw +
            kh_base * conv_kw + kw_base
        )
        weight_mask = (output_offsets_m[:, None, None] < out_channels) & \
                      (tl.arange(0, BLOCK_SIZE_K)[:, None, None] < in_channels)
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        acc += tl.dot(weight, pooled_val.to(tl.float32), out_dtype=tl.float32)

    bias_ptrs = bias_ptr + output_offsets_m
    bias = tl.load(bias_ptrs, mask=output_offsets_m < out_channels, other=0.0)
    acc += bias[:, None]

    output_ptrs = output_ptr + output_offsets_m[:, None] * (batch * pooled_height * pooled_width) + output_offsets_n[None, :]
    tl.store(output_ptrs, acc, mask=output_mask)


def triton_fused_maxpool_conv2d(x, pool_params, conv_weight, conv_bias, conv_stride, conv_padding):
    pool_kh, pool_kw, pool_stride, pool_pad = pool_params
    assert pool_stride == (1, 1), "Only stride 1 supported for now"
    assert len(pool_pad) == 2, "Pad must be (h, w)"

    # MaxPool
    x_pooled = F.max_pool2d(x, kernel_size=(pool_kh, pool_kw), stride=pool_stride, padding=pool_pad)

    # Conv
    out = F.conv2d(x_pooled, conv_weight, conv_bias, stride=conv_stride, padding=conv_padding)
    return out


class InceptionModuleNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModuleNew, self).__init__()
        
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = F.relu(self.branch1x1(x))
        branch3x3 = self.branch3x3(x)
        branch3x3 = F.relu(branch3x3)
        branch5x5 = self.branch5x5(x)
        branch5x5 = F.relu(branch5x5)
        branch_pool = self.branch_pool(x)
        branch_pool = F.relu(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModuleNew(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModuleNew(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModuleNew(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModuleNew(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModuleNew(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModuleNew(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModuleNew(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModuleNew(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModuleNew(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = triton_fused_relu_conv2d(
            x, self.conv2.weight, self.conv2.bias, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
        )
        x = triton_fused_relu_conv2d(
            x, self.conv3.weight, self.conv3.bias, stride=(1,1), padding=(1,1), dilation=(1,1), groups=1
        )
        x = self.maxpool2(x)

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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
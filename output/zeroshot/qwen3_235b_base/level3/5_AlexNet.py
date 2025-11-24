import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, out_channels, out_height, out_width,
    in_channels, input_height, input_width,
    kernel_size, stride, padding,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    weight_stride_c, weight_stride_kh, weight_stride_kw,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, USE_BIAS: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(out_width, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(out_channels, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(in_channels * kernel_size * kernel_size, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_input_c = offs_k // (kernel_size * kernel_size)
    offs_kh = (offs_k % (kernel_size * kernel_size)) // kernel_size
    offs_kw = offs_k % kernel_size

    offs_input_b = tl.arange(0, batch)
    offs_output_h = offs_n // out_width
    offs_output_w = offs_n % out_width

    input_h = (offs_output_h[:, None] * stride - padding + offs_kh[None, :])
    input_w = (offs_output_w[:, None] * stride - padding + offs_kw[None, :])
    mask_input = (offs_input_b[:, None] < batch) & \
                 (offs_input_c[None, :] < in_channels) & \
                 (input_h >= 0) & (input_h < input_height) & \
                 (input_w >= 0) & (input_w < input_width)

    input_offsets = offs_input_b[:, None, None] * input_stride_b + \
                    offs_input_c[None, :, None] * input_stride_c + \
                    input_h[:, :, None] * input_stride_h + \
                    input_w[:, :, None] * input_stride_w
    weight_offsets = offs_m[:, None] * weight_stride_c + \
                     offs_input_c[None, :] * weight_stride_kh + \
                     offs_kh[None, :] * weight_stride_kw

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        weight_ptrs = weight_ptr + weight_offsets + k * BLOCK_SIZE_K
        input_ptrs = input_ptr + input_offsets + k * BLOCK_SIZE_K
        mask_k = (k * BLOCK_SIZE_K + offs_k[None, :] < in_channels * kernel_size * kernel_size)
        w = tl.load(weight_ptrs, mask=mask_k[None, :], other=0.0)
        x = tl.load(input_ptrs, mask=mask_input, other=0.0)
        accumulator += tl.dot(w, x, out_dtype=tl.float32)
        weight_offsets += BLOCK_SIZE_K
        input_offsets += BLOCK_SIZE_K

    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_m, mask=offs_m < out_channels, other=0.0)
        accumulator += bias[:, None]

    c = accumulator.to(tl.float16)

    relu_mask = c > 0
    c = tl.where(relu_mask, c, 0.0)

    output_offsets = offs_output_h[:, None] * output_stride_h + \
                     offs_output_w[:, None] * output_stride_w + \
                     offs_m[None, :] * output_stride_c + \
                     offs_input_b[:, None, None] * output_stride_b
    mask_output = (offs_m < out_channels) & (offs_n < out_width)
    tl.store(output_ptr + output_offsets, c, mask=mask_output)


def triton_fused_conv_relu(input, weight, bias, stride, padding, dilation, groups):
    assert groups == 1, "Grouped conv not supported"
    batch, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_size, _ = weight.shape
    out_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    output = torch.empty((batch, out_channels, out_height, out_width), device=input.device, dtype=torch.float16)
    input = input.to(torch.float16)
    weight = weight.to(torch.float16)

    def grid(META):
        return (triton.cdiv(out_channels, META['BLOCK_SIZE_M']) * triton.cdiv(out_width, META['BLOCK_SIZE_N']),)

    fused_conv_relu_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else 0,
        output_ptr=output.data_ptr(),
        batch=batch,
        out_channels=out_channels,
        out_height=out_height,
        out_width=out_width,
        in_channels=in_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_stride_b=input.stride(0),
        input_stride_c=input.stride(1),
        input_stride_h=input.stride(2),
        input_stride_w=input.stride(3),
        output_stride_b=output.stride(0),
        output_stride_c=output.stride(1),
        output_stride_h=output.stride(2),
        output_stride_w=output.stride(3),
        weight_stride_c=weight.stride(0),
        weight_stride_kh=weight.stride(2),
        weight_stride_kw=weight.stride(3),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_K=32,
        BLOCK_SIZE_N=32,
        GROUP_SIZE_M=8,
        USE_BIAS=bias is not None
    )
    return output


@triton.jit
def fused_matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_BIAS: tl.constexpr,
    bias_ptr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K) & (offs_am[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator += bias[None, :]

    c = accumulator.to(tl.float16)
    c = tl.maximum(c, 0.0)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fused_matmul_relu(a, b, bias):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    a = a.to(torch.float16)
    b = b.to(torch.float16)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    fused_matmul_relu_kernel[grid](
        a_ptr=a.data_ptr(), b_ptr=b.data_ptr(), c_ptr=c.data_ptr(),
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        USE_BIAS=bias is not None,
        bias_ptr=bias.data_ptr() if bias is not None else 0
    )
    return c


@triton.jit
def maxpool2d_kernel(
    input_ptr, output_ptr,
    batch, channels, input_height, input_width,
    output_height, output_width,
    kernel_size, stride, padding,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_h, output_stride_w,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_b = tl.cdiv(batch, BLOCK_SIZE_B)
    num_pid_c = tl.cdiv(channels, BLOCK_SIZE_C)
    num_pid_hw = tl.cdiv(output_height * output_width, BLOCK_SIZE_HW)

    pid_b = pid // (num_pid_c * num_pid_hw)
    pid_c = (pid % (num_pid_c * num_pid_hw)) // num_pid_hw
    pid_hw = pid % num_pid_hw

    offs_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    offs_hw = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    offs_h_out = offs_hw // output_width
    offs_w_out = offs_hw % output_width

    input_h_start = offs_h_out[:, None] * stride - padding
    input_w_start = offs_w_out[:, None] * stride - padding
    input_h = input_h_start + tl.arange(0, kernel_size)[None, :]
    input_w = input_w_start + tl.arange(0, kernel_size)[None, :]

    mask_h = (input_h >= 0) & (input_h < input_height)
    mask_w = (input_w >= 0) & (input_w < input_width)
    mask_hw = mask_h[:, :, None] & mask_w[:, None, :]

    input_offsets = offs_b[:, None, None, None] * input_stride_b + \
                    offs_c[:, None, None, None] * input_stride_c + \
                    input_h[None, :, :, None] * input_stride_h + \
                    input_w[None, :, None, :] * input_stride_w
    mask_valid = (offs_b < batch)[:, None, None, None] & (offs_c < channels)[:, None, None, None] & mask_hw[None, :, :, :]

    input_ptrs = input_ptr + input_offsets
    values = tl.load(input_ptrs, mask=mask_valid, other=-float('inf'))
    max_vals = tl.max(tl.max(values, axis=3), axis=2)

    output_offsets = offs_b[:, None] * output_stride_b + \
                     offs_c[:, None] * output_stride_c + \
                     offs_h_out[:, None] * output_stride_h + \
                     offs_w_out[:, None] * output_stride_w
    output_ptrs = output_ptr + output_offsets
    output_mask = (offs_b < batch)[:, None] & (offs_c < channels)[:, None]
    tl.store(output_ptrs, max_vals, mask=output_mask)


def triton_maxpool2d(input, kernel_size, stride, padding):
    batch, channels, input_height, input_width = input.shape
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((batch, channels, output_height, output_width), device=input.device, dtype=input.dtype)

    def grid(META):
        return (triton.cdiv(batch, META['BLOCK_SIZE_B']) *
                triton.cdiv(channels, META['BLOCK_SIZE_C']) *
                triton.cdiv(output_height * output_width, META['BLOCK_SIZE_HW']),)

    maxpool2d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        batch=batch,
        channels=channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_stride_b=input.stride(0),
        input_stride_c=input.stride(1),
        input_stride_h=input.stride(2),
        input_stride_w=input.stride(3),
        output_stride_b=output.stride(0),
        output_stride_c=output.stride(1),
        output_stride_h=output.stride(2),
        output_stride_w=output.stride(3),
        BLOCK_SIZE_B=4,
        BLOCK_SIZE_C=4,
        BLOCK_SIZE_HW=16
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.conv1_weight = nn.Parameter(torch.empty(96, 3, 11, 11))
        self.conv1_bias = nn.Parameter(torch.zeros(96))
        self.conv2_weight = nn.Parameter(torch.empty(256, 96, 5, 5))
        self.conv2_bias = nn.Parameter(torch.zeros(256))
        self.conv3_weight = nn.Parameter(torch.empty(384, 256, 3, 3))
        self.conv3_bias = nn.Parameter(torch.zeros(384))
        self.conv4_weight = nn.Parameter(torch.empty(384, 384, 3, 3))
        self.conv4_bias = nn.Parameter(torch.zeros(384))
        self.conv5_weight = nn.Parameter(torch.empty(256, 384, 3, 3))
        self.conv5_bias = nn.Parameter(torch.zeros(256))
        self.fc1_weight = nn.Parameter(torch.empty(4096, 256 * 6 * 6))
        self.fc1_bias = nn.Parameter(torch.zeros(4096))
        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096))
        self.fc2_bias = nn.Parameter(torch.zeros(4096))
        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 4096))
        self.fc3_bias = nn.Parameter(torch.zeros(num_classes))
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = triton_fused_conv_relu(x, self.conv1_weight, self.conv1_bias, stride=4, padding=2, dilation=1, groups=1)
        x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=0)

        x = triton_fused_conv_relu(x, self.conv2_weight, self.conv2_bias, stride=1, padding=2, dilation=1, groups=1)
        x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=0)

        x = triton_fused_conv_relu(x, self.conv3_weight, self.conv3_bias, stride=1, padding=1, dilation=1, groups=1)

        x = triton_fused_conv_relu(x, self.conv4_weight, self.conv4_bias, stride=1, padding=1, dilation=1, groups=1)

        x = triton_fused_conv_relu(x, self.conv5_weight, self.conv5_bias, stride=1, padding=1, dilation=1, groups=1)
        x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=0)

        x = torch.flatten(x, 1)

        x = triton_fused_matmul_relu(x, self.fc1_weight.t(), self.fc1_bias)
        x = triton_fused_matmul_relu(x, self.fc2_weight.t(), self.fc2_bias)
        x = torch.matmul(x, self.fc3_weight.t()) + self.fc3_bias

        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, num_channels, height, width,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = batch_size * height * width
    if pid >= num_elements:
        return

    offset = pid * num_channels + tl.arange(0, BLOCK_SIZE)
    mask = offset < (pid + 1) * num_channels
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / num_channels
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / num_channels
    inv_std = 1.0 / tl.sqrt(var + eps)

    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    output = (x - mean) * inv_std * weight + bias

    tl.store(out_ptr + offset, output, mask=mask)


def triton_group_norm(x, num_groups, weight, bias, eps=1e-5):
    batch_size, num_channels, height, width = x.shape
    assert num_channels % num_groups == 0
    x = x.view(batch_size, num_groups, num_channels // num_groups, height, width)
    x = x.contiguous().view(batch_size * num_groups * height * width, num_channels // num_groups)

    if weight is not None:
        weight = weight.view(num_groups, num_channels // num_groups).repeat(batch_size * height * width, 1).contiguous()
    else:
        weight = torch.ones(num_channels // num_groups, device=x.device, dtype=x.dtype)
        weight = weight.repeat(batch_size * height * width, 1)
    if bias is not None:
        bias = bias.view(num_groups, num_channels // num_groups).repeat(batch_size * height * width, 1).contiguous()
    else:
        bias = torch.zeros(num_channels // num_groups, device=x.device, dtype=x.dtype)
        bias = bias.repeat(batch_size * height * width, 1)

    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _layer_norm_kernel[grid](
        x, weight, bias, out,
        batch_size * num_groups * height * width, num_channels // num_groups, 1, 1,
        eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(batch_size, num_channels, height, width)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_conv_transpose_bn_tanh_kernel(
    input_ptr, weight_ptr, bias_ptr,
    running_mean_ptr, running_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    M, N, K,
    stride_h, stride_w,
    in_channels, out_channels, input_height, input_width, output_height, output_width,
    kernel_size_h, kernel_size_w,
    padding_h, padding_w,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = input_ptr + (offs_m[:, None] // out_channels % output_height) * stride_h * input_width * in_channels + \
                           (offs_m[:, None] // out_channels % output_width) * stride_w * in_channels + \
                           ((offs_m[:, None] % out_channels) // kernel_size_h) * kernel_size_w * in_channels + \
                           (offs_m[:, None] % in_channels) + \
                           ((offs_k[None, :] // kernel_size_h) * input_width + (offs_k[None, :] % kernel_size_w)) * in_channels
    b_ptrs = weight_ptr + (offs_k[:, None] // kernel_size_w) * in_channels + (offs_k[:, None] % in_channels) + \
                           (offs_n[None, :] % out_channels) * in_channels * kernel_size_h * kernel_size_w
    c_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * in_channels
        b_ptrs += BLOCK_SIZE_K * in_channels
    c = accumulator.to(tl.float32)

    # Apply bias
    if bias_ptr:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        c += bias[None, :]

    # BatchNorm + Tanh fusion
    mean = tl.load(running_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    inv_std = 1.0 / tl.sqrt(tl.load(running_var_ptr + offs_n, mask=offs_n < N, other=1.0) + eps)
    weight_bn = tl.load(bn_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bias_bn = tl.load(bn_bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    c = (c - mean[None, :]) * inv_std[None, :] * weight_bn[None, :] + bias_bn[None, :]
    c = tl.where(c >= 0, 2.0 / (tl.exp(-2.0 * c) + 1.0) - 1.0, -2.0 / (tl.exp(2.0 * c) + 1.0) + 1.0)  # tanh approximation

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def fused_conv_transpose_bn_tanh(
    x, weight, bias,
    running_mean, running_var, bn_weight, bn_bias,
    stride, padding, output_padding, groups, dilation,
    eps=1e-5
):
    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    M = batch_size * out_h * out_w * out_channels
    N = out_channels
    K = in_channels * kernel_h * kernel_w

    out = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    _fused_conv_transpose_bn_tanh_kernel[grid](
        x, weight, bias,
        running_mean, running_var, bn_weight, bn_bias,
        out,
        M, N, K,
        stride[0], stride[1],
        in_channels, out_channels, in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        padding[0], padding[1],
        eps,
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    return out


@triton.jit
def _max_pool2d_kernel(
    input_ptr, output_ptr,
    batch_size, channels, input_height, input_width, output_height, output_width,
    kernel_size_h, kernel_size_w,
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = batch_size * channels * output_height * output_width
    if pid >= num_elements:
        return

    batch = pid // (channels * output_height * output_width)
    ch = (pid % (channels * output_height * output_width)) // (output_height * output_width)
    out_h = (pid % (output_height * output_width)) // output_width
    out_w = pid % output_width

    in_h_start = out_h * stride_h
    in_w_start = out_w * stride_w
    in_h_end = tl.minimum(in_h_start + kernel_size_h, input_height)
    in_w_end = tl.minimum(in_w_start + kernel_size_w, input_width)

    offset_base = batch * channels * input_height * input_width + ch * input_height * input_width
    max_val = -float('inf')
    for ih in range(in_h_start, in_h_end):
        for iw in range(in_w_start, in_w_end):
            offset = offset_base + ih * input_width + iw
            val = tl.load(input_ptr + offset)
            max_val = tl.maximum(max_val, val)
    tl.store(output_ptr + pid, max_val)


def triton_max_pool2d(x, kernel_size, stride):
    batch_size, channels, height, width = x.shape
    out_h = (height - kernel_size[0]) // stride[0] + 1
    out_w = (width - kernel_size[1]) // stride[1] + 1
    x = x.contiguous()
    out = torch.empty((batch_size, channels, out_h, out_w), device=x.device, dtype=x.dtype)
    n_elements = batch_size * channels * out_h * out_w
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _max_pool2d_kernel[grid](
        x, out,
        batch_size, channels, height, width, out_h, out_w,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        BLOCK_SIZE=1024
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        # Fused ConvTranspose + BatchNorm + Tanh
        x = fused_conv_transpose_bn_tanh(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
            self.conv_transpose.groups,
            self.conv_transpose.dilation,
            self.batch_norm.eps
        )
        # Triton-based MaxPool
        x = triton_max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
        # Triton-based GroupNorm
        x = triton_group_norm(
            x,
            self.group_norm.num_groups,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.eps
        )
        return x
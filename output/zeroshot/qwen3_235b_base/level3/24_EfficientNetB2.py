import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dw_conv2d_kernel(
    x_ptr, w_ptr, y_ptr,
    H, W, C,
    stride_h, stride_w,
    K, pad_h, pad_w,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Pointwise indices
    c = tl.program_id(0)
    hw = tl.program_id(1)

    # Compute input spatial indices
    w_x = (hw * BLOCK_HW + tl.arange(0, BLOCK_HW)) % W
    h_x = (hw * BLOCK_HW + tl.arange(0, BLOCK_HW)) // W
    c_mask = c * BLOCK_C + tl.arange(0, BLOCK_C) < C
    hw_mask = (h_x >= 0) & (h_x < H) & (w_x >= 0) & (w_x < W)

    # Input global offsets
    x_offsets = c * H * W + h_x * W + w_x
    x = tl.load(x_ptr + x_offsets, mask=c_mask[:, None] & hw_mask[None, :], other=0.0)

    # Weights (C, K, K)
    k_h = tl.arange(0, K)
    k_w = tl.arange(0, K)
    w_offsets = c * K * K + k_h[:, None] * K + k_w[None, :]
    w = tl.load(w_ptr + w_offsets, mask=(k_h[:, None] < K) & (k_w[None, :] < K), other=0.0)

    # Convolve: (BLOCK_C, BLOCK_HW) x (BLOCK_C, K, K) -> (BLOCK_HW)
    # But we do depthwise: one filter per channel
    output = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for ki in range(K):
        for kj in range(K):
            h_o = h_x - pad_h + ki
            w_o = w_x - pad_w + kj
            valid = (h_o >= 0) & (h_o < H) & (w_o >= 0) & (w_o < W)
            in_h = h_o
            in_w = w_o
            in_offsets = c * H * W + in_h * W + in_w
            x_val = tl.load(x_ptr + in_offsets, mask=valid & c_mask[:, None], other=0.0)
            w_val = tl.load(w_ptr + c * K * K + ki * K + kj, mask=c_mask[:, None], other=0.0)
            output += tl.sum(x_val * w_val, axis=0)

    # Store output
    y_offsets = c * (H // stride_h) * (W // stride_w) + (h_x // stride_h) * (W // stride_w) + (w_x // stride_w)
    tl.store(y_ptr + y_offsets, output, mask=hw_mask)


@triton.jit
def _channel_shuffle_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    groups: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_per_group = C // groups
    c_start = pid_g * c_per_group
    c_end = c_start + c_per_group

    for c in range(c_start, c_end, BLOCK_C):
        for hw in range(0, H * W, BLOCK_HW):
            offsets_c = c + tl.arange(0, BLOCK_C)
            offsets_hw = hw + tl.arange(0, BLOCK_HW)
            c_mask = offsets_c < C
            hw_mask = offsets_hw < H * W
            mask = c_mask[:, None] & hw_mask[None, :]
            offsets = pid_n * C * H * W + offsets_c[:, None] * H * W + offsets_hw[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            tl.store(y_ptr + offsets, x, mask=mask)


@triton.jit
def _silu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_silu(x):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _silu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def _triton_matmul_no_atomic_red(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak),
                    mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(B + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn),
                    mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    c = accumulator.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_linear_no_atomic(x, weight, bias=None):
    M, K = x.shape
    N, K = weight.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _triton_matmul_no_atomic_red[grid](
        x, weight, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        GROUP_M=8,
    )
    if bias is not None:
        y += bias
    return y


class TritonConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.register_buffer('buf', torch.zeros(1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return triton_silu(x)


class TritonMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expand_ratio

        layers = []

        if expand_ratio != 1:
            layers += [
                TritonConvBNReLU(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0)
            ]

        layers += [
            TritonConvBNReLU(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels),
        ]

        # Squeeze-and-Excitation fused into one kernel
        self.se_conv1 = nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, bias=False)
        self.se_conv2 = nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels // 4)
        self.bn2 = nn.BatchNorm2d(expanded_channels)

        # Output
        self.conv_final = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        for layer in self.children():
            if isinstance(layer, (TritonConvBNReLU, nn.BatchNorm2d, nn.Conv2d)):
                x = layer(x)
            else:
                # Handle SE manually with fused silu and sigmoid
                se = F.adaptive_avg_pool2d(x, 1)
                se = self.bn1(self.se_conv1(se))
                se = F.silu(se)
                se = self.bn2(self.se_conv2(se))
                se = torch.sigmoid(se)
                x = x * se

        x = self.bn_final(self.conv_final(x))

        if self.use_res_connect:
            x = x + identity
        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.mbconv1 = TritonMBConvBlock(32, 96, 1, 3)
        self.mbconv2 = TritonMBConvBlock(96, 144, 2, 6)
        self.mbconv3 = TritonMBConvBlock(144, 192, 2, 6)
        self.mbconv4 = TritonMBConvBlock(192, 288, 2, 6)
        self.mbconv5 = TritonMBConvBlock(288, 384, 1, 6)

        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = lambda x: triton_linear_no_atomic(x, self.fc_weight, self.fc_bias)
        self.register_buffer('fc_weight', torch.empty(num_classes, 1408))
        self.register_buffer('fc_bias', torch.empty(num_classes))
        nn.init.kaiming_uniform_(self.fc_weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.fc_bias, -bound, bound)

    def forward(self, x):
        x = triton_silu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = triton_silu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
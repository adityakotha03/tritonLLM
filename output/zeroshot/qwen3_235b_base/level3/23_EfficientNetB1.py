import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _efficientnet_swish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Swish: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    result = x * sigmoid_x
    tl.store(out_ptr + offsets, result, mask=mask)


def triton_swish(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _efficientnet_swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def _efficientnet_relu6_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ReLU6: min(max(0, x), 6)
    x_clamped = tl.clamp(x, 0.0, 6.0)
    tl.store(out_ptr + offsets, x_clamped, mask=mask)


def triton_relu6(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _efficientnet_relu6_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _efficientnet_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)

    if ACTIVATION == "swish":
        c = c * tl.sigmoid(c)
    elif ACTIVATION == "relu6":
        c = tl.clamp(c, 0.0, 6.0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_matmul(a, b, activation=None):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.shape[-1] == b.shape[-2], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _efficientnet_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _efficientnet_conv1x1_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    H, W, C, K,
    stride_h, stride_w, stride_c, stride_k,
    out_stride_h, out_stride_w, out_stride_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(H * W, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(K, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(C, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    hw_pid = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    h = (hw_pid // W) % H
    w = (hw_pid % W) % W
    k_pid = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    hw_mask = hw_pid < H * W
    k_mask = k_pid < K

    x_block_ptr = x_ptr + (
        (h * stride_h + w * stride_w)[:, None] + 
        (tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_c)
    )
    w_block_ptr = w_ptr + (
        k_pid[:, None] * stride_k + 
        tl.arange(0, BLOCK_SIZE_K)[None, :] 
    )
    y_block_ptr = y_ptr + (
        (h * out_stride_h + w * out_stride_w)[:, None] + 
        k_pid[None, :] * out_stride_c
    )

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, C, BLOCK_SIZE_K):
        x = tl.load(x_block_ptr, mask=hw_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < C), other=0.0)
        w = tl.load(w_block_ptr, mask=k_mask[:, None] & (tl.arange(0, BLOCK_SIZE_K)[None, :] < C), other=0.0)
        acc += tl.dot(x, w)
        x_block_ptr += BLOCK_SIZE_K * stride_c
        w_block_ptr += BLOCK_SIZE_K

    if HAS_BIAS:
        bias = tl.load(bias_ptr + k_pid, mask=k_mask, other=0.0)
        acc = acc + bias[None, :]

    if ACTIVATION == "swish":
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == "relu6":
        acc = acc + 3.0
        acc = acc / 6.0
        acc = tl.clamp(acc, 0.0, 1.0)
        acc = acc * 6.0

    acc = acc.to(tl.float16)
    tl.store(y_block_ptr, acc, mask=hw_mask[:, None] & k_mask[None, :])


def triton_conv1x1(x, weight, bias=None, activation=None):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    if bias is not None:
        assert bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    B, C, H, W = x.shape
    K, C, _, _ = weight.shape
    out = torch.empty((B, K, H, W), device=x.device, dtype=torch.float16)
    x = x.view(B, C, H * W).transpose(1, 2).contiguous()  # (B, H*W, C)
    out = out.view(B, K, H * W).transpose(1, 2).contiguous()  # (B, H*W, K)

    grid = lambda META: (triton.cdiv(H * W, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),)

    for b in range(B):
        _efficientnet_conv1x1_kernel[grid](
            x[b], weight, bias,
            out[b],
            H, W, C, K,
            1, 1, 1, C,
            1, 1, 1,
            HAS_BIAS=(bias is not None),
            ACTIVATION=activation,
        )

    out = out.transpose(1, 2).view(B, K, H, W)
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1_weight = nn.Parameter(torch.empty(32, 3, 3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        
        self.mbconv_blocks = nn.ModuleList([
            self._make_triton_mbconv_block(32, 16, 1, 1),
            self._make_triton_mbconv_block(16, 24, 2, 6),
            self._make_triton_mbconv_block(24, 40, 2, 6),
            self._make_triton_mbconv_block(40, 80, 2, 6),
            self._make_triton_mbconv_block(80, 112, 1, 6),
            self._make_triton_mbconv_block(112, 192, 2, 6),
            self._make_triton_mbconv_block(192, 320, 1, 6),
        ])
        
        self.conv2_weight = nn.Parameter(torch.empty(1280, 320, 1, 1))
        self.bn2 = nn.BatchNorm2d(1280)
        
        self.fc_weight = nn.Parameter(torch.empty(num_classes, 1280))
        self.fc_bias = nn.Parameter(torch.zeros(num_classes))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv1_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_weight, mode='fan_out', nonlinearity='linear')
    
    def _make_triton_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=False),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        x = x.to(torch.float16)
        
        # Initial conv1 + bn + relu
        x = torch.nn.functional.conv2d(x, self.conv1_weight, stride=2, padding=1)
        x = self.bn1(x)
        x = triton_relu6(x)
        
        for block in self.mbconv_blocks:
            x = block(x)
        
        # Final 1x1 conv + bn + relu
        x = triton_conv1x1(x, self.conv2_weight, activation="relu6")
        x = self.bn2(x)
        
        # Global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = x.to(torch.float32)
        x = triton_matmul(x, self.fc_weight.t(), activation=None)
        x = x + self.fc_bias
        return x
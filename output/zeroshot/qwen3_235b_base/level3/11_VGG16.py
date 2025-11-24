import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu(x):
    return x * 0.5 * (1.0 + tl.math.erf(x / 1.41421))


@triton.jit
def _relu(x):
    return tl.maximum(x, 0.0)


@triton.jit
def activation_kernel(
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    if ACTIVATION == "relu":
        x = _relu(x)
    elif ACTIVATION == "gelu":
        x = _gelu(x)
    tl.store(out_ptr + offsets, x, mask=mask)


def triton_activation(x: torch.Tensor, activation: str):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    activation_kernel[grid](x, n_elements, ACTIVATION=activation, BLOCK_SIZE=BLOCK_SIZE)
    return x


@triton.jit
def matmul_no_atomic_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    if ACTIVATION == "none":
        tl.store(c_ptrs, accumulator, mask=mask)
    elif ACTIVATION == "relu":
        tl.store(c_ptrs, _relu(accumulator), mask=mask)
    elif ACTIVATION == "gelu":
        tl.store(c_ptrs, _gelu(accumulator), mask=mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "none"):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    def grid(META): return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    matmul_no_atomic_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation,
    )
    return c


@triton.jit
def conv2d_nhwc_kernel(
    in_ptr, out_ptr, weight_ptr, bias_ptr,
    batch, out_ch, out_h, out_w, in_ch, in_h, in_w,
    k_h, k_w,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(out_ch, BLOCK_SIZE_N)
    num_pid_n = tl.cdiv(out_h * out_w, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_out_ch = offs_m
    offs_hw = offs_n
    out_ch_mask = offs_out_ch < out_ch

    weight = tl.load(
        weight_ptr + offs_out_ch[:, None] * (in_ch * k_h * k_w) + offs_k[None, :],
        mask=out_ch_mask[:, None],
        other=0.0
    )

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for c in range(in_ch):
        for kh in range(k_h):
            for kw in range(k_w):
                h_offset = kh * dilation_h - pad_h
                w_offset = kw * dilation_w - pad_w
                in_h_coord = ((offs_hw // out_w) * stride_h) + h_offset
                in_w_coord = ((offs_hw % out_w) * stride_w) + w_offset
                in_hw_valid = (in_h_coord >= 0) & (in_h_coord < in_h) & (in_w_coord >= 0) & (in_w_coord < in_w)
                in_hw = in_h_coord * in_w + in_w_coord
                in_nhw = (tl.arange(0, batch)[:, None] * in_h * in_w) + in_hw[None, :]
                in_ptrs = in_ptr + in_nhw * in_ch + c
                in_vals = tl.load(in_ptrs, mask=in_hw_valid[None, :], other=0.0)
                in_vals_flat = tl.reshape(in_vals, (batch * BLOCK_SIZE_M,))
                w_val = tl.load(
                    weight_ptr + offs_out_ch * (in_ch * k_h * k_w) + c * (k_h * k_w) + kh * k_w + kw,
                    mask=out_ch_mask,
                    other=0.0
                )
                acc += w_val[:, None] * in_vals_flat[None, :]
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_m, mask=out_ch_mask, other=0.0)
        acc += bias[:, None]
    acc = _relu(acc)
    out_nhw = (tl.arange(0, batch)[:, None] * out_h * out_w) + offs_hw[None, :]
    out_ptrs = out_ptr + out_nhw * out_ch + offs_m
    tl.store(out_ptrs, acc, mask=in_hw_valid[None, :])


def triton_conv2d_nhwc(x, weight, bias, stride, padding, dilation):
    batch, in_h, in_w, in_ch = x.shape
    out_ch, _, k_h, k_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    out_h = (in_h + 2 * pad_h - dilation_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (k_w - 1) - 1) // stride_w + 1
    y = torch.empty((batch, out_h, out_w, out_ch), device=x.device, dtype=x.dtype)
    def grid(META): return (triton.cdiv(out_ch, META['BLOCK_SIZE_N']) * triton.cdiv(out_h * out_w, META['BLOCK_SIZE_M']),)
    matmul_no_atomic_kernel[grid](
        x, weight, y,
        batch * out_h * out_w, out_ch, in_ch * k_h * k_w,
        x.stride(0), x.stride(-1),
        weight.stride(0), weight.stride(-1),
        y.stride(0), y.stride(-1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION="relu"
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        ])

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        for layer in self.features:
            if isinstance(layer, nn.Conv2d):
                w = layer.weight.permute(1, 2, 3, 0).contiguous()
                b = layer.bias
                x = triton_conv2d_nhwc(x, w, b, layer.stride, layer.padding, layer.dilation)
            elif isinstance(layer, nn.ReLU):
                x = triton_activation(x, "relu")
            elif isinstance(layer, nn.MaxPool2d):
                x = F.max_pool2d(x.permute(0, 3, 1, 2).contiguous(), kernel_size=2, stride=2).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = triton_matmul(x, self.classifier[0].weight.t(), activation="relu") + self.classifier[0].bias
        x = triton_matmul(x, self.classifier[1].weight.t(), activation="relu") + self.classifier[1].bias
        x = triton_matmul(x, self.classifier[2].weight.t(), activation="none") + self.classifier[2].bias
        return x
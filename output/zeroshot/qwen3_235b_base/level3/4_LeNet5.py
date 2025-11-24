import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_relu(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    BLOCK_SIZE = 1024
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def matmul_relu_kernel(
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

    c = accumulator.to(tl.float32)
    if ACTIVATION:
        c = tl.where(c > 0, c, 0.0)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul_relu(a, b, activation=True):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_relu_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION=activation
    )
    return c


@triton.jit
def max_pool2d_kernel(
    x_ptr,
    y_ptr,
    N, C, H, W,
    KH, KW,
    SH, SW,
    PH, PW,
    OH, OW,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_c = tl.program_id(1)

    hw_block_size = min(BLOCK_SIZE_HW, OH * OW)
    hw_pid = pid_nhw % (OH * OW // hw_block_size)
    n_pid = pid_nhw // (OH * OW // hw_block_size)

    hw_start = hw_pid * hw_block_size
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE_HW)
    oh = hw_offsets // OW
    ow = hw_offsets % OW

    n_offsets = n_pid + tl.arange(0, BLOCK_SIZE_N)
    mask_n = n_offsets < N

    h_start = oh * SH
    w_start = ow * SW

    input_offsets = (
        n_offsets[:, None, None] * C * H * W +
        pid_c * H * W +
        (h_start[None, :, None] + tl.arange(0, KH)[None, None, :]) * W +
        (w_start[None, None, :] + tl.arange(0, KW)[None, :, None])
    )
    mask_hw = hw_offsets < OH * OW
    mask = mask_n[:, None, None] & mask_hw[None, :, None] & (oh[None, :, None] < OH) & (ow[None, None, :] < OW)

    x = tl.load(x_ptr + input_offsets, mask=mask, other=-float('inf'))
    y = tl.max(x, axis=[1, 2])

    output_offsets = n_offsets[:, None] * C * OH * OW + pid_c * OH * OW + oh * OW + ow
    output_mask = mask_n[:, None] & mask_hw[None, :]
    tl.store(y_ptr + output_offsets, y, mask=output_mask)


def triton_max_pool2d(x, kernel_size=2, stride=2, padding=0):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    N, C, H, W = x.shape
    if isinstance(kernel_size, int):
        KH, KW = kernel_size, kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        SH, SW = stride, stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH, PW = padding, padding
    else:
        PH, PW = padding
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    y = torch.empty((N, C, OH, OW), dtype=x.dtype, device=x.device)
    grid = lambda meta: ((triton.cdiv(N, meta['BLOCK_SIZE_N']) * (OH * OW) // meta['BLOCK_SIZE_HW'], C))
    max_pool2d_kernel[grid](
        x, y,
        N, C, H, W,
        KH, KW, SH, SW, PH, PW,
        OH, OW,
        BLOCK_SIZE_N=4,
        BLOCK_SIZE_HW=64
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 16*5*5)

        x = triton_matmul_relu(x, self.fc1.weight.t(), activation=True) + self.fc1.bias
        x = triton_matmul_relu(x, self.fc2.weight.t(), activation=True) + self.fc2.bias
        x = torch.matmul(x, self.fc3.weight.t()) + self.fc3.bias

        return x
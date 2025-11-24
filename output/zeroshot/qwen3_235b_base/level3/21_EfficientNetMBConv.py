import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_stride, n_channels, image_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_ch = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_ch = offset_ch < n_channels

    # Compute mean
    mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    cnt = 0
    for idx in range(0, image_size, 1):
        offset = pid * BLOCK_SIZE * image_size + idx * n_channels + offset_ch
        mask = (offset_ch < n_channels) & (idx < image_size)
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        mean += x
        cnt += 1
    mean = mean / cnt

    # Compute variance
    var = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for idx in range(0, image_size, 1):
        offset = pid * BLOCK_SIZE * image_size + idx * n_channels + offset_ch
        mask = (offset_ch < n_channels) & (idx < image_size)
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        var += x_centered * x_centered
    var = var / cnt
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine transform
    weight = tl.load(weight_ptr + offset_ch, mask=mask_ch, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offset_ch, mask=mask_ch, other=0.0).to(tl.float32)
    for idx in range(0, image_size, 1):
        offset_in = pid * BLOCK_SIZE * image_size + idx * n_channels + offset_ch
        offset_out = pid * BLOCK_SIZE * image_size + idx * n_channels + offset_ch
        mask = (offset_ch < n_channels) & (idx < image_size)
        x = tl.load(x_ptr + offset_in, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * weight + bias
        tl.store(out_ptr + offset_out, out, mask=mask)


def triton_layer_norm(input, normalized_shape, weight, bias, eps=1e-5):
    n_channels = normalized_shape[0]
    batch_size = input.shape[0]
    image_size = input.shape[2] * input.shape[3]
    out = torch.empty_like(input)
    assert input.stride(1) == image_size, "Input must be contiguous per channel"

    def grid(meta): return ((n_channels + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], batch_size)

    _layer_norm_kernel[grid](
        input, weight, bias, out,
        input.stride(0), n_channels, image_size,
        eps,
        BLOCK_SIZE=1024
    )
    return out


@triton.jit
def _add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x + y > 0, x + y, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add_relu(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _add_relu_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


@triton.jit
def _conv2d_1x1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    batch, in_channels, out_channels, height, width,
    stride_x, stride_y, stride_h, stride_w,
    stride_out_x, stride_out_y, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(out_channels, BLOCK_M)
    pid_m = pid // num_blocks_n
    pid_n = pid % num_blocks_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    w = tl.load(w_ptr + offs_m[:, None] * stride_y + offs_n[None, :] * stride_x,
                mask=(offs_m < out_channels)[:, None] & (offs_n < in_channels)[None, :], other=0.0)

    for b in range(batch):
        for i in range(height):
            for j in range(width):
                offs_k = tl.arange(0, in_channels)
                x = tl.load(x_ptr + b * stride_x + i * stride_h + j * stride_w + offs_k,
                            mask=offs_k < in_channels, other=0.0)
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                acc += tl.dot(w, x[None, :], trans_b=False)
                acc = acc.to(tl.float16)
                if b_ptr:
                    acc += tl.load(b_ptr + offs_m, mask=offs_m < out_channels, other=0.0)[:, None]
                out_offset = b * stride_out_x + i * stride_out_h + j * stride_out_w + \
                             offs_m * stride_out_y + offs_n[None, :] * stride_out_w
                out_mask = (offs_m < out_channels)[:, None] & (offs_n < out_channels)[None, :]
                tl.store(out_ptr + out_offset, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['in_channels', 'out_channels'],
)
@triton.jit
def _matmul_no_reduce(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_tile_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_tile_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_tile_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_tile_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def triton_conv2d_1x1(x, weight, bias=None, groups=1):
    if groups != 1:
        return F.conv2d(x, weight, bias, groups=groups)
    batch, in_channels, height, width = x.shape
    out_channels, _, _, _ = weight.shape
    x_reshaped = x.view(batch, in_channels, -1).transpose(1, 2)  # (B, H*W, C_in)
    w_reshaped = weight.view(out_channels, in_channels)
    output_reshaped = torch.empty(batch, height * width, out_channels, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(out_channels, meta['BLOCK_M']) * triton.cdiv(in_channels, meta['BLOCK_N']), batch)
    _matmul_no_reduce[grid](
        x_reshaped, w_reshaped, output_reshaped,
        M=batch * height * width, N=out_channels, K=in_channels,
        stride_am=x_reshaped.stride(0), stride_ak=1,
        stride_bk=w_reshaped.stride(0), stride_bn=1,
        stride_cm=output_reshaped.stride(0), stride_cn=1,
    )
    if bias is not None:
        output_reshaped += bias
    out = output_reshaped.transpose(1, 2).view(batch, out_channels, height, width)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv[0](x)
        x = triton_layer_norm(x, (x.shape[1],), self.project_conv[1].weight, self.project_conv[1].bias)
        
        if self.use_residual:
            x = triton_add_relu(x, identity)
        
        return x
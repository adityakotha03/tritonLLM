import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid_x = tl.sigmoid(x)
    swish_x = x * sigmoid_x

    tl.store(out_ptr + offsets, swish_x, mask=mask)


@triton.jit
def hardswish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    zero = tl.full((BLOCK_SIZE,), 0.0, dtype=tl.float32)
    three = tl.full((BLOCK_SIZE,), 3.0, dtype=tl.float32)
    six = tl.full((BLOCK_SIZE,), 6.0, dtype=tl.float32)
    threshold = tl.where(x <= -3.0, zero, tl.where(x >= 3.0, x, x * (x + three) / six))
    hardswish_x = tl.where(x <= -3.0, zero, threshold)

    tl.store(out_ptr + offsets, hardswish_x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
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

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float32)
    if ACTIVATION == "swish":
        sigmoid_c = tl.sigmoid(c)
        c = c * sigmoid_c
    elif ACTIVATION == "hardswish":
        three = tl.full(c.shape, 3.0, dtype=tl.float32)
        six = tl.full(c.shape, 6.0, dtype=tl.float32)
        zero = tl.full(c.shape, 0.0, dtype=tl.float32)
        threshold = tl.where(c <= -3.0, zero, tl.where(c >= 3.0, c, c * (c + three) / six))
        c = tl.where(c <= -3.0, zero, threshold)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_swish(x):
    assert x.is_cuda, "Input tensor must be on GPU."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    swish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_hardswish(x):
    assert x.is_cuda, "Input tensor must be on GPU."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
    return out


def triton_conv_transpose3d_swish(
    x, weight, bias=None,
    stride=(1, 1, 1), padding=(0, 0, 0), output_padding=(0, 0, 0), groups=1
):
    batch, in_channels, in_depth, in_height, in_width = x.shape
    _, out_channels_per_group, kw, kh, kw_ = weight.shape
    out_channels = out_channels_per_group * groups
    kernel_size = (kw, kh, kw_)
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    out_depth = (in_depth - 1) * stride_d - 2 * pad_d + kernel_size[0] + out_pad_d
    out_height = (in_height - 1) * stride_h - 2 * pad_h + kernel_size[1] + out_pad_h
    out_width = (in_width - 1) * stride_w - 2 * pad_w + kernel_size[2] + out_pad_w

    x_unf = torch.nn.functional.unfold(
        x.transpose(1, 2).reshape(batch * in_depth, in_channels, in_height, in_width),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w)
    )
    x_unf = x_unf.view(batch, in_depth, in_channels * kernel_size[1] * kernel_size[2], -1)
    x_unf = x_unf.permute(0, 2, 1, 3).reshape(batch, in_channels * kernel_size[1] * kernel_size[2], -1)

    weight_flat = weight.view(groups, -1, in_channels // groups * kernel_size[0] * kernel_size[1] * kernel_size[2])
    weight_flat = weight_flat.permute(0, 2, 1).contiguous()

    out_unf = torch.zeros(batch, groups, out_channels // groups, out_depth * out_height * out_width, device=x.device, dtype=x.dtype)
    for g in range(groups):
        w_g = weight_flat[g]
        x_g = x_unf[:, g * w_g.shape[1]:(g + 1) * w_g.shape[1], :]
        M, N, K = x_g.shape[0] * x_g.shape[2], w_g.shape[1], x_g.shape[1]
        x_g = x_g.permute(0, 2, 1).contiguous().view(-1, K)
        c = torch.empty((M, N), device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        matmul_kernel[grid](
            x_g, w_g, c,
            M, N, K,
            x_g.stride(0), x_g.stride(1),
            w_g.stride(0), w_g.stride(1),
            c.stride(0), c.stride(1),
            ACTIVATION="swish"
        )
        c = c.view(batch, out_depth * out_height * out_width, -1).permute(0, 2, 1)
        out_unf[:, g, :, :] = c

    out_unf = out_unf.reshape(batch, out_channels, out_depth * out_height * out_width)
    out = torch.nn.functional.fold(
        out_unf, output_size=(out_depth, out_height), kernel_size=(1, 1)
    ).view(batch, out_channels, out_depth, out_height, out_width)

    if bias is not None:
        out += bias.view(1, -1, 1, 1, 1)

    return out


class GroupNormKernel:
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device='cuda'):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = torch.ones(num_channels, device=device)
            self.bias = torch.zeros(num_channels, device=device)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x):
        N, C, D, H, W = x.shape
        G = self.num_groups
        x = x.view(N, G, C // G, D, H, W)
        mean = x.mean(dim=(2, 3, 4, 5), keepdim=True)
        var = x.var(dim=(2, 3, 4, 5), unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(N, C, D, H, W)

        if self.affine:
            x_norm = x_norm * self.weight.view(1, C, 1, 1, 1) + self.bias.view(1, C, 1, 1, 1)
        return x_norm


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.eps = eps
        self.bias = bias

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size
        ))
        if bias:
            self.bias_param = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_param', None)

        self.group_norm_op = GroupNormKernel(groups, out_channels, eps, affine=True)

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)

    def forward(self, x):
        x = triton_conv_transpose3d_swish(
            x, self.weight, self.bias_param,
            stride=(self.stride, self.stride, self.stride),
            padding=(self.padding, self.padding, self.padding),
            output_padding=(0, 0, 0),
            groups=self.groups
        )
        x = self.group_norm_op(x)
        x = triton_hardswish(x)
        return x
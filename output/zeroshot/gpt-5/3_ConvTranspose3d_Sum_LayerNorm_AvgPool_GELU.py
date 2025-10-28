import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_layernorm_ncdhw_kernel(
    x_ptr,            # *f32/f16 input
    y_ptr,            # *f32/f16 output
    gamma_ptr,        # *f32/f16 gamma (C,)
    beta_ptr,         # *f32/f16 beta (C,)
    sum_w_ptr,        # *f32/f16 scalar tensor
    N, C, D, H, W,    # sizes
    stride_n, stride_c, stride_d, stride_h, stride_w,  # input strides
    eps,              # float32 epsilon
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    P = N * D * H * W
    if pid >= P:
        return

    # Decode (n, d, h, w) from flattened pid
    w = pid % W
    tmp = pid // W
    h = tmp % H
    tmp = tmp // H
    d = tmp % D
    n = tmp // D

    base_offset = n * stride_n + d * stride_d + h * stride_h + w * stride_w

    # Load sum weight scalar
    sum_w = tl.load(sum_w_ptr).to(tl.float32)

    # First pass: compute mean and variance across C
    c_sum = tl.zeros((), dtype=tl.float32)
    c_sumsq = tl.zeros((), dtype=tl.float32)
    c = 0
    while c < C:
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_ptr + base_offset + offs * stride_c, mask=mask, other=0.0)
        x = x.to(tl.float32) + sum_w
        c_sum += tl.sum(x, axis=0)
        c_sumsq += tl.sum(x * x, axis=0)
        c += BLOCK_SIZE
    C_f = tl.float32(C)
    mean = c_sum / C_f
    var = c_sumsq / C_f - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine
    c = 0
    while c < C:
        offs = c + tl.arange(0, BLOCK_SIZE)
        mask = offs < C
        x = tl.load(x_ptr + base_offset + offs * stride_c, mask=mask, other=0.0).to(tl.float32)
        x = (x + sum_w - mean) * inv_std
        if HAS_WEIGHTS:
            gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * gamma + beta
        else:
            y = x
        tl.store(y_ptr + base_offset + offs * stride_c, y, mask=mask)
        c += BLOCK_SIZE


@triton.jit
def avgpool3d_gelu_ncdhw_kernel(
    x_ptr, y_ptr,
    N, C, Di, Hi, Wi,
    Do, Ho, Wo,
    kD, kH, kW,
    sD, sH, sW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_od = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_owb = tl.program_id(3)

    n = pid_nc // C
    c = pid_nc % C

    ow_offsets = pid_owb * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < Wo

    # Compute base for input and output (without w)
    in_d = pid_od * sD
    in_h = pid_oh * sH

    # Accumulator for sum over the pooling window
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    kd = 0
    while kd < kD:
        kh = 0
        while kh < kH:
            kw = 0
            while kw < kW:
                iw = ow_offsets * sW + kw
                id = in_d + kd
                ih = in_h + kh
                in_base = (
                    n * stride_n +
                    c * stride_c +
                    id * stride_d +
                    ih * stride_h
                )
                mask_in = mask_ow & (iw < Wi) & (id < Di) & (ih < Hi)
                vals = tl.load(x_ptr + in_base + iw * stride_w, mask=mask_in, other=0.0)
                acc += vals.to(tl.float32)
                kw += 1
            kh += 1
        kd += 1

    pool_area = tl.float32(kD * kH * kW)
    avg = acc / pool_area

    # GELU approximate (tanh)
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    gelu = 0.5 * avg * (1.0 + tl.tanh(k0 * (avg + k1 * avg * avg * avg)))

    out_base = (
        n * out_stride_n +
        c * out_stride_c +
        pid_od * out_stride_d +
        pid_oh * out_stride_h
    )
    tl.store(y_ptr + out_base + ow_offsets * out_stride_w, gelu, mask=mask_ow)


def triton_add_layernorm_ncdhw(x: torch.Tensor, sum_weight: torch.Tensor, ln: nn.LayerNorm):
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 5, "Expected NCDHW input"
    x = x.contiguous()
    y = torch.empty_like(x)

    N, C, D, H, W = x.shape
    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    eps = ln.eps if hasattr(ln, "eps") else 1e-5

    has_affine = getattr(ln, "elementwise_affine", True) and (ln.weight is not None and ln.bias is not None)
    gamma = ln.weight if has_affine else torch.empty(0, device=x.device, dtype=x.dtype)
    beta = ln.bias if has_affine else torch.empty(0, device=x.device, dtype=x.dtype)

    # Ensure parameters are on device and contiguous
    if has_affine:
        gamma = gamma.contiguous()
        beta = beta.contiguous()

    # sum_weight should be a CUDA tensor (0-dim) to avoid host sync
    if not isinstance(sum_weight, torch.Tensor):
        sum_weight = torch.tensor(sum_weight, device=x.device, dtype=x.dtype)
    else:
        if not sum_weight.is_cuda:
            sum_weight = sum_weight.to(device=x.device, dtype=x.dtype)
    sum_weight = sum_weight.contiguous()

    BLOCK_SIZE = 128 if C >= 128 else (64 if C >= 64 else 32)
    P = N * D * H * W
    grid = lambda meta: (P,)

    add_layernorm_ncdhw_kernel[grid](
        x, y,
        gamma, beta,
        sum_weight,
        N, C, D, H, W,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        eps,
        HAS_WEIGHTS=has_affine,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def triton_avgpool3d_gelu_ncdhw(x: torch.Tensor, kernel_size, stride=None):
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dim() == 5, "Expected NCDHW input"
    x = x.contiguous()
    N, C, Di, Hi, Wi = x.shape

    if isinstance(kernel_size, int):
        kD = kH = kW = kernel_size
    else:
        kD, kH, kW = kernel_size

    if stride is None:
        sD, sH, sW = kD, kH, kW
    else:
        if isinstance(stride, int):
            sD = sH = sW = stride
        else:
            sD, sH, sW = stride

    Do = (Di - kD) // sD + 1
    Ho = (Hi - kH) // sH + 1
    Wo = (Wi - kW) // sW + 1

    y = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w = y.stride()

    NC = N * C
    BLOCK_W = 64
    grid = (
        NC,
        Do,
        Ho,
        (Wo + BLOCK_W - 1) // BLOCK_W,
    )

    avgpool3d_gelu_ncdhw_kernel[grid](
        x, y,
        N, C, Di, Hi, Wi,
        Do, Ho, Wo,
        kD, kH, kW,
        sD, sH, sW,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
        BLOCK_W=BLOCK_W,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model with custom Triton kernels:
    - Fused add + LayerNorm across channel dimension (C) on NCDHW tensor
    - Fused AvgPool3d + GELU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Keep sum_weight as a learnable parameter
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        # LayerNorm parameters; we will apply across channel (C) using Triton
        self.norm = nn.LayerNorm(norm_shape)
        # Keep AvgPool config for shape/stride; Triton kernel will implement the op
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        # GELU will be fused into pooling kernel
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused add + layernorm over channels (C) on NCDHW
        x = triton_add_layernorm_ncdhw(x, self.sum_weight, self.norm)
        # Fused avgpool3d + gelu
        stride = self.avg_pool.stride if self.avg_pool.stride is not None else self.avg_pool.kernel_size
        x = triton_avgpool3d_gelu_ncdhw(x, self.avg_pool.kernel_size, stride)
        return x


batch_size = 32
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]
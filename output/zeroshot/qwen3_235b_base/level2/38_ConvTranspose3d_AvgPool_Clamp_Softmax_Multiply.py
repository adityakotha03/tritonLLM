import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_3d_kernel(
    x_ptr, output_ptr,
    batch, channels, depth, height, width,
    pool_kd, pool_kh, pool_kw,
    stride_d, stride_h, stride_w,
    out_d, out_h, out_w,
    input_d, input_h, input_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_out_d = tl.program_id(axis=2)
    pid_out_hw = tl.program_id(axis=3)

    # Compute starting positions for output tile
    d_range = pid_out_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    hw_range = pid_out_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    h_range = hw_range // out_w
    w_range = hw_range % out_w

    d_mask = d_range < out_d
    h_mask = h_range < out_h
    w_mask = w_range < out_w
    hw_mask = h_mask & w_mask
    mask = d_mask[:, None] & hw_mask[None, :]

    d_start = d_range[:, None] * stride_d
    h_start = h_range[None, :] * stride_h
    w_start = w_range[None, :] * stride_w

    pool_d = tl.arange(0, pool_kd)
    pool_h = tl.arange(0, pool_kh)
    pool_w = tl.arange(0, pool_kw)

    d_offset = d_start[:, None, None, None] + pool_d[None, :, None, None]
    h_offset = h_start[None, :, None, None] + pool_h[None, None, :, None]
    w_offset = w_start[None, None, :, None] + pool_w[None, None, None, :]

    d_valid = (d_offset >= 0) & (d_offset < input_d)
    h_valid = (h_offset >= 0) & (h_offset < input_h)
    w_valid = (w_offset >= 0) & (w_offset < input_w)

    total_valid = d_valid & h_valid & w_valid

    n_elements = tl.sum(total_valid.to(tl.int32))

    d_clamped = tl.clamp(d_offset, 0, input_d - 1)
    h_clamped = tl.clamp(h_offset, 0, input_h - 1)
    w_clamped = tl.clamp(w_offset, 0, input_w - 1)

    input_idx = pid_b * channels * input_d * input_h * input_w + \
                pid_c * input_d * input_h * input_w + \
                d_clamped * input_h * input_w + h_clamped * input_w + w_clamped

    values = tl.load(x_ptr + input_idx, mask=total_valid, other=0.0)
    pool_sum = tl.sum(values, axis=[1, 2, 3])
    pool_avg = pool_sum / (pool_kd * pool_kh * pool_kw)

    output_idx = pid_b * channels * out_d * out_h * out_w + \
                 pid_c * out_d * out_h * out_w + \
                 d_range[:, None] * out_h * out_w + h_range[None, :] * out_w + w_range[None, :]

    tl.store(output_ptr + output_idx, pool_avg, mask=mask)


@triton.jit
def conv_transpose_3d_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels, out_channels, input_d, input_h, input_w,
    output_d, output_h, output_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    groups,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_DHW: tl.constexpr
):
    pid_b = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_ohw = tl.program_id(axis=2)

    oc_block_start = pid_oc * BLOCK_SIZE_C
    oc_range = oc_block_start + tl.arange(0, BLOCK_SIZE_C)
    oc_mask = oc_range < out_channels

    ohw_block_start = pid_ohw * BLOCK_SIZE_DHW
    ohw_range = ohw_block_start + tl.arange(0, BLOCK_SIZE_DHW)
    ow_range = ohw_range % output_w
    oh_range = (ohw_range // output_w) % output_h
    od_range = ohw_range // (output_h * output_w)
    od_mask = od_range < output_d
    oh_mask = oh_range < output_h
    ow_mask = ow_range < output_w
    ohw_mask = od_mask & oh_mask & ow_mask
    mask = oc_mask[:, None] & ohw_mask[None, :]

    id_start = od_range * stride_d - padding_d
    ih_start = oh_range * stride_h - padding_h
    iw_start = ow_range * stride_w - padding_w

    kd_range = tl.arange(0, kernel_d)
    kh_range = tl.arange(0, kernel_h)
    kw_range = tl.arange(0, kernel_w)

    id_offset = id_start[None, None, None, :] + kd_range[:, None, None, None]
    ih_offset = ih_start[None, None, :, None] + kh_range[None, :, None, None]
    iw_offset = iw_start[None, :, None, None] + kw_range[None, None, :, None]

    valid_id = (id_offset >= 0) & (id_offset < input_d)
    valid_ih = (ih_offset >= 0) & (ih_offset < input_h)
    valid_iw = (iw_offset >= 0) & (iw_offset < input_w)
    valid = valid_id & valid_ih & valid_iw

    for ic in range(0, in_channels, BLOCK_SIZE_C):
        ic_block_start = ic
        ic_range = ic_block_start + tl.arange(0, BLOCK_SIZE_C)
        ic_mask = ic_range < in_channels

        weight_idx = oc_range[:, None, None, None, None] * in_channels * kernel_d * kernel_h * kernel_w + \
                     ic_range[None, :, None, None, None] * kernel_d * kernel_h * kernel_w + \
                     kd_range[None, None, :, None, None] * kernel_h * kernel_w + \
                     kh_range[None, None, None, :, None] * kernel_w + \
                     kw_range[None, None, None, None, :]
        weight = tl.load(weight_ptr + weight_idx, mask=oc_mask[:, None, None, None, None] & ic_mask[None, :, None, None, None], other=0.0)

        input_idx = pid_b * in_channels * input_d * input_h * input_w + \
                    ic_range[None, :, None, None, None] * input_d * input_h * input_w + \
                    id_offset[None, None, :, :, :] * input_h * input_w + \
                    ih_offset[None, None, :, :, :] * input_w + \
                    iw_offset[None, None, :, :, :]
        input_val = tl.load(x_ptr + input_idx, mask=valid[None, None, :, :, :] & ic_mask[:, :, None, None, None], other=0.0)

        local_sum = tl.sum(input_val * weight[None, :, :, :, :], axis=[1, 2, 3, 4])

        tl.atomic_add(output_ptr + pid_b * out_channels * output_d * output_h * output_w + oc_range * output_d * output_h * output_w + od_range * output_h * output_w + oh_range * output_w + ow_range,
                      local_sum, mask=mask)


@triton.jit
def clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, output_ptr,
    batch, channels, depth, height, width,
    clamp_min, clamp_max,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    total_elements = batch * channels * depth * height * width
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x_clamped = tl.clamp(x, clamp_min, clamp_max)

    b = offsets // (channels * depth * height * width)
    remainder = offsets % (channels * depth * height * width)
    c = remainder // (depth * height * width)

    scale = tl.load(scale_ptr + c, mask=mask, other=1.0)
    x_scaled = x_clamped * scale

    x_flat = x_scaled
    x_flat = tl.where(mask, x_flat, -float('inf'))

    max_val = tl.max(x_flat, axis=0)
    x_shifted = x_flat - max_val
    exp_val = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_val, axis=0)
    softmax_val = exp_val / (sum_exp + 1e-8)

    tl.store(output_ptr + offsets, softmax_val, mask=mask)


def triton_avg_pool3d(x, kernel_size, stride, padding):
    if isinstance(kernel_size, int):
        kd = kh = kw = kernel_size
    else:
        kd, kh, kw = kernel_size
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride
    if isinstance(padding, int):
        pd = ph = pw = padding
    else:
        pd, ph, pw = padding

    b, c, d, h, w = x.shape
    out_d = (d + 2 * pd - kd) // sd + 1
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    out = torch.empty((b, c, out_d, out_h, out_w), dtype=x.dtype, device=x.device)

    grid = lambda meta: (b, c, triton.cdiv(out_d, meta['BLOCK_SIZE_D']), triton.cdiv(out_h * out_w, meta['BLOCK_SIZE_HW']))
    avg_pool_3d_kernel[grid](
        x, out,
        b, c, d, h, w,
        kd, kh, kw,
        sd, sh, sw,
        out_d, out_h, out_w,
        d, h, w,
        BLOCK_SIZE_D=16,
        BLOCK_SIZE_HW=256
    )
    return out


def triton_conv_transpose3d(x, weight, bias, stride, padding, output_padding):
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride
    if isinstance(padding, int):
        pd = ph = pw = padding
    else:
        pd, ph, pw = padding
    if isinstance(output_padding, int):
        opd = oph = opw = output_padding
    else:
        opd, oph, opw = output_padding

    b, in_c, in_d, in_h, in_w = x.shape
    out_c, _, kd, kh, kw = weight.shape

    out_d = (in_d - 1) * sd - 2 * pd + kd + opd
    out_h = (in_h - 1) * sh - 2 * ph + kh + oph
    out_w = (in_w - 1) * sw - 2 * pw + kw + opw

    out = torch.zeros((b, out_c, out_d, out_h, out_w), dtype=x.dtype, device=x.device)

    grid = lambda meta: (b, triton.cdiv(out_c, meta['BLOCK_SIZE_C']), triton.cdiv(out_d * out_h * out_w, meta['BLOCK_SIZE_DHW']))
    conv_transpose_3d_kernel[grid](
        x, weight, bias, out,
        b, in_c, out_c, in_d, in_h, in_w,
        out_d, out_h, out_w,
        kd, kh, kw,
        sd, sh, sw,
        pd, ph, pw,
        opd, oph, opw,
        1,
        BLOCK_SIZE_C=16,
        BLOCK_SIZE_DHW=256
    )
    return out


def triton_clamp_softmax_scale(x, clamp_min, clamp_max, scale):
    b, c, d, h, w = x.shape
    total_elements = b * c * d * h * w
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    clamp_softmax_scale_kernel[grid](
        x, scale,
        out,
        b, c, d, h, w,
        clamp_min, clamp_max,
        BLOCK_SIZE=1024
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for average pooling, transposed convolution,
    clamping, softmax, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.pool_kernel_size = pool_kernel_size
        self.conv_transpose_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_transpose_bias = nn.Parameter(torch.zeros(out_channels))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = triton_avg_pool3d(x, self.pool_kernel_size, self.pool_kernel_size, 0)
        x = triton_conv_transpose3d(x, self.conv_transpose_weight, self.conv_transpose_bias, self.stride, self.padding, self.output_padding)
        x = triton_clamp_softmax_scale(x, self.clamp_min, self.clamp_max, self.scale.squeeze())
        return x
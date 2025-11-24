import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, norm_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < norm_size

    # Compute mean
    row_offset = pid * norm_size
    x_row = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    mean = tl.sum(x_row, axis=0) / norm_size

    # Compute variance
    diff = tl.where(offsets < norm_size, x_row - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / norm_size

    # Normalize and apply affine transform
    inv_stdev = 1.0 / tl.sqrt(var + eps)
    normed = diff * inv_stdev
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    output = normed * weight + bias
    tl.store(out_ptr + row_offset + offsets, output, mask=mask)


@triton.jit
def avg_pool_3d_kernel(
    x_ptr, out_ptr,
    batch, out_channels, out_d, out_h, out_w,
    in_d, in_h, in_w,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // (out_channels * out_d)
    residual = pid % (out_channels * out_d)
    ch = residual // out_d
    d = residual % out_d

    # Define ranges for spatial dimensions
    offs_hw = tl.arange(0, BLOCK_SIZE_HW)
    offs_d = tl.arange(0, BLOCK_SIZE_D)

    # Loop over output height and width in blocks
    for h0 in range(0, out_h, BLOCK_SIZE_HW):
        for w0 in range(0, out_w, BLOCK_SIZE_HW):
            h_block = h0 + offs_hw
            w_block = w0 + offs_hw
            mask_hw = (h_block < out_h)[:, None] & (w_block < out_w)[None, :]

            # Compute input indices
            in_h_start = h_block * stride_h
            in_w_start = w_block * stride_w
            in_d_start = d * stride_d + offs_d[:, None, None]

            # Load input values
            in_offsets = (batch_id * out_channels * in_d * in_h * in_w +
                         ch * in_d * in_h * in_w +
                         in_d_start * in_h * in_w +
                         in_h_start[:, None, None] * in_w +
                         in_w_start[None, :, None])
            x = tl.load(x_ptr + in_offsets, mask=mask_hw[None, :, :], other=0.0)

            # Average over kernel
            pooled = tl.sum(tl.sum(tl.sum(x, axis=0), axis=1), axis=0) / (kernel_d * kernel_h * kernel_w)
            out_offsets = (batch_id * out_channels * out_d * out_h * out_w +
                          ch * out_d * out_h * out_w +
                          d * out_h * out_w +
                          h_block[:, None] * out_w +
                          w_block[None, :])
            tl.store(out_ptr + out_offsets, pooled, mask=mask_hw)


@triton.jit
def gelu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh method
    out = x * 0.5 * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    batch_size = x.numel() // weight.numel()
    norm_size = weight.numel()
    BLOCK_SIZE = triton.next_power_of_2(norm_size)
    grid = lambda meta: (batch_size,)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        batch_size, norm_size,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_avg_pool_3d(x, kernel_size, stride):
    assert x.is_cuda
    x = x.contiguous()
    batch, ch, d, h, w = x.shape
    kd, kh, kw = kernel_size
    sd, sh, sw = stride
    out_d, out_h, out_w = (d - kd) // sd + 1, (h - kh) // sh + 1, (w - kw) // sw + 1
    out = torch.empty((batch, ch, out_d, out_h, out_w), device=x.device, dtype=x.dtype)

    BLOCK_SIZE_D = 4
    BLOCK_SIZE_HW = 16
    grid = lambda meta: (batch * ch * out_d,)
    avg_pool_3d_kernel[grid](
        x, out,
        batch, ch, out_d, out_h, out_w,
        d, h, w,
        kd, kh, kw,
        sd, sh, sw,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW
    )
    return out


def triton_gelu(x):
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for LayerNorm, AvgPool3d, and GELU.
    ConvTranspose3d and addition with scalar are kept as-is due to complexity and limited gain.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        x = triton_layer_norm(x, self.norm.weight, self.norm.bias, self.norm.eps)
        x = triton_avg_pool_3d(x, self.pool_kernel_size, self.stride)
        x = triton_gelu(x)
        return x
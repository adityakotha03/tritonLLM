import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_transpose3d_avgpool_bias_scale_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, in_depth, in_height, in_width,
    out_depth, out_height, out_width,
    kernel_d, kernel_h, kernel_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    avgpool_kernel_d, avgpool_kernel_h, avgpool_kernel_w,
    scale1, scale2,
    input_stride_cd, input_stride_ch, input_stride_cw,
    input_stride_ncd, input_stride_nch, input_stride_ncw,
    input_stride_n, input_stride_c,
    weight_stride_kcdd, weight_stride_kcdw, weight_stride_kcd,
    weight_stride_kc, weight_stride_k,
    output_stride_kd, output_stride_kh, output_stride_kw,
    output_stride_nd, output_stride_nh, output_stride_nw,
    output_stride_n, output_stride_k,
    bias_ptr_k,
    n_elements,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(out_depth * out_height * out_width, BLOCK_SIZE_N)
    pid_k = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_k = pid_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_d = offs_n // (out_height * out_width)
    offs_h = (offs_n % (out_height * out_width)) // out_width
    offs_w = offs_n % out_width

    d_mask = offs_d < out_depth
    h_mask = offs_h < out_height
    w_mask = offs_w < out_width
    n_mask = d_mask & h_mask & w_mask

    input_d = offs_d * stride_d - padding_d
    input_h = offs_h * stride_h - padding_h
    input_w = offs_w * stride_w - padding_w

    weight_d = tl.arange(0, kernel_d)
    weight_h = tl.arange(0, kernel_h)
    weight_w = tl.arange(0, kernel_w)

    input_d = input_d[:, None, None, None] + weight_d[None, :, None, None] * dilation_d
    input_h = input_h[:, None, None, None] + weight_h[None, None, :, None] * dilation_h
    input_w = input_w[:, None, None, None] + weight_w[None, None, None, :] * dilation_w

    d_in_bounds = (input_d >= 0) & (input_d < in_depth)
    h_in_bounds = (input_h >= 0) & (input_h < in_height)
    w_in_bounds = (input_w >= 0) & (input_w < in_width)

    input_d = tl.where(d_in_bounds, input_d, 0)
    input_h = tl.where(h_in_bounds, input_h, 0)
    input_w = tl.where(w_in_bounds, 0, input_w)

    input_c = tl.arange(0, in_channels)
    input_nc_offset = input_d * input_stride_cd + input_h * input_stride_ch + input_w * input_stride_cw
    input_n_offset = offs_d * input_stride_nd + offs_h * input_stride_nh + offs_w * input_stride_nw
    input_offset = input_nc_offset[:, :, :, :, None] + input_n_offset[:, None, None, None, None] + input_c[None, None, None, None, :] * input_stride_c
    input_mask = n_mask[:, None, None, None, None] & d_in_bounds[:, :, None, None, None] & h_in_bounds[:, None, :, None, None] & w_in_bounds[:, None, None, :, None]

    weight_offset = offs_k[:, None, None, None, None] * weight_stride_k + input_c[None, None, None, None, :] * weight_stride_kc + weight_d[None, :, None, None, None] * weight_stride_kcdd + weight_h[None, None, :, None, None] * weight_stride_kcdw + weight_w[None, None, None, :, None] * weight_stride_kcd
    weight_mask = offs_k < out_channels

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ic in range(0, in_channels * kernel_d * kernel_h * kernel_w, BLOCK_SIZE_K):
        k_offs = ic + tl.arange(0, BLOCK_SIZE_K)
        weight_ptrs = weight_ptr + weight_offset + k_offs[None, None, None, None, :]
        weight_mask_k = weight_mask[:, None, None, None, None] & (k_offs[None, None, None, None, :] < in_channels * kernel_d * kernel_h * kernel_w)
        weight = tl.load(weight_ptrs, mask=weight_mask_k, other=0.0)
        input_ptrs = input_ptr + input_offset + k_offs[None, :, :, :, :] * input_stride_c
        input_vals = tl.load(input_ptrs, mask=input_mask & (k_offs[None, :, :, :, :] < in_channels * kernel_d * kernel_h * kernel_w)[None, :, :, :, :], other=0.0)
        acc += tl.dot(weight, input_vals.to(tl.float32), out_dtype=tl.float32)

    out = acc.to(tl.float32) * scale1

    # Apply average pooling (2x2x2)
    pool_out_d = out_depth * 2
    pool_out_h = out_height * 2
    pool_out_w = out_width * 2
    pool_out_nhw = pool_out_d * pool_out_h * pool_out_w
    pool_pid_n = pid_n * BLOCK_SIZE_N
    pool_offs_n = pool_pid_n + tl.arange(0, BLOCK_SIZE_N)
    pool_d = pool_offs_n // (pool_out_h * pool_out_w)
    pool_h = (pool_offs_n % (pool_out_h * pool_out_w)) // pool_out_w
    pool_w = pool_offs_n % pool_out_w
    pool_mask = pool_offs_n < pool_out_nhw
    pool_d0 = pool_d // 2
    pool_h0 = pool_h // 2
    pool_w0 = pool_w // 2
    pool_d1 = (pool_d + 1) // 2
    pool_h1 = (pool_h + 1) // 2
    pool_w1 = (pool_w + 1) // 2

    val000 = tl.load(out_ptr + pool_d0 * output_stride_kd + pool_h0 * output_stride_kh + pool_w0 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d0 < out_depth) & (pool_h0 < out_height) & (pool_w0 < out_width), other=0.0)
    val001 = tl.load(out_ptr + pool_d0 * output_stride_kd + pool_h0 * output_stride_kh + pool_w1 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d0 < out_depth) & (pool_h0 < out_height) & (pool_w1 < out_width), other=0.0)
    val010 = tl.load(out_ptr + pool_d0 * output_stride_kd + pool_h1 * output_stride_kh + pool_w0 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d0 < out_depth) & (pool_h1 < out_height) & (pool_w0 < out_width), other=0.0)
    val011 = tl.load(out_ptr + pool_d0 * output_stride_kd + pool_h1 * output_stride_kh + pool_w1 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d0 < out_depth) & (pool_h1 < out_height) & (pool_w1 < out_width), other=0.0)
    val100 = tl.load(out_ptr + pool_d1 * output_stride_kd + pool_h0 * output_stride_kh + pool_w0 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d1 < out_depth) & (pool_h0 < out_height) & (pool_w0 < out_width), other=0.0)
    val101 = tl.load(out_ptr + pool_d1 * output_stride_kd + pool_h0 * output_stride_kh + pool_w1 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d1 < out_depth) & (pool_h0 < out_height) & (pool_w1 < out_width), other=0.0)
    val110 = tl.load(out_ptr + pool_d1 * output_stride_kd + pool_h1 * output_stride_kh + pool_w0 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d1 < out_depth) & (pool_h1 < out_height) & (pool_w0 < out_width), other=0.0)
    val111 = tl.load(out_ptr + pool_d1 * output_stride_kd + pool_h1 * output_stride_kh + pool_w1 * output_stride_kw + pid_k * output_stride_k, mask=pool_mask & (pool_d1 < out_depth) & (pool_h1 < out_height) & (pool_w1 < out_width), other=0.0)

    pool_out = (val000 + val001 + val010 + val011 + val100 + val101 + val110 + val111) * 0.125

    # Add bias and scale2
    bias = tl.load(bias_ptr + pid_k * bias_ptr_k, mask=pid_k < out_channels, other=0.0)
    pool_out = pool_out + bias
    pool_out = pool_out * scale2

    output_offs = pool_offs_n + pid_k * pool_out_nhw
    tl.store(output_ptr + output_offs, pool_out, mask=pool_mask)


class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernel for transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale1 = scale1
        self.scale2 = scale2
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv_transpose_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.conv_transpose_weight)

    def forward(self, x):
        in_depth, in_height, in_width = x.shape[-3:]
        out_depth = (in_depth - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.kernel_size

        pool_out_depth = out_depth // 2
        pool_out_height = out_height // 2
        pool_out_width = out_width // 2

        output = torch.empty(x.shape[0], self.out_channels, pool_out_depth, pool_out_height, pool_out_width, device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(self.out_channels, meta['BLOCK_SIZE_M']) * triton.cdiv(pool_out_depth * pool_out_height * pool_out_width, meta['BLOCK_SIZE_N']),)

        fused_conv_transpose3d_avgpool_bias_scale_kernel[grid](
            x, self.conv_transpose_weight, self.bias, output,
            self.in_channels, self.out_channels, in_depth, in_height, in_width,
            out_depth, out_height, out_width,
            self.kernel_size, self.kernel_size, self.kernel_size,
            self.stride, self.stride, self.stride,
            self.padding, self.padding, self.padding,
            1, 1, 1,
            2, 2, 2,
            self.scale1, self.scale2,
            x.stride(2), x.stride(3), x.stride(4),
            x.stride(1), x.stride(2), x.stride(3),
            x.stride(0), x.stride(1),
            self.conv_transpose_weight.stride(0), self.conv_transpose_weight.stride(1), self.conv_transpose_weight.stride(2),
            self.conv_transpose_weight.stride(3), self.conv_transpose_weight.stride(4),
            output.stride(2), output.stride(3), output.stride(4),
            output.stride(1), output.stride(2), output.stride(3),
            output.stride(0), output.stride(1),
            self.bias.stride(0),
            output.numel(),
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64
        )
        return output
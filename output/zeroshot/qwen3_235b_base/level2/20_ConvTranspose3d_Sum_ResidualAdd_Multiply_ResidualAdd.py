import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_transpose3d_bias_res_mul_res_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, input_depth, input_height, input_width,
    output_depth, output_height, output_width,
    kernel_size_d, kernel_size_h, kernel_size_w,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    output_padding_d, output_padding_h, output_padding_w,
    input_stride_c, input_stride_d, input_stride_h,
    output_stride_c, output_stride_d, output_stride_h,
    weight_stride_k, weight_stride_r, weight_stride_s, weight_stride_t,
    n_elements, has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (tl.cdiv(out_channels, BLOCK_SIZE_M))
    oc_block_idx = pid % (tl.cdiv(out_channels, BLOCK_SIZE_M))

    # Pointers for this batch and output channel block
    input_ptr += batch_idx * input_stride_c * in_channels
    output_ptr += batch_idx * output_stride_c * out_channels
    output_block_ptr = output_ptr + oc_block_idx * BLOCK_SIZE_M * output_stride_c

    # Load weights for this output channel block
    weight_offset = oc_block_idx * BLOCK_SIZE_M * weight_stride_k
    weight_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < out_channels - oc_block_idx * BLOCK_SIZE_M) & \
                  (tl.arange(0, BLOCK_SIZE_K)[None, :] < in_channels)
    weight_ptrs = weight_ptr + weight_offset + \
                  (tl.arange(0, BLOCK_SIZE_M)[:, None] * weight_stride_k +
                   tl.arange(0, BLOCK_SIZE_K)[None, :] * weight_stride_r)
    weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

    # Load bias if present
    bias = tl.load(bias_ptr + oc_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M),
                   mask=tl.arange(0, BLOCK_SIZE_M) < out_channels - oc_block_idx * BLOCK_SIZE_M,
                   other=0.0) if has_bias else 0.0

    # Iterate over output spatial positions
    for idx in range(tl.cdiv(n_elements, BLOCK_SIZE_N)):
        out_idx = idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_out = out_idx < n_elements

        # Decode output index to (oc, od, oh, ow)
        ow = out_idx % output_width
        oh = (out_idx // output_width) % output_height
        od = (out_idx // (output_width * output_height)) % output_depth
        oc = out_idx // (output_width * output_height * output_depth)

        # Compute input indices
        id_start = od * stride_d - padding_d
        ih_start = oh * stride_h - padding_h
        iw_start = ow * stride_w - padding_w

        # Initialize accumulator
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for r in range(kernel_size_d):
            for s in range(kernel_size_h):
                for t in range(kernel_size_w):
                    id_val = id_start + r
                    ih_val = ih_start + s
                    iw_val = iw_start + t

                    # Check bounds
                    id_mask = (id_val >= 0) & (id_val < input_depth)
                    ih_mask = (ih_val >= 0) & (ih_val < input_height)
                    iw_mask = (iw_val >= 0) & (iw_val < input_width)
                    mask_3d = id_mask & ih_mask & iw_mask

                    # Load input tile
                    input_offset = input_ptr + \
                        (id_val * input_stride_d + ih_val * input_stride_h + iw_val * 1) * input_stride_c
                    input_mask = mask_3d[None, :] & (tl.arange(0, in_channels)[None, :] < in_channels)
                    input_vals = tl.load(input_offset + tl.arange(0, in_channels), mask=input_mask, other=0.0)

                    # Update accumulator
                    weight_rst = tl.load(weight_ptr +
                                         (oc_block_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) * weight_stride_k +
                                         r * weight_stride_s + s * weight_stride_t + t,
                                         mask=(tl.arange(0, BLOCK_SIZE_M) < out_channels - oc_block_idx * BLOCK_SIZE_M),
                                         other=0.0)
                    acc += weight_rst[:, None] * input_vals[None, :] @ weight.T

        # Convert to float32 and apply bias
        acc = acc.to(tl.float32)
        acc += bias[:, None]

        # Store original values before residual operations
        orig_vals = acc

        # First residual add: x = x + original_x
        acc += orig_vals

        # Multiply: x = x * original_x
        acc *= orig_vals

        # Second residual add: x = x + original_x
        acc += orig_vals

        # Write back to output
        output_offset = output_block_ptr + \
            (oc * output_stride_c + od * output_stride_d + oh * output_stride_h + ow) * 1
        output_mask = mask_out & (oc < out_channels)
        tl.store(output_offset, acc, mask=output_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, (list, tuple)) else (stride,) * 3
        self.padding = padding if isinstance(padding, (list, tuple)) else (padding,) * 3
        self.output_padding = output_padding if isinstance(output_padding, (list, tuple)) else (output_padding,) * 3
        self.bias_shape = bias_shape

        # Initialize transposed convolution weights and bias
        k, r, s, t = self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], 1
        weight = torch.empty(in_channels, out_channels, k, r, s)
        nn.init.kaiming_uniform_(weight, a=5**0.5)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        B, C_in, D, H, W = x.shape
        k, r, s = self.kernel_size
        d_stride, h_stride, w_stride = self.stride
        d_pad, h_pad, w_pad = self.padding
        d_opad, h_opad, w_opad = self.output_padding

        # Compute output shape
        D_out = (D - 1) * d_stride - 2 * d_pad + k + d_opad
        H_out = (H - 1) * h_stride - 2 * h_pad + r + h_opad
        W_out = (W - 1) * w_stride - 2 * w_pad + s + w_opad

        # Output tensor
        output = torch.empty(B, self.out_channels, D_out, H_out, W_out, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        output_strides = output.stride()
        weight_strides = self.weight.stride()

        n_elements = output.numel()
        def grid(meta):
            return (triton.cdiv(self.out_channels, meta['BLOCK_SIZE_M']) * B,)

        # Launch fused kernel
        fused_conv_transpose3d_bias_res_mul_res_kernel[grid](
            x, self.weight, self.bias, output,
            C_in, self.out_channels, D, H, W,
            D_out, H_out, W_out,
            k, r, s,
            d_stride, h_stride, w_stride,
            d_pad, h_pad, w_pad,
            d_opad, h_opad, w_opad,
            input_strides[1], input_strides[2], input_strides[3],
            output_strides[1], output_strides[2], output_strides[3],
            weight_strides[1], weight_strides[0], weight_strides[2], weight_strides[3],
            n_elements, has_bias=True,
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32
        )

        return output
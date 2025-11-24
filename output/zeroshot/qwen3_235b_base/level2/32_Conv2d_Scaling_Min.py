import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_min_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr, scale_factor,
    batch_size, out_channels, out_height, out_width,
    in_channels, in_height, in_width, kernel_size,
    stride_h, stride_w, padding_h, padding_w,
    dilation_h, dilation_w,
    input_stride_b, input_stride_c, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_r, weight_stride_s,
    output_stride_b, output_stride_k, output_stride_h, output_stride_w,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_height * out_width)
    remaining = pid % (out_height * out_width)
    out_h = remaining // out_width
    out_w = remaining % out_width

    if batch_idx >= batch_size:
        return

    # Pointers to output feature map
    output_offset_base = batch_idx * output_stride_b + out_h * output_stride_h + out_w * output_stride_w
    output_ptrs = out_ptr + output_offset_base + tl.arange(0, BLOCK_SIZE)

    # Initialize accumulator for output channels
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Spatial dimensions of kernel
    Kh = kernel_size
    Kw = kernel_size
    c_offset = tl.arange(0, BLOCK_SIZE)

    # Loop over input channels and kernel
    for c in range(0, in_channels):
        for r in range(0, Kh):
            for s in range(0, Kw):
                h_im = out_h * stride_h - padding_h + r * dilation_h
                w_im = out_w * stride_w - padding_w + s * dilation_w

                mask_h = (h_im >= 0) & (h_im < in_height)
                mask_w = (w_im >= 0) & (w_im < in_width)
                mask_hw = mask_h & mask_w

                input_offset = batch_idx * input_stride_b + c * input_stride_c + h_im * input_stride_h + w_im * input_stride_w
                input_mask = mask_hw[None] & (c_offset < out_channels)
                x = tl.load(x_ptr + input_offset + c_offset * input_stride_c, mask=input_mask, other=0.0)

                weight_offset = c * weight_stride_c + r * weight_stride_r + s * weight_stride_s
                w = tl.load(weight_ptr + weight_offset + c_offset * weight_stride_k, mask=c_offset < out_channels, other=0.0)

                acc += x * w

    # Add bias
    bias_ptrs = bias_ptr + c_offset
    bias = tl.load(bias_ptrs, mask=c_offset < out_channels, other=0.0)
    acc += bias

    # Scale by scale_factor
    acc = acc * scale_factor

    # Store full output (before min reduction)
    tl.store(output_ptrs, acc, mask=c_offset < out_channels)

    # Now perform channel-wise min reduction in register
    # We want min across channel dim -> reduce BLOCK_SIZE values
    acc = tl.where(c_offset < out_channels, acc, float('inf'))
    min_val = tl.min(acc)

    # Only write to first channel (keepdim=True), others are unused
    if out_channels > 0:
        min_output_ptr = out_ptr + batch_idx * output_stride_b + 0 * output_stride_k + out_h * output_stride_h + out_w * output_stride_w
        tl.store(min_output_ptr, min_val)


def triton_conv_min(x, weight, bias, scale_factor, stride, padding, dilation):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # Output tensor: we return only 1 channel due to min(dim=1, keepdim=True)
    out = torch.empty((batch_size, 1, out_height, out_width), dtype=torch.float32, device=x.device)

    # Flatten grid: one block per (batch, out_h, out_w)
    n_tiles = batch_size * out_height * out_width
    BLOCK_SIZE = triton.next_power_of_2(out_channels)
    grid = lambda meta: (n_tiles,)

    _conv_min_kernel[grid](
        x, weight, bias, out, scale_factor,
        batch_size, out_channels, out_height, out_width,
        in_channels, in_height, in_width, kernel_h,
        stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1],
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_elements=out.numel(), BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for conv + scale + min.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Initialize convolution weights and bias
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        # Conv parameters
        self.stride = (1, 1)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.dilation = (1, 1)

    def forward(self, x):
        return triton_conv_min(
            x, self.weight, self.bias, self.scale_factor,
            self.stride, self.padding, self.dilation
        )
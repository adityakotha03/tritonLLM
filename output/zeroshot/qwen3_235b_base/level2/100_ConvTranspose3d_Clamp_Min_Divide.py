import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_clamp_div_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, out_channels, out_depth, out_height, out_width,
    in_channels, in_depth, in_height, in_width,
    kernel_size, stride, padding,
    min_value, divisor,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    weight_stride_k, weight_stride_c, weight_stride_z, weight_stride_y, weight_stride_x,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # 2D block ID
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    # Compute output spatial indices (d, h, w)
    out_d = pid_dhw // (out_height * out_width)
    out_hw = pid_dhw % (out_height * out_width)
    out_h = out_hw // out_width
    out_w = out_hw % out_width

    # Compute input tile bounds
    di_start = out_d * stride - padding
    hi_start = out_h * stride - padding
    wi_start = out_w * stride - padding

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over input channels and kernel space
    for ic in range(0, in_channels, BLOCK_SIZE_K):
        for kz in range(0, kernel_size):
            for ky in range(0, kernel_size):
                for kx in range(0, kernel_size):
                    # Input spatial coordinates
                    di = di_start + kz
                    hi = hi_start + ky
                    wi = wi_start + kx

                    # Check bounds
                    in_bounds = (di >= 0) & (di < in_depth) & (hi >= 0) & (hi < in_height) & (wi >= 0) & (wi < in_width)

                    # Load input tile (batch, ic:ic+BLOCK_SIZE_K, di, hi, wi)
                    offs_ic = tl.arange(0, BLOCK_SIZE_K)
                    input_mask = (offs_ic[:, None] < in_channels) & in_bounds
                    input_offs = pid_b * input_stride_b + \
                                 offs_ic * input_stride_c + \
                                 di * input_stride_d + \
                                 hi * input_stride_h + \
                                 wi * input_stride_w
                    input_vals = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

                    # Load weights (out_c, ic:ic+BLOCK_SIZE_K, kz, ky, kx)
                    offs_k = pid_c * weight_stride_k
                    weight_offs = offs_k + \
                                  offs_ic * weight_stride_c + \
                                  kz * weight_stride_z + \
                                  ky * weight_stride_y + \
                                  kx * weight_stride_x
                    weight_mask = (offs_ic < in_channels)
                    weight_vals = tl.load(weight_ptr + weight_offs, mask=weight_mask, other=0.0)

                    # Matrix multiplication
                    acc += tl.dot(weight_vals[None, :], input_vals, out_dtype=tl.float32)

    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_c)
        acc += bias

    # Clamp and divide
    acc = tl.maximum(acc, min_value)
    acc = acc / divisor

    # Store output
    offs_m = pid_b * output_stride_b + pid_c * output_stride_c + \
             out_d * output_stride_d + out_h * output_stride_h + out_w * output_stride_w
    output_mask = (tl.arange(0, BLOCK_SIZE_M) < batch_size) & (tl.arange(0, BLOCK_SIZE_N) < 1)
    tl.store(output_ptr + offs_m, acc, mask=output_mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for transposed 3D convolution,
    clamp, and division.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.min_value = min_value
        self.divisor = divisor

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # Output dimensions
        out_depth = (x.shape[2] - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_height = (x.shape[3] - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_width = (x.shape[4] - 1) * self.stride - 2 * self.padding + self.kernel_size

        # Output tensor
        out = torch.empty(x.shape[0], self.out_channels, out_depth, out_height, out_width, device=x.device, dtype=x.dtype)

        # Strides
        input_strides = x.stride()
        weight_strides = self.weight.stride()
        output_strides = out.stride()

        # Grid configuration
        grid = (x.shape[0], self.out_channels, out_depth * out_height * out_width)

        # Launch kernel
        conv_transpose3d_clamp_div_kernel[grid](
            x, self.weight, self.bias, out,
            x.shape[0], self.out_channels, out_depth, out_height, out_width,
            self.in_channels, x.shape[2], x.shape[3], x.shape[4],
            self.kernel_size, self.stride, self.padding,
            self.min_value, self.divisor,
            input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
            weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3], weight_strides[4],
            output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
        )

        return out
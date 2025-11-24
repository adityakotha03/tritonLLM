import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_scale_tanh_bias_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, scaling_factor_ptr,
    output_ptr, in_channels, out_channels, depth, height, width,
    kernel_size, stride, padding,
    input_stride_b, input_stride_c, input_stride_d, input_stride_h, input_stride_w,
    output_stride_b, output_stride_c, output_stride_d, output_stride_h, output_stride_w,
    weight_stride_cin, weight_stride_cout, weight_stride_d, weight_stride_h, weight_stride_w,
    n_elements, conv_out_elements_per_sample,
    BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (out_channels * (depth + 2 * padding - kernel_size) // stride + 1) // ((height + 2 * padding - kernel_size) // stride + 1) // ((width + 2 * padding - kernel_size) // stride + 1 + BLOCK_D - 1) * BLOCK_D
    residual = pid % ((out_channels * (depth + 2 * padding - kernel_size) // stride + 1) // ((height + 2 * padding - kernel_size) // stride + 1) // ((width + 2 * padding - kernel_size) // stride + 1 + BLOCK_D - 1) * BLOCK_D)
    out_c = residual // ((depth + 2 * padding - kernel_size) // stride + 1) // ((height + 2 * padding - kernel_size) // stride + 1) // ((width + 2 * padding - kernel_size) // stride + 1 + BLOCK_D - 1) * BLOCK_D
    out_d = (residual // ((height + 2 * padding - kernel_size) // stride + 1) // ((width + 2 * padding - kernel_size) // stride + 1)) % BLOCK_D
    out_h = (residual // ((width + 2 * padding - kernel_size) // stride + 1)) % BLOCK_H
    out_w = residual % BLOCK_W

    if out_d >= (depth + 2 * padding - kernel_size) // stride + 1 or out_h >= (height + 2 * padding - kernel_size) // stride + 1 or out_w >= (width + 2 * padding - kernel_size) // stride + 1:
        return

    # Initialize accumulator for convolution
    acc = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=tl.float32)

    for ic in range(0, in_channels):
        for kd in range(0, kernel_size):
            for kh in range(0, kernel_size):
                for kw in range(0, kernel_size):
                    # Compute input position with padding and stride
                    in_d = out_d * stride - padding + kd
                    in_h = out_h * stride - padding + kh
                    in_w = out_w * stride - padding + kw

                    # Bounds check
                    in_d_mask = (in_d >= 0) & (in_d < depth)
                    in_h_mask = (in_h >= 0) & (in_h < height)
                    in_w_mask = (in_w >= 0) & (in_w < width)

                    # Load input value (with padding zero)
                    input_offset = batch_idx * input_stride_b + ic * input_stride_c + in_d * input_stride_d + in_h * input_stride_h + in_w * input_stride_w
                    input_val = tl.load(input_ptr + input_offset, mask=in_d_mask & in_h_mask & in_w_mask, other=0.0)

                    # Load weight
                    weight_offset = ic * weight_stride_cin + out_c * weight_stride_cout + kd * weight_stride_d + kh * weight_stride_h + kw * weight_stride_w
                    weight_val = tl.load(weight_ptr + weight_offset)

                    # Multiply and accumulate
                    acc += input_val * weight_val

    # Add bias
    bias_val = tl.load(bias_ptr + out_c)
    acc += bias_val

    # Scale by scaling factor
    scaling_val = tl.load(scaling_factor_ptr + out_c)
    acc = acc * scaling_val

    # Apply tanh
    acc = tl.tanh(acc)

    # Multiply by bias (second bias)
    acc = acc * bias_val

    # Apply sigmoid
    acc = tl.sigmoid(acc)

    # Store output
    output_offset = batch_idx * output_stride_b + out_c * output_stride_c + out_d * output_stride_d + out_h * output_stride_h + out_w * output_stride_w
    tl.store(output_ptr + output_offset, acc)


def triton_fused_conv_scale_tanh_bias_sigmoid(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_bias: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 1
):
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda and scaling_factor.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    scaling_factor = scaling_factor.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, kernel_size, _, _ = weight.shape

    # Compute output spatial dimensions
    out_depth = (depth + 2 * padding - kernel_size) // stride + 1
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), device=x.device, dtype=x.dtype)

    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    fused_conv_scale_tanh_bias_sigmoid_kernel[grid](
        x, weight, conv_bias, scaling_factor,
        out,
        in_channels, out_channels, depth, height, width,
        kernel_size, stride, padding,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        weight.stride(1), weight.stride(0), weight.stride(2), weight.stride(3), weight.stride(4),
        n_elements, out_depth * out_height * out_width,
        BLOCK_SIZE=1024,
        BLOCK_D=4, BLOCK_H=4, BLOCK_W=4
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model using a fused Triton kernel for 3D convolution, scaling, tanh, bias multiplication, and sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2

        # Conv3d parameters
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Init
        nn.init.kaiming_uniform_(self.weight, nonlinearity='tanh')
        nn.init.zeros_(self.conv_bias)

    def forward(self, x):
        return triton_fused_conv_scale_tanh_bias_sigmoid(
            x, self.weight, self.conv_bias, self.scaling_factor, self.bias,
            stride=self.stride, padding=self.padding
        )
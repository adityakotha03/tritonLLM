import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_transpose_gelu_min_mul_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, input_height, input_width,
    output_height, output_width, kernel_size, stride,
    pad, dilation, add_value, multiply_value,
    input_stride, output_stride,
    weight_stride_c, weight_stride_kh, weight_stride_kw,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Calculate output spatial dimensions
    total_output_elements = out_channels * output_height * output_width
    output_hw = output_height * output_width
    oh = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % output_height
    ow = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) // output_height
    oc = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Bounds for valid channels
    mask_c = oc < out_channels
    mask_hw = (oh < output_height) & (ow < output_width)

    # Calculate input region for transposed convolution
    ih_start = oh * stride - pad
    iw_start = ow * stride - pad

    # Base offsets in output
    output_offset = pid_batch * out_channels * output_hw + oc * output_hw + oh * output_width + ow
    output_mask = mask_c[:, None] & mask_hw[None, :]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    # Loop over input channels and kernel dimensions
    for ic in range(0, in_channels):
        for kh in range(0, kernel_size):
            for kw in range(0, kernel_size):
                # Input coordinates
                ih = ih_start + kh * dilation
                iw = iw_start + kw * dilation
                mask_in = (ih >= 0) & (ih < input_height) & (iw >= 0) & (iw < input_width)
                input_offset = pid_batch * in_channels * input_height * input_width + \
                               ic * input_height * input_width + ih * input_width + iw
                input_val = tl.load(input_ptr + input_offset, mask=mask_in & output_mask, other=0.0)

                # Weight offset
                weight_offset = oc * weight_stride_c + kh * weight_stride_kh + kw * weight_stride_kw
                weight_val = tl.load(weight_ptr + weight_offset, mask=mask_c[:, None], other=0.0)

                # Accumulate
                acc += weight_val * input_val

    # Add bias
    bias_val = tl.load(bias_ptr + oc, mask=mask_c, other=0.0)
    acc += bias_val[:, None]

    # Add constant value
    acc += add_value

    # Apply min(x, 0)
    acc = tl.where(acc < 0, acc, 0.0)

    # Apply GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using approximate formula for better performance
    x = acc
    x_cubed = x * x * x
    inner = 0.044715 * x_cubed + x
    tanh_inner = tl.tanh(0.79788456 * inner)
    gelu_out = 0.5 * x * (1.0 + tanh_inner)

    # Multiply by constant
    result = gelu_out * multiply_value

    # Store result
    tl.store(output_ptr + output_offset, result, mask=output_mask)


class ModelNew(nn.Module):
    """
    Optimized model with fused transposed convolution, add, min, GELU, and multiply using Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.pad = kernel_size // 2
        self.dilation = 1

        # Initialize transposed convolution weights and bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Calculate output shape
        output_height = (x.shape[2] - 1) * self.stride + self.kernel_size - 2 * self.pad
        output_width = (x.shape[3] - 1) * self.stride + self.kernel_size - 2 * self.pad

        # Output tensor
        output = torch.empty(x.shape[0], self.out_channels, output_height, output_width, device=x.device, dtype=x.dtype)

        # Launch kernel
        def grid(meta):
            batch_size = x.shape[0]
            total_m_blocks = triton.cdiv(output_height * output_width, meta['BLOCK_SIZE_M'])
            total_n_blocks = triton.cdiv(self.out_channels, meta['BLOCK_SIZE_N'])
            return (batch_size, total_m_blocks, total_n_blocks)

        # Use autotuning for optimal block sizes
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
                triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
            ],
            key=['in_channels', 'out_channels', 'input_height', 'input_width', 'output_height', 'output_width'],
        )
        @triton.jit
        def _kernel_caller(
            input_ptr, weight_ptr, bias_ptr, output_ptr,
            in_channels, out_channels, input_height, input_width,
            output_height, output_width, kernel_size, stride,
            pad, dilation, add_value, multiply_value,
            input_stride, output_stride,
            weight_stride_c, weight_stride_kh, weight_stride_kw,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
        ):
            fused_conv_transpose_gelu_min_mul_kernel(
                input_ptr, weight_ptr, bias_ptr, output_ptr,
                in_channels, out_channels, input_height, input_width,
                output_height, output_width, kernel_size, stride,
                pad, dilation, add_value, multiply_value,
                input_stride, output_stride,
                weight_stride_c, weight_stride_kh, weight_stride_kw,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )

        _kernel_caller[grid](
            x, self.weight, self.bias, output,
            self.in_channels, self.out_channels, x.shape[2], x.shape[3],
            output_height, output_width, self.kernel_size, self.stride,
            self.pad, self.dilation, self.add_value, self.multiply_value,
            x.stride(1), output.stride(1),
            self.out_channels, self.kernel_size, self.kernel_size
        )

        return output
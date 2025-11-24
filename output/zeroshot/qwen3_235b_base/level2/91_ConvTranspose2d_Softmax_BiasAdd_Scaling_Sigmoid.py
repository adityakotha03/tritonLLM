import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_channels, height, width,
    stride_batch, stride_channel, stride_height,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hw_idx = tl.program_id(1)

    # Pointers to batch and spatial location
    offset_ch = tl.arange(0, BLOCK_SIZE)
    mask_ch = offset_ch < n_channels

    input_offsets = (batch_idx * stride_batch + hw_idx * stride_height + offset_ch * stride_channel)
    x = tl.load(input_ptr + input_offsets, mask=mask_ch, other=-float('inf'))

    # Online softmax (subtract max for numerical stability)
    x_max = tl.max(x, 0)
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, 0)
    softmax_output = exp_x / sum_exp

    tl.store(output_ptr + input_offsets, softmax_output, mask=mask_ch)


@triton.jit
def bias_sigmoid_kernel(
    input_ptr, bias_ptr, output_ptr,
    n_channels, height, width,
    stride_batch, stride_channel, stride_height,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    hw_idx = tl.program_id(1)

    offset_ch = tl.arange(0, BLOCK_SIZE)
    mask_ch = offset_ch < n_channels

    input_offsets = (batch_idx * stride_batch + hw_idx * stride_height + offset_ch * stride_channel)
    bias_offsets = offset_ch

    x = tl.load(input_ptr + input_offsets, mask=mask_ch)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask_ch)
    x = x + bias
    x = x * scaling_factor
    x = tl.sigmoid(x)

    tl.store(output_ptr + input_offsets, x, mask=mask_ch)


def triton_softmax_bias_sigmoid(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    bias = bias.contiguous()

    n, c, h, w = x.shape
    total_elements = n * h * w

    # Use block size as power of 2 covering all channels
    BLOCK_SIZE = triton.next_power_of_2(c)

    # Grid: one block per (batch, spatial) location
    grid = (n, h * w)

    # Allocate output
    out = torch.empty_like(x)

    # Launch softmax
    softmax_kernel[grid](
        x, out,
        c, h, w,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Then bias + scaling + sigmoid
    bias_sigmoid_kernel[grid](
        out, bias, out,
        c, h, w,
        out.stride(0), out.stride(1), out.stride(2),
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    """
    Optimized model with fused softmax, bias, scaling, and sigmoid using Triton kernels.
    Transposed convolution remains as-is since it's already optimized in PyTorch (uses cuDNN).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_softmax_bias_sigmoid(x, self.bias, self.scaling_factor)
        return x
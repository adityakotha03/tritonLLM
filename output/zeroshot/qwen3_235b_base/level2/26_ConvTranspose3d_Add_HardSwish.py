import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardswish_kernel(x_ptr, out_ptr, bias_ptr, numel, channels, bias_stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load bias (per-channel broadcast: bias[c, 1, 1, 1])
    c_offsets = (offsets // (numel // channels)) * bias_stride
    bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0)

    # Add bias and input
    x = x + bias

    # Hardswish: x * relu6(x + 3) / 6
    x_plus_3 = x + 3.0
    zero = 0.0
    six = 6.0
    relu6 = tl.where(x_plus_3 <= zero, zero, tl.where(x_plus_3 >= six, six, x_plus_3))
    hardswish_out = x * relu6 / 6.0

    # Store result
    tl.store(out_ptr + offsets, hardswish_out, mask=mask)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_hardswish_add_bias(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    numel = out.numel()
    channels = bias.shape[0]
    bias_stride = 1  # bias is of shape (out_channels, 1, 1, 1, 1) -> stride 1 in flattened layout
    BLOCK_SIZE = 1024
    grid = lambda meta: ((numel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    hardswish_kernel[grid](x, out, bias, numel, channels, bias_stride, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels for fused add and HardSwish with bias.
    The transposed convolution is kept as-is (PyTorch uses cuDNN which is already optimized),
    but the subsequent add and HardSwish+bias are fused into a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = triton_add(x, add_input)
        x = triton_hardswish_add_bias(x, self.bias)
        return x
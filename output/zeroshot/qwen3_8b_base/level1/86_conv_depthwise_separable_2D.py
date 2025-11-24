import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to depthwise weights
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID
    pid = tl.program_id(0)
    # Get the current thread ID within the block
    tid = tl.program_id(1)
    # Compute the block index in the output
    out_idx = pid
    # Compute the input index
    in_idx = out_idx * stride * stride
    # Compute the offset within the block
    offset = tid
    # Compute the input and output dimensions
    in_h = in_channels
    in_w = in_channels
    out_h = out_channels
    out_w = out_channels
    # Compute the number of elements in the block
    num_elements = BLOCK_SIZE
    # Compute the block start in the input
    block_start = in_idx + offset
    # Compute the block end
    block_end = block_start + num_elements
    # Compute the output index
    out_idx = out_idx * out_h * out_w + offset
    # Load input values
    x = tl.load(input_ptr + block_start, mask=block_start < in_h * in_w, other=0.0)
    # Apply depthwise convolution
    y = tl.dot(x, weight_ptr, axis=0)
    # Store the result
    tl.store(output_ptr + out_idx, y, mask=out_idx < out_h * out_w)


def triton_depthwise_conv(x: torch.Tensor, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
    """
    This function wraps the Triton kernel call for depthwise convolution.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Compute output dimensions
    out_h = (x.shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_w = (x.shape[3] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    # Initialize output tensor
    output = torch.empty((x.shape[0], out_channels, out_h, out_w), dtype=x.dtype, device=x.device)
    # Prepare weights
    weight = torch.randn(in_channels, kernel_size * kernel_size, device=x.device)
    # Launch the Triton kernel
    grid = lambda meta: (x.shape[0],)
    depthwise_conv_kernel[grid](x, weight, output, x.shape[0], in_channels, out_channels, kernel_size, stride, padding, dilation, BLOCK_SIZE=128)
    return output


@triton.jit
def pointwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to pointwise weights
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the current program ID
    pid = tl.program_id(0)
    # Get the current thread ID within the block
    tid = tl.program_id(1)
    # Compute the block index in the output
    out_idx = pid
    # Compute the input index
    in_idx = out_idx
    # Compute the offset within the block
    offset = tid
    # Compute the input and output dimensions
    in_h = in_channels
    in_w = in_channels
    out_h = out_channels
    out_w = out_channels
    # Compute the number of elements in the block
    num_elements = BLOCK_SIZE
    # Compute the block start in the input
    block_start = in_idx + offset
    # Compute the block end
    block_end = block_start + num_elements
    # Compute the output index
    out_idx = out_idx * out_h * out_w + offset
    # Load input values
    x = tl.load(input_ptr + block_start, mask=block_start < in_h * in_w, other=0.0)
    # Apply pointwise convolution
    y = tl.dot(x, weight_ptr, axis=0)
    # Store the result
    tl.store(output_ptr + out_idx, y, mask=out_idx < out_h * out_w)


def triton_pointwise_conv(x: torch.Tensor, in_channels: int, out_channels: int):
    """
    This function wraps the Triton kernel call for pointwise convolution.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Initialize output tensor
    output = torch.empty((x.shape[0], out_channels, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
    # Prepare weights
    weight = torch.randn(in_channels, out_channels, device=x.device)
    # Launch the Triton kernel
    grid = lambda meta: (x.shape[0],)
    pointwise_conv_kernel[grid](x, weight, output, x.shape[0], in_channels, out_channels, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise convolution with Triton kernel
        x = triton_depthwise_conv(x, self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding, self.dilation)
        # Pointwise convolution with Triton kernel
        x = triton_pointwise_conv(x, self.in_channels, self.out_channels)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    kernel_size,  # Kernel size
    stride,  # Stride
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (out_channels // BLOCK_SIZE)
    block_idx = pid % (out_channels // BLOCK_SIZE)

    # Compute the offset in the output
    out_offset = block_id * out_channels * in_channels * (kernel_size * kernel_size) + block_idx * BLOCK_SIZE

    # Compute the input offset
    in_offset = block_id * in_channels * (kernel_size * kernel_size) + block_idx * BLOCK_SIZE

    # Iterate over the output channels
    for out_ch in range(BLOCK_SIZE):
        # Compute the output position
        out_pos = out_offset + out_ch

        # Compute the input position
        in_pos = in_offset + out_ch

        # Load input data
        input_val = tl.load(input_ptr + in_pos, mask=in_pos < in_channels * (kernel_size * kernel_size), other=0.0)

        # Compute the output value using the weight
        output_val = input_val * tl.load(weight_ptr + out_pos, mask=out_pos < out_channels, other=0.0)

        # Store the result
        tl.store(output_ptr + out_pos, output_val, mask=out_pos < out_channels)


def triton_depthwise_conv(input: torch.Tensor, weight: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int):
    """
    Triton implementation of depthwise convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, input.size(2) // stride, input.size(3) // stride), device=input.device)

    # Determine the number of blocks needed
    num_blocks = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    depthwise_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, BLOCK_SIZE=128)
    return output


@triton.jit
def expand_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (out_channels // BLOCK_SIZE)
    block_idx = pid % (out_channels // BLOCK_SIZE)

    # Compute the offset in the output
    out_offset = block_id * out_channels * in_channels + block_idx * BLOCK_SIZE

    # Compute the input offset
    in_offset = block_id * in_channels + block_idx * BLOCK_SIZE

    # Iterate over the output channels
    for out_ch in range(BLOCK_SIZE):
        # Compute the output position
        out_pos = out_offset + out_ch

        # Compute the input position
        in_pos = in_offset + out_ch

        # Load input data
        input_val = tl.load(input_ptr + in_pos, mask=in_pos < in_channels, other=0.0)

        # Compute the output value using the weight
        output_val = input_val * tl.load(weight_ptr + out_pos, mask=out_pos < out_channels, other=0.0)

        # Store the result
        tl.store(output_ptr + out_pos, output_val, mask=out_pos < out_channels)


def triton_expand_conv(input: torch.Tensor, weight: torch.Tensor, batch_size: int, in_channels: int, out_channels: int):
    """
    Triton implementation of expand convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, input.size(2), input.size(3)), device=input.device)

    # Determine the number of blocks needed
    num_blocks = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    expand_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, BLOCK_SIZE=128)
    return output


@triton.jit
def project_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of data
    pid = tl.program_id(0)
    block_id = pid // (out_channels // BLOCK_SIZE)
    block_idx = pid % (out_channels // BLOCK_SIZE)

    # Compute the offset in the output
    out_offset = block_id * out_channels * in_channels + block_idx * BLOCK_SIZE

    # Compute the input offset
    in_offset = block_id * in_channels + block_idx * BLOCK_SIZE

    # Iterate over the output channels
    for out_ch in range(BLOCK_SIZE):
        # Compute the output position
        out_pos = out_offset + out_ch

        # Compute the input position
        in_pos = in_offset + out_ch

        # Load input data
        input_val = tl.load(input_ptr + in_pos, mask=in_pos < in_channels, other=0.0)

        # Compute the output value using the weight
        output_val = input_val * tl.load(weight_ptr + out_pos, mask=out_pos < out_channels, other=0.0)

        # Store the result
        tl.store(output_ptr + out_pos, output_val, mask=out_pos < out_channels)


def triton_project_conv(input: torch.Tensor, weight: torch.Tensor, batch_size: int, in_channels: int, out_channels: int):
    """
    Triton implementation of project convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty((batch_size, out_channels, input.size(2), input.size(3)), device=input.device)

    # Determine the number of blocks needed
    num_blocks = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = lambda meta: (num_blocks,)
    project_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation with Triton kernels.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            # Expand convolution weights
            self.expand_weight = nn.Parameter(torch.randn(hidden_dim, in_channels, 1, 1))
            # Expand convolution bias
            self.expand_bias = nn.Parameter(torch.randn(hidden_dim))
            
            # Depthwise convolution weights
            self.depthwise_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim, kernel_size, 1))
            # Depthwise convolution bias
            self.depthwise_bias = nn.Parameter(torch.randn(hidden_dim))
            
            # Project convolution weights
            self.project_weight = nn.Parameter(torch.randn(out_channels, hidden_dim, 1, 1))
            # Project convolution bias
            self.project_bias = nn.Parameter(torch.randn(out_channels))
        else:
            # Depthwise convolution weights
            self.depthwise_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim, kernel_size, 1))
            # Depthwise convolution bias
            self.depthwise_bias = nn.Parameter(torch.randn(hidden_dim))
            
            # Project convolution weights
            self.project_weight = nn.Parameter(torch.randn(out_channels, hidden_dim, 1, 1))
            # Project convolution bias
            self.project_bias = nn.Parameter(torch.randn(out_channels))
    
    def forward(self, x):
        """
        Forward pass of the MBConv block with Triton kernels.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        batch_size = x.size(0)
        in_channels = x.size(1)
        h = x.size(2)
        w = x.size(3)
        
        if hasattr(self, 'expand_weight'):
            # Expand convolution
            x = triton_expand_conv(x, self.expand_weight, batch_size, in_channels, self.expand_weight.size(0))
            x = x + self.expand_bias.view(1, -1, 1, 1)
        
        # Depthwise convolution
        x = triton_depthwise_conv(x, self.depthwise_weight, batch_size, in_channels, self.depthwise_weight.size(0), self.depthwise_weight.size(2), stride)
        x = x + self.depthwise_bias.view(1, -1, 1, 1)
        
        # Project convolution
        x = triton_project_conv(x, self.project_weight, batch_size, self.project_weight.size(1), self.project_weight.size(0))
        x = x + self.project_bias.view(1, -1, 1, 1)
        
        if self.use_residual:
            x += identity
        
        return x
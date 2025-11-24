import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_channels,  # Number of input channels
    output_channels,  # Number of output channels
    kernel_size,  # Kernel size (assumed square)
    stride,  # Stride
    padding,  # Padding
    BLOCK_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Compute the 2D position in the output
    pid = tl.program_id(0)
    # Compute the 2D position in the input
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the 2D position in the output
    out_y = offset // (output_channels // GROUPS)
    out_x = offset % (output_channels // GROUPS)
    # Compute the input position
    in_y = out_y * stride - padding
    in_x = out_x * stride - padding
    # Compute the input channel index
    in_c = tl.arange(0, input_channels // GROUPS)
    # Compute the weight index
    weight_c = tl.arange(0, output_channels // GROUPS)
    # Compute the input offset
    in_offset = in_y * input_channels * width + in_x * input_channels + in_c
    # Compute the weight offset
    weight_offset = weight_c * input_channels * kernel_size * kernel_size + in_c * kernel_size * kernel_size
    # Load input values
    input_val = tl.load(input_ptr + in_offset, mask=offset < input_channels * height * width, other=0.0)
    # Load weight values
    weight_val = tl.load(weight_ptr + weight_offset, mask=offset < output_channels * input_channels * kernel_size * kernel_size, other=0.0)
    # Compute the output value
    output_val = tl.dot(input_val, weight_val)
    # Store the output value
    tl.store(output_ptr + offset, output_val, mask=offset < output_channels * height * width)

def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    This function wraps the Triton kernel call for 2D convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input_channels, output_channels, kernel_size, stride, padding, BLOCK_SIZE=BLOCK_SIZE, GROUPS=GROUPS)
    return output

@triton.jit
def batchnorm_kernel(
    input_ptr,  # Pointer to input tensor
    gamma_ptr,  # Pointer to gamma tensor
    beta_ptr,  # Pointer to beta tensor
    mean_ptr,  # Pointer to mean tensor
    var_ptr,  # Pointer to variance tensor
    output_ptr,  # Pointer to output tensor
    num_channels,  # Number of channels
    num_elements,  # Number of elements in input
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 1D position in the tensor
    pid = tl.program_id(0)
    # Compute the 1D position in the tensor
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load input values
    input_val = tl.load(input_ptr + offset, mask=offset < num_elements, other=0.0)
    # Load gamma, beta, mean, var
    gamma = tl.load(gamma_ptr, mask=offset < num_channels, other=1.0)
    beta = tl.load(beta_ptr, mask=offset < num_channels, other=0.0)
    mean = tl.load(mean_ptr, mask=offset < num_channels, other=0.0)
    var = tl.load(var_ptr, mask=offset < num_channels, other=1.0)
    # Normalize
    input_val = (input_val - mean) / tl.sqrt(var + 1e-5)
    # Scale and shift
    output_val = input_val * gamma + beta
    # Store the output value
    tl.store(output_ptr + offset, output_val, mask=offset < num_elements)

def triton_batchnorm(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, mean: torch.Tensor, var: torch.Tensor):
    """
    This function wraps the Triton kernel call for batch normalization.
    """
    assert input.is_cuda and gamma.is_cuda and beta.is_cuda and mean.is_cuda and var.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    gamma = gamma.contiguous()
    beta = beta.contiguous()
    mean = mean.contiguous()
    var = var.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    batchnorm_kernel[grid](input, gamma, beta, mean, var, output, input.size(1), n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

@triton.jit
def channel_shuffle_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    channels,  # Number of channels
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the 2D position in the tensor
    pid = tl.program_id(0)
    # Compute the 2D position in the tensor
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Compute the 2D position in the tensor
    out_y = offset // (channels // groups)
    out_x = offset % (channels // groups)
    # Compute the input position
    in_y = out_x * groups
    in_x = out_y
    # Compute the input offset
    in_offset = in_y * channels + in_x
    # Load input values
    input_val = tl.load(input_ptr + in_offset, mask=offset < channels, other=0.0)
    # Store the output value
    tl.store(output_ptr + offset, input_val, mask=offset < channels)

def triton_channel_shuffle(input: torch.Tensor, groups: int):
    """
    This function wraps the Triton kernel call for channel shuffle.
    """
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)

    # Number of elements in the tensor
    n_elements = input.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    channel_shuffle_kernel[grid](input, output, input.size(1), groups, BLOCK_SIZE=BLOCK_SIZE)
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        Optimized ShuffleNet unit with custom Triton kernels.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ModelNew, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1_weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bn1_gamma = nn.Parameter(torch.randn(out_channels))
        self.bn1_beta = nn.Parameter(torch.randn(out_channels))
        self.bn1_mean = nn.Parameter(torch.randn(out_channels))
        self.bn1_var = nn.Parameter(torch.randn(out_channels))
        
        # Depthwise 3x3 convolution
        self.conv2_weight = nn.Parameter(torch.randn(out_channels, out_channels, 3, 3))
        self.bn2_gamma = nn.Parameter(torch.randn(out_channels))
        self.bn2_beta = nn.Parameter(torch.randn(out_channels))
        self.bn2_mean = nn.Parameter(torch.randn(out_channels))
        self.bn2_var = nn.Parameter(torch.randn(out_channels))
        
        # Second 1x1 group convolution
        self.conv3_weight = nn.Parameter(torch.randn(out_channels, mid_channels, 1, 1))
        self.bn3_gamma = nn.Parameter(torch.randn(out_channels))
        self.bn3_beta = nn.Parameter(torch.randn(out_channels))
        self.bn3_mean = nn.Parameter(torch.randn(out_channels))
        self.bn3_var = nn.Parameter(torch.randn(out_channels))
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass for ShuffleNet unit.

        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # First 1x1 group convolution
        out = triton_conv2d(x, self.conv1_weight, stride=1, padding=0)
        out = triton_batchnorm(out, self.bn1_gamma, self.bn1_beta, self.bn1_mean, self.bn1_var)
        out = F.relu(out)
        
        # Depthwise 3x3 convolution
        out = triton_conv2d(out, self.conv2_weight, stride=1, padding=1)
        out = triton_batchnorm(out, self.bn2_gamma, self.bn2_beta, self.bn2_mean, self.bn2_var)
        
        # Second 1x1 group convolution
        out = triton_conv2d(out, self.conv3_weight, stride=1, padding=0)
        out = triton_batchnorm(out, self.bn3_gamma, self.bn3_beta, self.bn3_mean, self.bn3_var)
        out = F.relu(out)
        
        # Shuffle operation
        out = triton_channel_shuffle(out, self.groups)
        
        # Shortcut connection
        out += self.shortcut(x)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x
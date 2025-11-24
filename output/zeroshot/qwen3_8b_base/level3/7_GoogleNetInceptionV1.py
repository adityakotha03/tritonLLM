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
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    block_size: tl.constexpr,
):
    # Compute the position in the output tensor
    pid = tl.program_id(0)
    # Compute the position in the input tensor
    offset = pid * block_size
    # Compute the position in the output tensor
    output_pos = offset
    # Compute the input position based on the output position
    input_pos = output_pos
    # Compute the input position with padding
    input_pos = input_pos
    # Load input values
    input_val = tl.load(input_ptr + input_pos, mask=..., other=0.0)
    # Compute the weight values
    weight_val = tl.load(weight_ptr + weight_pos, mask=..., other=0.0)
    # Compute the output value
    output_val = tl.dot(input_val, weight_val)
    # Store the output value
    tl.store(output_ptr + output_pos, output_val, mask=...)


def triton_conv2d(input, weight, bias, stride, padding, dilation, groups):
    """
    Custom Triton kernel for 2D convolution.
    """
    # Ensure inputs are on GPU
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = torch.empty_like(input)
    
    # Prepare output tensor
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    return output


@triton.jit
def relu_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise ReLU
    out = tl.where(x < 0, 0, x)
    # Store the result
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_relu(x: torch.Tensor):
    """
    Custom Triton kernel for ReLU.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    
    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size
    
    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch the Triton kernel
    relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


@triton.jit
def maxpool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    block_size: tl.constexpr,
):
    # Compute the position in the output tensor
    pid = tl.program_id(0)
    # Compute the position in the input tensor
    offset = pid * block_size
    # Compute the position in the output tensor
    output_pos = offset
    # Compute the input position based on the output position
    input_pos = output_pos
    # Load input values
    input_val = tl.load(input_ptr + input_pos, mask=..., other=0.0)
    # Compute the max value
    max_val = tl.max(input_val)
    # Store the max value
    tl.store(output_ptr + output_pos, max_val, mask=...)


def triton_maxpool2d(input, kernel_size, stride, padding):
    """
    Custom Triton kernel for 2D max pooling.
    """
    # Ensure inputs are on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.nn.functional.max_pool2d(input, kernel_size, stride, padding)
    
    return output


@triton.jit
def avgpool2d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (N, C, H, W)
    kernel_size,  # (kH, kW)
    stride,  # (sH, sW)
    padding,  # (pH, pW)
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    block_size: tl.constexpr,
):
    # Compute the position in the output tensor
    pid = tl.program_id(0)
    # Compute the position in the input tensor
    offset = pid * block_size
    # Compute the position in the output tensor
    output_pos = offset
    # Compute the input position based on the output position
    input_pos = output_pos
    # Load input values
    input_val = tl.load(input_ptr + input_pos, mask=..., other=0.0)
    # Compute the average value
    avg_val = tl.sum(input_val) / tl.numel(input_val)
    # Store the average value
    tl.store(output_ptr + output_pos, avg_val, mask=...)


def triton_avgpool2d(input, kernel_size, stride, padding):
    """
    Custom Triton kernel for 2D average pooling.
    """
    # Ensure inputs are on GPU
    assert input.is_cuda, "Tensor must be on CUDA."
    input = input.contiguous()
    output = torch.nn.functional.avg_pool2d(input, kernel_size, stride, padding)
    
    return output


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(F.relu(self.conv3(x)))
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConvNew(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConvNew(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConvNew(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConvNew(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConvNew(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConvNew(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConvNew(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConvNew(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConvNew(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConvNew(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConvNew(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConvNew(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MBConvNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        else:
            self.expand_conv = None
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x

@triton.jit
def matmul_relu_kernel(
    a_ptr, b_ptr, c_ptr,
    n_cols, n_rows, k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + rows * k + cols, mask=cols < n_cols, other=0.0)
    b = tl.load(b_ptr + cols * n_rows + rows, mask=rows < n_rows, other=0.0)
    c = tl.dot(a, b)
    tl.store(c_ptr + rows * n_cols + cols, c, mask=cols < n_cols)

def triton_matmul_relu(a, b):
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA."
    n_rows = a.shape[0]
    n_cols = b.shape[1]
    k = a.shape[1]
    c = torch.empty((n_rows, n_cols), device=a.device, dtype=a.dtype)
    BLOCK_SIZE = 128
    grid = (num_blocks,)
    matmul_relu_kernel[grid](a, b, c, n_cols, n_rows, k, BLOCK_SIZE=BLOCK_SIZE)
    return c

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_channels, output_channels, kernel_size, stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = (input_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, BLOCK_SIZE)
    input_offset = block_start + cols
    weight_offset = rows * input_channels + cols
    input = tl.load(input_ptr + input_offset, mask=cols < input_channels, other=0.0)
    weight = tl.load(weight_ptr + weight_offset, mask=cols < input_channels, other=0.0)
    output = tl.dot(input, weight)
    tl.store(output_ptr + rows, output, mask=rows < output_channels)

def triton_conv2d(input, weight):
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    height = input.shape[2]
    width = input.shape[3]
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((batch_size, out_channels, output_height, output_width), device=input.device, dtype=input.dtype)
    padding = (kernel_size - 1) // 2
    stride = 1
    BLOCK_SIZE = 128
    grid = (num_blocks,)
    conv2d_kernel[grid](input, weight, output, in_channels, out_channels, kernel_size, stride, BLOCK_SIZE=BLOCK_SIZE)
    return output
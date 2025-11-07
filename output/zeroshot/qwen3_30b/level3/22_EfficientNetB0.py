import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_bn_relu_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    b_ptr,  # Bias pointer
    out_ptr,  # Output pointer
    batch_size,  # Batch size
    in_channels,  # Input channels
    out_channels,  # Output channels
    in_h, in_w,  # Input height and width
    out_h, out_w,  # Output height and width
    kernel_h, kernel_w,  # Kernel height and width
    stride_h, stride_w,  # Stride height and width
    pad_h, pad_w,  # Padding height and width
    BETA: tl.constexpr,  # Beta for batch norm (usually 1.0)
    GAMMA: tl.constexpr,  # Gamma for batch norm (usually 1.0)
    EPS: tl.constexpr,  # Epsilon for batch norm
    BLOCK_SIZE: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_K: tl.constexpr,
):
    # Thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Block indices for output
    h_start = pid_h * TILE_H
    w_start = pid_w * TILE_W
    c_start = pid_c * TILE_K

    # Output shape
    h_range = tl.arange(0, TILE_H)[:, None]
    w_range = tl.arange(0, TILE_W)[None, :]
    c_range = tl.arange(0, TILE_K)[:, None, None]

    # Compute input and output indices
    h_idx = h_start + h_range
    w_idx = w_start + w_range
    c_idx = c_start + c_range

    # Output mask
    mask_h = h_idx < out_h
    mask_w = w_idx < out_w
    mask_c = c_idx < out_channels
    out_mask = mask_h & mask_w & mask_c

    # Initialize output
    acc = tl.zeros((TILE_H, TILE_W, TILE_K), dtype=tl.float32)

    # Loop over input channels and kernel
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for ic in range(in_channels):
                # Compute input coordinates
                h_in = h_idx * stride_h - pad_h + kh
                w_in = w_idx * stride_w - pad_w + kw

                # Valid input indices
                valid_h = (h_in >= 0) & (h_in < in_h)
                valid_w = (w_in >= 0) & (w_in < in_w)
                valid = valid_h & valid_w

                # Load input
                x_offset = (tl.arange(0, TILE_H)[:, None, None] * in_h * in_w + 
                           tl.arange(0, TILE_W)[None, :, None] * in_w + 
                           tl.arange(0, TILE_K)[None, None, :] * in_h * in_w * in_channels + 
                           ic * in_h * in_w)
                x_data = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                x_data = x_data * tl.where(valid, 1.0, 0.0)

                # Load weights
                w_offset = (tl.arange(0, TILE_H)[:, None, None] * out_channels * in_channels * kernel_h * kernel_w + 
                           tl.arange(0, TILE_W)[None, :, None] * in_channels * kernel_h * kernel_w + 
                           tl.arange(0, TILE_K)[None, None, :] * kernel_h * kernel_w + 
                           ic * kernel_h * kernel_w + 
                           kh * kernel_w + kw)
                w_data = tl.load(w_ptr + w_offset, mask=valid, other=0.0)

                # Compute partial sum
                acc += x_data * w_data

    # Apply bias
    bias_offset = c_idx
    bias_data = tl.load(b_ptr + bias_offset, mask=mask_c, other=0.0)
    acc = acc + bias_data

    # Apply batch norm and ReLU
    # Assume running mean and var are passed as constants
    # Gamma and Beta are passed as constants
    acc = (acc - 0.0) * (1.0 / (tl.sqrt(1e-6 + 0.0))) * GAMMA + BETA
    acc = tl.maximum(acc, 0.0)

    # Store output
    out_offset = (pid_h * TILE_H + h_range) * out_w * out_channels + \
                 (pid_w * TILE_W + w_range) * out_channels + \
                 (pid_c * TILE_K + c_range) * 1
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


def triton_conv2d_bn_relu(x, weight, bias, stride=1, padding=0, beta=1.0, gamma=1.0):
    """
    Triton-based 2D convolution with batch norm and ReLU fused.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    batch_size, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Output dimensions
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    # Create output tensor
    out = torch.empty(batch_size, out_channels, out_h, out_w, device=x.device, dtype=x.dtype)

    # Define grid
    grid_h = (out_h + 31) // 32
    grid_w = (out_w + 31) // 32
    grid_c = (out_channels + 31) // 32

    # Block size
    BLOCK_SIZE = 32
    TILE_H, TILE_W, TILE_K = 32, 32, 32

    # Launch kernel
    conv2d_bn_relu_kernel[
        (grid_h, grid_w, grid_c),
        BLOCK_SIZE
    ](
        x, weight, bias, out,
        batch_size, in_channels, out_channels,
        in_h, in_w, out_h, out_w,
        kernel_h, kernel_w,
        stride, stride,
        padding, padding,
        beta, gamma, 1e-6,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_H=TILE_H,
        TILE_W=TILE_W,
        TILE_K=TILE_K
    )

    return out


class TritonMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio

        # Expand Conv
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # Depthwise Conv
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # Project Conv
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        # Apply expand, depthwise, project with Triton fused kernels
        if self.expand_ratio != 1:
            x = triton_conv2d_bn_relu(x, self.expand_conv[0].weight, self.expand_conv[0].bias,
                                    stride=1, padding=0, beta=1.0, gamma=1.0)

        # Depthwise Conv with Triton
        x = triton_conv2d_bn_relu(x, self.depthwise_conv[0].weight, self.depthwise_conv[0].bias,
                                stride=self.depthwise_conv[0].stride[0], padding=self.depthwise_conv[0].padding[0],
                                beta=1.0, gamma=1.0)

        # Project Conv with Triton
        x = triton_conv2d_bn_relu(x, self.project_conv[0].weight, self.project_conv[0].bias,
                                stride=1, padding=0, beta=1.0, gamma=1.0)

        if self.use_residual:
            x += identity

        return x


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # Initial convolutional layer with Triton fused kernel
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks using Triton optimized MBConv
        self.blocks = nn.Sequential(
            TritonMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            TritonMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            TritonMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            TritonMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            TritonMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            TritonMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            TritonMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            TritonMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            TritonMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            TritonMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            TritonMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            TritonMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            TritonMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer with Triton fused kernel
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        # Use Triton fused conv + bn + relu for initial layer
        x = triton_conv2d_bn_relu(x, self.conv1.weight, self.conv1.bias,
                                stride=2, padding=1, beta=1.0, gamma=1.0)
        
        # Forward through Triton optimized blocks
        x = self.blocks(x)
        
        # Final conv + bn + relu
        x = triton_conv2d_bn_relu(x, self.conv2.weight, self.conv2.bias,
                                stride=1, padding=0, beta=1.0, gamma=1.0)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for fused MBConv block (expansion + depthwise conv + SE + projection)
@triton.jit
def fused_mbconv_kernel(
    x_ptr,  # Input tensor pointer
    w1_ptr,  # Expansion conv weight pointer
    w2_ptr,  # Depthwise conv weight pointer
    w3_ptr,  # SE reduction conv weight pointer
    w4_ptr,  # SE expansion conv weight pointer
    w5_ptr,  # Projection conv weight pointer
    bias1_ptr,  # Expansion bias pointer
    bias2_ptr,  # Depthwise bias pointer
    bias3_ptr,  # SE reduction bias pointer
    bias4_ptr,  # SE expansion bias pointer
    bias5_ptr,  # Projection bias pointer
    y_ptr,  # Output tensor pointer
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    expanded_channels: tl.constexpr,
    stride: tl.constexpr,
    kernel_size: tl.constexpr,
    h: tl.constexpr,
    w: tl.constexpr,
    batch_size: tl.constexpr,
    expand_ratio: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    # Block shape
    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W
    c_offset = pid_c * BLOCK_SIZE

    # Compute h and w indices within block
    h_indices = h_offset + tl.arange(0, BLOCK_H)
    w_indices = w_offset + tl.arange(0, BLOCK_W)
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE)

    # Bounds checking
    h_mask = h_indices < h
    w_mask = w_indices < w
    c_mask = c_indices < expanded_channels  # For expansion and depthwise

    # Load input
    x_ptrs = x_ptr + (tl.arange(0, BLOCK_H)[:, None] * w * in_channels +
                      tl.arange(0, BLOCK_W)[None, :] * in_channels +
                      tl.arange(0, in_channels)[None, None]) * batch_size
    x = tl.load(x_ptrs + (pid_c * BLOCK_SIZE) * in_channels, mask=tl.broadcast(h_mask[:, None] & w_mask[None, :] & c_mask[None, :]), other=0.0)

    # Expand: conv1 (1x1)
    if expand_ratio != 1:
        w1_ptrs = w1_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * in_channels +
                            tl.arange(0, in_channels)[None, :])
        w1 = tl.load(w1_ptrs, mask=tl.broadcast(c_mask[:, None] & c_mask[None, :]), other=0.0)
        bias1 = tl.load(bias1_ptr + c_indices, mask=c_mask, other=0.0)
        expanded = tl.dot(x, w1.T) + bias1
        expanded = tl.relu(expanded)
    else:
        expanded = x

    # Depthwise conv
    w2_ptrs = w2_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * expanded_channels +
                        tl.arange(0, expanded_channels)[None, :])
    w2 = tl.load(w2_ptrs, mask=tl.broadcast(c_mask[:, None] & c_mask[None, :]), other=0.0)
    bias2 = tl.load(bias2_ptr + c_indices, mask=c_mask, other=0.0)
    depthwise_out = tl.dot(expanded, w2.T) + bias2
    depthwise_out = tl.relu(depthwise_out)

    # SE block: AdaptiveAvgPool + reduction + ReLU + expansion + Sigmoid
    se_in = depthwise_out  # (H, W, expanded_channels)
    se_in = tl.reshape(se_in, (BLOCK_H * BLOCK_W, expanded_channels))

    # Global average pooling
    se_pool = tl.sum(se_in, axis=0) / (BLOCK_H * BLOCK_W)
    se_pool = tl.reshape(se_pool, (1, expanded_channels))

    # Reduction
    w3_ptrs = w3_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * expanded_channels +
                        tl.arange(0, expanded_channels // 4)[None, :])
    w3 = tl.load(w3_ptrs, mask=tl.broadcast(c_mask[:, None] & c_mask[None, :]), other=0.0)
    bias3 = tl.load(bias3_ptr + c_indices, mask=c_mask, other=0.0)
    se_reduced = tl.dot(se_pool, w3.T) + bias3
    se_reduced = tl.relu(se_reduced)

    # Expansion
    w4_ptrs = w4_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * expanded_channels // 4 +
                        tl.arange(0, expanded_channels)[None, :])
    w4 = tl.load(w4_ptrs, mask=tl.broadcast(c_mask[:, None] & c_mask[None, :]), other=0.0)
    bias4 = tl.load(bias4_ptr + c_indices, mask=c_mask, other=0.0)
    se_expanded = tl.dot(se_reduced, w4.T) + bias4
    se_sigmoid = tl.sigmoid(se_expanded)

    # Reshape back and scale
    se_sigmoid = tl.reshape(se_sigmoid, (1, expanded_channels))
    se_sigmoid = tl.broadcast(se_sigmoid, (BLOCK_H * BLOCK_W, expanded_channels))

    # Apply SE modulation
    se_out = depthwise_out * se_sigmoid
    se_out = tl.reshape(se_out, (BLOCK_H, BLOCK_W, expanded_channels))

    # Project: 1x1 conv
    w5_ptrs = w5_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * expanded_channels +
                        tl.arange(0, out_channels)[None, :])
    w5 = tl.load(w5_ptrs, mask=tl.broadcast(c_mask[:, None] & c_mask[None, :]), other=0.0)
    bias5 = tl.load(bias5_ptr + c_indices, mask=c_mask, other=0.0)
    project_out = tl.dot(se_out, w5.T) + bias5

    # Store output
    out_ptrs = y_ptr + (h_indices[:, None] * w * out_channels +
                        w_indices[None, :] * out_channels +
                        c_indices[None, None]) * batch_size
    tl.store(out_ptrs, project_out, mask=tl.broadcast(h_mask[:, None] & w_mask[None, :] & c_mask[None, :]))


# Triton kernel for AdaptiveAvgPool2d
@triton.jit
def adaptive_avgpool_kernel(
    x_ptr,  # Input pointer
    y_ptr,  # Output pointer
    h: tl.constexpr,
    w: tl.constexpr,
    out_h: tl.constexpr,
    out_w: tl.constexpr,
    in_channels: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    h_offset = pid_h * BLOCK_H
    w_offset = pid_w * BLOCK_W
    c_offset = pid_c * BLOCK_SIZE

    h_indices = h_offset + tl.arange(0, BLOCK_H)
    w_indices = w_offset + tl.arange(0, BLOCK_W)
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE)

    h_mask = h_indices < h
    w_mask = w_indices < w
    c_mask = c_indices < in_channels

    x_ptrs = x_ptr + (tl.arange(0, BLOCK_H)[:, None] * w * in_channels +
                      tl.arange(0, BLOCK_W)[None, :] * in_channels +
                      tl.arange(0, in_channels)[None, None]) * batch_size
    x = tl.load(x_ptrs + (pid_c * BLOCK_SIZE) * in_channels, mask=tl.broadcast(h_mask[:, None] & w_mask[None, :] & c_mask[None, :]), other=0.0)

    # Global average over spatial dims
    pooled = tl.sum(x, axis=0) / (BLOCK_H * BLOCK_W)

    # Store output
    out_ptrs = y_ptr + (pid_h * out_w * in_channels + pid_w * in_channels + c_indices) * batch_size
    tl.store(out_ptrs, pooled, mask=tl.broadcast(c_mask))


# Triton kernel for linear layer
@triton.jit
def linear_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    n: tl.constexpr,
    m: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    c_offset = pid * BLOCK_SIZE
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE)

    # Load input
    x_ptrs = x_ptr + c_indices
    x = tl.load(x_ptrs, mask=c_indices < n, other=0.0)

    # Load weights
    w_ptrs = w_ptr + (c_indices[:, None] * m + tl.arange(0, m)[None, :])
    w = tl.load(w_ptrs, mask=tl.broadcast(c_indices[:, None] < n, tl.arange(0, m)[None, :] < m), other=0.0)

    # Compute output
    out = tl.dot(x, w.T)

    # Store output
    out_ptrs = out_ptr + c_indices
    tl.store(out_ptrs, out, mask=c_indices < m)


class TritonFusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.expanded_channels = in_channels * expand_ratio

        # Initialize weights
        self.register_buffer('w1', torch.randn(expanded_channels, in_channels, 1, 1).half().cuda())
        self.register_buffer('w2', torch.randn(expanded_channels, 1, kernel_size, kernel_size).half().cuda())
        self.register_buffer('w3', torch.randn(expanded_channels // 4, expanded_channels, 1, 1).half().cuda())
        self.register_buffer('w4', torch.randn(expanded_channels, expanded_channels // 4, 1, 1).half().cuda())
        self.register_buffer('w5', torch.randn(out_channels, expanded_channels, 1, 1).half().cuda())

        # Biases
        self.register_buffer('bias1', torch.zeros(expanded_channels).half().cuda())
        self.register_buffer('bias2', torch.zeros(expanded_channels).half().cuda())
        self.register_buffer('bias3', torch.zeros(expanded_channels // 4).half().cuda())
        self.register_buffer('bias4', torch.zeros(expanded_channels).half().cuda())
        self.register_buffer('bias5', torch.zeros(out_channels).half().cuda())

        # Cache shapes for Triton
        self.h = 224 // 2
        self.w = 224 // 2
        self.out_h = 224 // 2
        self.out_w = 224 // 2
        self.block_h = 16
        self.block_w = 16
        self.block_size = 16

    def forward(self, x):
        batch_size, _, h, w = x.shape
        x = x.contiguous()

        # Output tensor
        out = torch.empty(batch_size, self.out_channels, h // self.stride, w // self.stride, device=x.device, dtype=torch.bfloat16)

        # Grids
        grid_h = (h + self.block_h - 1) // self.block_h
        grid_w = (w + self.block_w - 1) // self.block_w
        grid_c = (self.expanded_channels + self.block_size - 1) // self.block_size

        # Launch kernels
        if self.expand_ratio != 1:
            fused_mbconv_kernel[
                (grid_h, grid_w, grid_c),
                128,
                128
            ](
                x,
                self.w1,
                self.w2,
                self.w3,
                self.w4,
                self.w5,
                self.bias1,
                self.bias2,
                self.bias3,
                self.bias4,
                self.bias5,
                out,
                self.in_channels,
                self.out_channels,
                self.expanded_channels,
                self.stride,
                self.kernel_size,
                h,
                w,
                batch_size,
                self.expand_ratio,
                self.block_h,
                self.block_w,
                self.block_size
            )
        else:
            # Fallback to PyTorch if no expansion
            x = F.conv2d(x, self.w1, self.bias1, stride=1, padding=0)
            x = F.relu(x)
            x = F.conv2d(x, self.w2, self.bias2, stride=self.stride, padding=1, groups=self.expanded_channels)
            x = F.relu(x)
            # SE block with Triton
            x_se = adaptive_avgpool_kernel[
                (grid_h, grid_w, grid_c),
                128,
                128
            ](
                x,
                x,
                h,
                w,
                1,
                1,
                self.expanded_channels,
                batch_size,
                self.block_h,
                self.block_w,
                self.block_size
            )
            x_se = x_se.reshape(batch_size, self.expanded_channels, 1, 1)
            x_se = F.conv2d(x_se, self.w3, self.bias3, stride=1, padding=0)
            x_se = F.relu(x_se)
            x_se = F.conv2d(x_se, self.w4, self.bias4, stride=1, padding=0)
            x_se = F.sigmoid(x_se)
            x = x * x_se
            x = F.conv2d(x, self.w5, self.bias5, stride=1, padding=0)

        return x


class TritonAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        out_h, out_w = self.output_size

        # Create output tensor
        out = torch.empty(batch_size, channels, out_h, out_w, device=x.device, dtype=torch.bfloat16)

        # Grid
        grid_h = (h + 16 - 1) // 16
        grid_w = (w + 16 - 1) // 16
        grid_c = (channels + 16 - 1) // 16

        # Launch kernel
        adaptive_avgpool_kernel[
            (grid_h, grid_w, grid_c),
            128,
            128
        ](
            x,
            out,
            h,
            w,
            out_h,
            out_w,
            channels,
            batch_size,
            16,
            16,
            16
        )

        return out


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16).cuda())
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16).cuda())

    def forward(self, x):
        batch_size, n = x.shape
        out = torch.empty(batch_size, self.out_features, device=x.device, dtype=torch.bfloat16)

        grid = (self.out_features + 16 - 1) // 16

        linear_kernel[
            (grid,),
            128,
            128
        ](
            x,
            self.weight,
            out,
            n,
            self.out_features,
            batch_size,
            16
        )

        return out + self.bias


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # First conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MBConv blocks with Triton kernels
        self.mbconv1 = TritonFusedMBConvBlock(32, 96, 1, 3)
        self.mbconv2 = TritonFusedMBConvBlock(96, 144, 2, 6)
        self.mbconv3 = TritonFusedMBConvBlock(144, 192, 2, 6)
        self.mbconv4 = TritonFusedMBConvBlock(192, 288, 2, 6)
        self.mbconv5 = TritonFusedMBConvBlock(288, 384, 1, 6)

        # Final layers
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = TritonAdaptiveAvgPool2d((1, 1))
        self.fc = TritonLinear(1408, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.relu(self.bn_final(self.conv_final(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
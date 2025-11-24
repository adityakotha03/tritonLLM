import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, channels, H, W)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, k, k)
    bias_ptr,  # pointer to bias tensor (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H_out, W_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_h: tl.constexpr,
    input_w: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Define block indices
    block_id = tl.program_id(0)
    block_h = block_id // (input_h // BLOCK_SIZE_H)
    block_w = block_id % (input_w // BLOCK_SIZE_W)
    
    # Compute the output dimensions
    h_start = block_h * BLOCK_SIZE_H
    h_end = min(h_start + BLOCK_SIZE_H, input_h)
    w_start = block_w * BLOCK_SIZE_W
    w_end = min(w_start + BLOCK_SIZE_W, input_w)
    
    # Define the output coordinates
    h_offsets = tl.arange(0, BLOCK_SIZE_H)
    w_offsets = tl.arange(0, BLOCK_SIZE_W)
    offsets_h = h_offsets + h_start
    offsets_w = w_offsets + w_start
    
    # Compute the output indices
    output_h = offsets_h // stride
    output_w = offsets_w // stride
    
    # Compute the input indices with padding
    input_h_idx = offsets_h - padding
    input_w_idx = offsets_w - padding
    
    # Mask for valid input access
    valid_h = (input_h_idx >= 0) & (input_h_idx < input_h)
    valid_w = (input_w_idx >= 0) & (input_w_idx < input_w)
    valid = valid_h & valid_w
    
    # Load input features
    input_features = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, in_channels), dtype=tl.float32)
    input_offsets = tl.arange(0, in_channels)
    input_batch = tl.arange(0, batch_size)
    
    # Load input data using valid indices
    for i in range(in_channels):
        input_idx = input_h_idx + input_w_idx * input_h
        input_data = tl.load(input_ptr + (input_batch[:, None] * input_h * input_w + input_idx), mask=valid, other=0.0)
        input_features = input_features + input_data[:, :, None] * tl.one_hot(input_offsets, in_channels)
    
    # Load weights
    weight_offsets = tl.arange(0, out_channels)
    weight_data = tl.load(weight_ptr + (weight_offsets[:, None] * in_channels * kernel_size * kernel_size + input_offsets[None, :]), mask=valid, other=0.0)
    
    # Compute output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, out_channels), dtype=tl.float32)
    for oc in range(out_channels):
        for ic in range(in_channels):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    h = kh + input_h_idx
                    w = kw + input_w_idx
                    if (h < input_h and w < input_w):
                        input_val = tl.load(input_ptr + (input_batch[:, None] * input_h * input_w + h * input_w + w), mask=(h >= 0) & (h < input_h) & (w >= 0) & (w < input_w), other=0.0)
                        weight_val = tl.load(weight_ptr + (oc * in_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw), mask=(kh < kernel_size) & (kw < kernel_size), other=0.0)
                        output += input_val * weight_val
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + oc, mask=(oc < out_channels), other=0.0)
            output[:, :, oc] += bias_val
    
    # Store output
    output_ptr_offset = (output_h[:, None] * output_w[None, :] + output_w[:, None])
    tl.store(output_ptr + (batch_size * out_channels * output_h.size(0) * output_w.size(0) + output_h * output_w + output_w), output, mask=valid)


@triton.jit
def batch_norm_kernel(
    x_ptr,  # input tensor (batch, channels, H, W)
    weight_ptr,  # weight tensor (channels)
    bias_ptr,  # bias tensor (channels)
    running_mean_ptr,  # running mean (channels)
    running_var_ptr,  # running variance (channels)
    eps: tl.constexpr,
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of channels
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, channels)
    
    # Load input data
    x = tl.load(x_ptr + (block_start + tl.arange(0, BLOCK_SIZE)), mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    
    # Load running statistics
    mean = tl.load(running_mean_ptr + block_start, mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    var = tl.load(running_var_ptr + block_start, mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    
    # Compute batch norm
    mean_val = tl.sum(x, axis=0) / (batch_size * H * W)
    var_val = tl.sum((x - mean_val) ** 2, axis=0) / (batch_size * H * W)
    
    # Normalize
    x_norm = (x - mean_val) / tl.sqrt(var_val + eps)
    
    # Apply scale and bias
    scale = tl.load(weight_ptr + block_start, mask=(tl.arange(0, BLOCK_SIZE) < channels), other=1.0)
    bias_val = tl.load(bias_ptr + block_start, mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    
    output = x_norm * scale + bias_val
    
    # Store output
    tl.store(x_ptr + block_start, output, mask=(tl.arange(0, BLOCK_SIZE) < channels))


@triton.jit
def relu_kernel(
    x_ptr,  # input tensor (batch, channels, H, W)
    out_ptr,  # output tensor (batch, channels, H, W)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, channels)
    
    # Load input
    x = tl.load(x_ptr + (block_start + tl.arange(0, BLOCK_SIZE)), mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    # Apply ReLU
    out = tl.maximum(x, 0.0)
    # Store
    tl.store(out_ptr + block_start, out, mask=(tl.arange(0, BLOCK_SIZE) < channels))


@triton.jit
def adaptive_avgpool2d_kernel(
    x_ptr,  # input tensor (batch, channels, H, W)
    out_ptr,  # output tensor (batch, channels, 1, 1)
    batch_size: tl.constexpr,
    channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, channels)
    
    # Load input
    x = tl.load(x_ptr + (block_start + tl.arange(0, BLOCK_SIZE)), mask=(tl.arange(0, BLOCK_SIZE) < channels), other=0.0)
    # Compute average over H and W
    h_avg = tl.sum(x, axis=1) / (H * W)
    w_avg = tl.sum(x, axis=2) / (H * W)
    # Store average
    tl.store(out_ptr + block_start, h_avg, mask=(tl.arange(0, BLOCK_SIZE) < channels))


@triton.jit
def linear_kernel(
    x_ptr,  # input tensor (batch, in_features)
    w_ptr,  # weight tensor (out_features, in_features)
    b_ptr,  # bias tensor (out_features)
    out_ptr,  # output tensor (batch, out_features)
    batch_size: tl.constexpr,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, out_features)
    
    # Load input
    x = tl.load(x_ptr + (block_start + tl.arange(0, BLOCK_SIZE)), mask=(tl.arange(0, BLOCK_SIZE) < in_features), other=0.0)
    # Load weights
    w = tl.load(w_ptr + (block_start + tl.arange(0, BLOCK_SIZE) * in_features), mask=(tl.arange(0, BLOCK_SIZE) < out_features), other=0.0)
    # Compute output
    out = tl.dot(x, w)
    if b_ptr is not None:
        b = tl.load(b_ptr + block_start, mask=(tl.arange(0, BLOCK_SIZE) < out_features), other=0.0)
        out += b
    # Store
    tl.store(out_ptr + block_start, out, mask=(tl.arange(0, BLOCK_SIZE) < out_features))


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: int = 1, padding: int = 1):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    
    batch_size, in_channels, H, W = x.shape
    out_channels, _, k, k = weight.shape
    output_h = (H + 2 * padding - k) // stride + 1
    output_w = (W + 2 * padding - k) // stride + 1
    
    output = torch.empty((batch_size, out_channels, output_h, output_w), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    grid = lambda meta: (
        (output_h + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (output_w + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )
    
    conv2d_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels, H, W, k, stride, padding,
        BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    return output


def triton_batch_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-5):
    assert x.is_cuda and weight.is_cuda and running_mean.is_cuda and running_var.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    running_mean = running_mean.contiguous()
    running_var = running_var.contiguous()
    
    batch_size, channels, H, W = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    batch_norm_kernel[grid](
        x, weight, bias, running_mean, running_var, eps,
        batch_size, channels, H, W, BLOCK_SIZE
    )
    return output


def triton_relu(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    relu_kernel[grid](
        x, output, x.shape[0], x.shape[1], x.shape[2], x.shape[3], BLOCK_SIZE
    )
    return output


def triton_adaptive_avgpool2d(x: torch.Tensor):
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    output = torch.empty((x.shape[0], x.shape[1], 1, 1), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((x.shape[1] + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    adaptive_avgpool2d_kernel[grid](
        x, output, x.shape[0], x.shape[1], x.shape[2], x.shape[3], BLOCK_SIZE
    )
    return output


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    
    batch_size, in_features = x.shape
    out_features = weight.shape[0]
    output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 128
    grid = lambda meta: ((out_features + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    linear_kernel[grid](
        x, weight, bias, output,
        batch_size, in_features, out_features, BLOCK_SIZE
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Define the EfficientNetB2 architecture components
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Define the MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the MBConv block.
        :return: A sequential container of layers forming the MBConv block.
        """
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        # Output phase
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
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
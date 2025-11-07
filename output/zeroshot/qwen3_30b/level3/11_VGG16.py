import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    kernel_size, output_channels, output_height, output_width,
    stride, padding, dilation, 
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    TILE_SIZE: tl.constexpr
):
    # Get the thread indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Compute the output position
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Create mask for valid indices
    mask_h = h < output_height
    mask_w = w < output_width
    mask_c = c < output_channels
    
    # Compute the input positions
    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding
    
    # Load output feature map
    out_ptr = output_ptr + (pid_h * output_width + pid_w) * output_channels + c
    out = tl.load(out_ptr, mask=mask_c & mask_h & mask_w, other=0.0)

    # Load kernel and bias
    kernel = tl.load(weight_ptr + (c[:, None, None] * kernel_size * kernel_size + 
                                   (h_start + tl.arange(0, kernel_size)[:, None, None]) * input_width + 
                                   (w_start + tl.arange(0, kernel_size)[None, :, None])) * input_channels + 
                                   tl.arange(0, input_channels)[None, None, :], 
                    mask=mask_c & (h_start + tl.arange(0, kernel_size)[:, None, None] >= 0) & 
                         (h_start + tl.arange(0, kernel_size)[:, None, None] < input_height) & 
                         (w_start + tl.arange(0, kernel_size)[None, :, None] >= 0) & 
                         (w_start + tl.arange(0, kernel_size)[None, :, None] < input_width), 
                    other=0.0)
    
    # Load input feature map
    input_data = tl.load(input_ptr + (pid_h * stride * input_width + pid_w * stride) * input_channels + 
                         (h_start + tl.arange(0, kernel_size)[:, None, None]) * input_width + 
                         (w_start + tl.arange(0, kernel_size)[None, :, None]) * input_channels + 
                         tl.arange(0, input_channels)[None, None, :], 
                    mask=mask_c & (h_start + tl.arange(0, kernel_size)[:, None, None] >= 0) & 
                         (h_start + tl.arange(0, kernel_size)[:, None, None] < input_height) & 
                         (w_start + tl.arange(0, kernel_size)[None, :, None] >= 0) & 
                         (w_start + tl.arange(0, kernel_size)[None, :, None] < input_width), 
                    other=0.0)
    
    # Compute convolution
    for i in range(kernel_size):
        for j in range(kernel_size):
            for c_in in range(input_channels):
                out += kernel[c_in, i, j] * input_data[c_in, i, j]
    
    # Add bias
    bias = tl.load(bias_ptr + c, mask=mask_c, other=0.0)
    out += bias
    
    # Store output
    tl.store(out_ptr, out, mask=mask_c & mask_h & mask_w)


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_height, output_width, kernel_size, stride, padding,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)

    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_h = h < output_height
    mask_w = w < output_width
    mask_c = c < input_channels

    h_start = pid_h * stride - padding
    w_start = pid_w * stride - padding

    output_ptr = output_ptr + (pid_h * output_width + pid_w) * input_channels + c
    out = tl.load(output_ptr, mask=mask_c & mask_h & mask_w, other=-float('inf'))

    for i in range(kernel_size):
        for j in range(kernel_size):
            h_idx = h_start + i
            w_idx = w_start + j
            mask = (h_idx >= 0) & (h_idx < input_height) & (w_idx >= 0) & (w_idx < input_width)
            input_data = tl.load(input_ptr + (h_idx * input_width + w_idx) * input_channels + c, 
                                 mask=mask & mask_c, other=-float('inf'))
            out = tl.maximum(out, input_data)

    tl.store(output_ptr, out, mask=mask_c & mask_h & mask_w)


@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, input_features, output_features,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)

    mask_m = m < batch_size
    mask_n = n < output_features
    mask_k = k < input_features

    input_ptr = input_ptr + m[:, None] * input_features + k[None, :]
    weight_ptr = weight_ptr + k[:, None] * output_features + n[None, :]
    output_ptr = output_ptr + m[:, None] * output_features + n[None, :]

    input = tl.load(input_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    weight = tl.load(weight_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    output = tl.dot(input, weight, out_dtype=tl.float32)
    
    bias = tl.load(bias_ptr + n, mask=mask_n, other=0.0)
    output += bias

    tl.store(output_ptr, output, mask=mask_m[:, None] & mask_n[None, :])


def triton_conv2d(input, weight, bias, stride=1, padding=0, dilation=1):
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size, input_channels, input_height, input_width = input.shape
    output_channels, _, kernel_size, _ = weight.shape
    
    output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    output = torch.empty(batch_size, output_channels, output_height, output_width, device=input.device, dtype=input.dtype)

    # Define kernel launch parameters
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16
    TILE_SIZE = 16

    grid_h = triton.cdiv(output_height, BLOCK_H)
    grid_w = triton.cdiv(output_width, BLOCK_W)
    grid_c = triton.cdiv(output_channels, BLOCK_C)

    conv2d_kernel[grid_h, grid_w, grid_c](
        input, weight, bias, output,
        batch_size, input_channels, input_height, input_width,
        kernel_size, output_channels, output_height, output_width,
        stride, padding, dilation,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C,
        TILE_SIZE=TILE_SIZE
    )
    
    return output


def triton_relu(input):
    assert input.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    output = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_max_pool2d(input, kernel_size, stride, padding=0):
    assert input.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    batch_size, channels, height, width = input.shape
    
    output_height = (height + 2 * padding - kernel_size) // stride + 1
    output_width = (width + 2 * padding - kernel_size) // stride + 1
    
    output = torch.empty(batch_size, channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 16
    
    grid_h = triton.cdiv(output_height, BLOCK_H)
    grid_w = triton.cdiv(output_width, BLOCK_W)
    grid_c = triton.cdiv(channels, BLOCK_C)
    
    max_pool2d_kernel[grid_h, grid_w, grid_c](
        input, output,
        batch_size, channels, height, width,
        output_height, output_width, kernel_size, stride, padding,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C
    )
    
    return output


def triton_linear(input, weight, bias):
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    batch_size, input_features = input.shape
    output_features, _ = weight.shape
    
    output = torch.empty(batch_size, output_features, device=input.device, dtype=input.dtype)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    grid_m = triton.cdiv(batch_size, BLOCK_M)
    grid_n = triton.cdiv(output_features, BLOCK_N)
    
    linear_kernel[grid_m, grid_n](
        input, weight, bias, output,
        batch_size, input_features, output_features,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        # Replace all conv, relu, and maxpool with Triton kernels
        x = self.features[0](x)
        x = triton_relu(x)
        x = self.features[2](x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.features[4](x)
        x = triton_relu(x)
        x = self.features[6](x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.features[8](x)
        x = triton_relu(x)
        x = self.features[10](x)
        x = triton_relu(x)
        x = self.features[12](x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.features[14](x)
        x = triton_relu(x)
        x = self.features[16](x)
        x = triton_relu(x)
        x = self.features[18](x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.features[20](x)
        x = triton_relu(x)
        x = self.features[22](x)
        x = triton_relu(x)
        x = self.features[24](x)
        x = triton_relu(x)
        x = triton_max_pool2d(x, kernel_size=2, stride=2)
        
        x = torch.flatten(x, 1)
        x = triton_linear(x, self.classifier[0].weight, self.classifier[0].bias)
        x = triton_relu(x)
        x = triton_linear(x, self.classifier[3].weight, self.classifier[3].bias)
        x = triton_relu(x)
        x = triton_linear(x, self.classifier[6].weight, self.classifier[6].bias)
        return x
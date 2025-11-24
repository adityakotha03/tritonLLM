import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Define block and grid dimensions
    pid = tl.program_id(0)
    block_start_h = pid // (input_shape[2] // BLOCK_SIZE)
    block_start_w = pid % (input_shape[2] // BLOCK_SIZE)
    
    # Compute the output position
    h_offset = block_start_h * BLOCK_SIZE
    w_offset = block_start_w * BLOCK_SIZE
    
    # Loop over output positions in the block
    for h in range(BLOCK_SIZE):
        for w in range(BLOCK_SIZE):
            # Compute input coordinates
            input_h = h_offset + h
            input_w = w_offset + w
            
            # Check bounds
            if input_h >= input_shape[2] or input_w >= input_shape[3]:
                continue
                
            # Compute output coordinates
            output_h = input_h // stride_h
            output_w = input_w // stride_w
            
            # Check output bounds
            if output_h >= input_shape[1] or output_w >= input_shape[3]:
                continue
                
            # Compute input and weight indices
            input_idx = input_h * input_shape[3] + input_w
            weight_idx = (output_h * weight_shape[2] + output_w) * weight_shape[3] + (output_h * weight_shape[2] + output_w)
            
            # Load input and weight
            input_val = tl.load(input_ptr + input_idx, mask=(input_h < input_shape[2]) & (input_w < input_shape[3]), other=0.0)
            weight_val = tl.load(weight_ptr + weight_idx, mask=(output_h < weight_shape[2]) & (output_w < weight_shape[3]), other=0.0)
            
            # Compute output
            output_val = input_val * weight_val
            
            # Accumulate output
            tl.store(output_ptr + output_h * input_shape[3] + output_w, output_val)


@triton.jit
def matmul_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_SIZE
    start_n = 0
    
    # Load matrix A
    a = tl.zeros((BLOCK_SIZE, k), dtype=tl.float16)
    for i in range(BLOCK_SIZE):
        a_i = tl.load(a_ptr + start_m + i, mask=(i < m), other=0.0)
        a[i, :] = a_i
    
    # Load matrix B
    b = tl.zeros((k, BLOCK_SIZE), dtype=tl.float16)
    for j in range(BLOCK_SIZE):
        b_j = tl.load(b_ptr + j, mask=(j < k), other=0.0)
        b[:, j] = b_j
    
    # Compute dot product
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float16)
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            out[i, j] = tl.dot(a[i, :], b[:, j])
    
    # Apply ReLU
    out = tl.where(out > 0, out, 0.0)
    
    # Store result
    tl.store(out_ptr + start_m, out, mask=(start_m < m))


@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    n_in: tl.constexpr,
    n_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Load input
    x = tl.zeros((BLOCK_SIZE, n_in), dtype=tl.float16)
    for i in range(BLOCK_SIZE):
        x_i = tl.load(x_ptr + block_start + i, mask=(i < n_in), other=0.0)
        x[i, :] = x_i
    
    # Load weights
    w = tl.zeros((n_in, BLOCK_SIZE), dtype=tl.float16)
    for j in range(BLOCK_SIZE):
        w_j = tl.load(w_ptr + j, mask=(j < n_in), other=0.0)
        w[:, j] = w_j
    
    # Compute output
    out = tl.dot(x, w)
    
    # Add bias
    bias = tl.load(b_ptr, mask=(0 < n_out), other=0.0)
    out = out + bias
    
    # Store output
    tl.store(out_ptr + block_start, out, mask=(block_start < n_out))


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride_h: int, stride_w: int, pad_h: int, pad_w: int):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    
    # Get dimensions
    batch_size, in_channels, h, w = x.shape
    out_channels, in_channels_k, kh, kw = weight.shape
    
    # Output dimensions
    out_h = (h + 2 * pad_h - kh) // stride_h + 1
    out_w = (w + 2 * pad_w - kw) // stride_w + 1
    
    # Allocate output
    out = torch.empty((batch_size, out_channels, out_h, out_w), device=x.device, dtype=torch.float16)
    
    # Define block size
    BLOCK_SIZE = 16
    
    # Grid
    grid = lambda meta: ((out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    conv2d_kernel[grid](x, weight, bias, out, (batch_size, in_channels, h, w), (out_channels, in_channels_k, kh, kw), stride_h, stride_w, pad_h, pad_w, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def triton_matmul_relu(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()
    
    m, k = x.shape
    n = y.shape[1]
    
    # Output shape
    out_shape = (m, n)
    
    # Allocate output
    out = torch.empty(out_shape, device=x.device, dtype=torch.float16)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid
    grid = lambda meta: ((m + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    matmul_relu_kernel[grid](x, y, out, m, n, k, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    assert x.is_cuda and w.is_cuda and b.is_cuda, "All tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()
    
    n_in, n_out = w.shape
    
    # Allocate output
    out = torch.empty((x.shape[0], n_out), device=x.device, dtype=torch.float16)
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Grid
    grid = lambda meta: ((n_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    
    # Launch kernel
    linear_kernel[grid](x, w, b, out, n_in, n_out, BLOCK_SIZE=BLOCK_SIZE)
    
    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.relu3(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Fifth conv block
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
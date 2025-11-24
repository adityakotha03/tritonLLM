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
    output_shape,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    # Get the program ID for the block
    pid = tl.program_id(0)
    # Compute the block's starting position in the output
    block_start_h = pid // (output_shape[2] // BLOCK_SIZE)
    block_start_w = pid % (output_shape[2] // BLOCK_SIZE)
    # Create the offset for each thread in the block
    offset_h = tl.arange(0, BLOCK_SIZE)
    offset_w = tl.arange(0, BLOCK_SIZE)
    # Compute the full offset in the output
    h_idx = block_start_h * BLOCK_SIZE + offset_h
    w_idx = block_start_w * BLOCK_SIZE + offset_w
    # Create the full index in the output
    h_idx = h_idx % output_shape[2]
    w_idx = w_idx % output_shape[3]
    # Compute the corresponding input indices (with padding and stride)
    h_in = h_idx * STRIDE
    w_in = w_idx * STRIDE
    # Apply padding to input indices
    h_in = h_in + PAD
    w_in = w_in + PAD
    # Check bounds for input
    mask_h = (h_in < input_shape[2]) & (h_in >= 0)
    mask_w = (w_in < input_shape[3]) & (w_in >= 0)
    mask = mask_h & mask_w
    # Load input features
    input_features = tl.load(input_ptr + (h_in * input_shape[3] + w_in) * input_shape[1], mask=mask, other=0.0)
    # Load weights
    weight = tl.load(weight_ptr + (offset_h * weight_shape[3] + offset_w) * weight_shape[1], mask=mask, other=0.0)
    # Compute output
    output = tl.dot(input_features, weight)
    # Add bias
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offset_h, mask=mask, other=0.0)
        output = output + bias
    # Store output
    tl.store(output_ptr + (h_idx * output_shape[3] + w_idx), output, mask=mask)


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def max_pool2d_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start_h = pid // (output_shape[2] // BLOCK_SIZE)
    block_start_w = pid % (output_shape[2] // BLOCK_SIZE)
    h_idx = block_start_h * BLOCK_SIZE
    w_idx = block_start_w * BLOCK_SIZE
    h_idx = h_idx % input_shape[2]
    w_idx = w_idx % input_shape[3]
    h_offset = tl.arange(0, BLOCK_SIZE)
    w_offset = tl.arange(0, BLOCK_SIZE)
    h_in = h_idx + h_offset
    w_in = w_idx + w_offset
    mask_h = (h_in < input_shape[2]) & (h_in >= 0)
    mask_w = (w_in < input_shape[3]) & (w_in >= 0)
    mask = mask_h & mask_w
    input_vals = tl.load(input_ptr + (h_in * input_shape[3] + w_in) * input_shape[1], mask=mask, other=-1e9)
    max_val = tl.max(input_vals, axis=0)
    tl.store(output_ptr + (h_idx * output_shape[3] + w_idx), max_val, mask=mask)


@triton.jit
def linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_shape[1]
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load weights
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Compute dot product
    y = tl.dot(x, w)
    # Add bias
    if bias_ptr is not None:
        b = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        y = y + b
    # Store output
    tl.store(output_ptr + offsets, y, mask=mask)


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
):
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    # Compute output shape
    b, c, h, w = input.shape
    out_c, out_h, out_w = weight.shape[0], (h + 2 * padding - dilation * (weight.shape[2] - 1) - 1) // stride + 1, (w + 2 * padding - dilation * (weight.shape[3] - 1) - 1) // stride + 1

    # Prepare output
    output = torch.empty((b, out_c, out_h, out_w), dtype=input.dtype, device=input.device)

    # Grid
    BLOCK_SIZE = 16
    grid = lambda meta: ((output.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        input_shape=input.shape,
        weight_shape=weight.shape,
        output_shape=output.shape,
        BLOCK_SIZE=BLOCK_SIZE,
        STRIDE=stride,
        PAD=padding,
    )
    return output


def triton_relu(
    input: torch.Tensor,
    output: torch.Tensor,
):
    assert input.is_cuda and output.is_cuda, "Inputs must be on CUDA"
    input = input.contiguous()
    output = output.contiguous()

    BLOCK_SIZE = 128
    grid = lambda meta: ((input.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    relu_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        n_elements=input.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def triton_maxpool2d(
    input: torch.Tensor,
    output: torch.Tensor,
):
    assert input.is_cuda and output.is_cuda, "Inputs must be on CUDA"
    input = input.contiguous()
    output = output.contiguous()

    b, c, h, w = input.shape
    out_h, out_w = (h + 1) // 2, (w + 1) // 2

    BLOCK_SIZE = 16
    grid = lambda meta: ((input.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    max_pool2d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_shape=input.shape,
        output_shape=(b, c, out_h, out_w),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def triton_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
):
    assert input.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    output = output.contiguous()

    b, in_features = input.shape
    out_features = weight.shape[1]

    BLOCK_SIZE = 128
    grid = lambda meta: ((input.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    linear_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        input_shape=input.shape,
        weight_shape=weight.shape,
        output_shape=(b, out_features),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        # VGG16 architecture with custom Triton kernels
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            triton_relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            triton_relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            triton_relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            triton_relu,
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            triton_relu,
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            triton_relu,
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
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
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
):
    # Compute the block indices
    block_id = tl.program_id(0)
    block_start_h = block_id // (HEIGHT // BLOCK_SIZE) * BLOCK_SIZE
    block_start_w = (block_id % (HEIGHT // BLOCK_SIZE)) * BLOCK_SIZE
    block_end_h = block_start_h + BLOCK_SIZE
    block_end_w = block_start_w + BLOCK_SIZE

    # Compute the output position
    output_h = tl.arange(0, BLOCK_SIZE)
    output_w = tl.arange(0, BLOCK_SIZE)
    output_idx = output_h[:, None] * WIDTH + output_w[None, :]
    output_idx = output_idx % (HEIGHT * WIDTH)

    # Compute the input indices
    input_h = output_h + PAD_H
    input_w = output_w + PAD_W
    input_idx = input_h[:, None] * WIDTH + input_w[None, :]
    input_idx = input_idx % (HEIGHT * WIDTH)

    # Create masks to avoid out-of-bounds access
    valid_h = (input_h < HEIGHT) & (input_h >= 0)
    valid_w = (input_w < WIDTH) & (input_w >= 0)
    valid_mask = valid_h & valid_w

    # Load input features
    input_features = tl.load(input_ptr + input_idx, mask=valid_mask, other=0.0)
    # Load weights
    weight = tl.load(weight_ptr + tl.arange(0, CHANNELS)[:, None] * (weight_shape[1] * weight_shape[2]) + tl.arange(0, weight_shape[1] * weight_shape[2])[None, :], mask=valid_mask, other=0.0)

    # Compute output
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(CHANNELS):
        weight_slice = weight[i]
        input_slice = input_features
        output += tl.dot(input_slice, weight_slice)

    # Store output
    tl.store(output_ptr + output_idx, output, mask=valid_mask)


@triton.jit
def conv1x1_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_shape,
    weight_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of output
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape[1] * output_shape[2] * output_shape[3]

    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Load weight
    weight_val = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    # Compute output
    output_val = input_val @ weight_val
    # Store output
    tl.store(output_ptr + offsets, output_val, mask=mask)


@triton.jit
def maxpool_kernel(
    input_ptr,
    output_ptr,
    input_shape,
    output_shape,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape[1] * output_shape[2] * output_shape[3]

    # Load input
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Max pool over 3x3 window
    max_val = tl.max(input_val)
    # Store output
    tl.store(output_ptr + offsets, max_val, mask=mask)


def triton_conv2d(
    input_tensor,
    weight_tensor,
    bias_tensor=None,
    stride_h=1,
    stride_w=1,
    pad_h=0,
    pad_w=0,
    output_shape=None,
    BLOCK_SIZE=128
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    if bias_tensor is not None:
        bias_tensor = bias_tensor.contiguous()

    # Prepare output
    output_shape = input_tensor.shape[:-2] + (weight_tensor.shape[0], input_tensor.shape[-2], input_tensor.shape[-1])
    output_tensor = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)

    # Grid size
    grid = lambda meta: ((output_tensor.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv2d_kernel[grid](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr() if bias_tensor is not None else None,
        output_tensor.data_ptr(),
        input_tensor.shape,
        weight_tensor.shape,
        output_tensor.shape,
        BLOCK_SIZE=BLOCK_SIZE,
        STRIDE_H=stride_h,
        STRIDE_W=stride_w,
        PAD_H=pad_h,
        PAD_W=pad_w,
        CHANNELS=weight_tensor.shape[1],
        HEIGHT=input_tensor.shape[-2],
        WIDTH=input_tensor.shape[-1],
    )
    return output_tensor


def triton_conv1x1(
    input_tensor,
    weight_tensor,
    output_shape=None,
    BLOCK_SIZE=128
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    output_shape = input_tensor.shape[:-1] + (weight_tensor.shape[0],)
    output_tensor = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda meta: ((output_tensor.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv1x1_kernel[grid](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        output_tensor.data_ptr(),
        input_tensor.shape,
        weight_tensor.shape,
        output_tensor.shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor


def triton_maxpool(
    input_tensor,
    output_shape=None,
    BLOCK_SIZE=128
):
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    input_tensor = input_tensor.contiguous()
    output_shape = input_tensor.shape[:-2] + (input_tensor.shape[-2], input_tensor.shape[-1])
    output_tensor = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda meta: ((output_tensor.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    maxpool_kernel[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        input_tensor.shape,
        output_tensor.shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
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
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)
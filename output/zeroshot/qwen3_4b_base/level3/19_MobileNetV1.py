import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (B, C_in, H, W)
    weight_ptr,  # pointer to weight tensor (O, C_in, 3, 3)
    bias_ptr,  # pointer to bias tensor (O,)
    output_ptr,  # pointer to output tensor (B, C_out, H_out, W_out)
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the block indices
    block_id = tl.program_id(0)
    block_start_h = block_id // (input_height // BLOCK_SIZE) * BLOCK_SIZE
    block_start_w = (block_id % (input_height // BLOCK_SIZE)) * BLOCK_SIZE
    block_h, block_w = block_start_h, block_start_w

    # Create offsets for output
    h_offset = tl.arange(0, BLOCK_SIZE)
    w_offset = tl.arange(0, BLOCK_SIZE)
    offsets_h = h_offset + block_h
    offsets_w = w_offset + block_w

    # Compute valid output region
    mask_h = offsets_h < input_height
    mask_w = offsets_w < input_width
    mask = mask_h & mask_w

    # Load input features (batch, channel, height, width)
    input_h = input_height
    input_w = input_width
    input_idx = tl.arange(0, input_channels)
    output_idx = tl.arange(0, output_channels)

    # Load input patch (batch, C_in, H, W)
    input_batch = tl.arange(0, batch_size)
    input_offset = input_batch[:, None] * input_channels * input_height * input_width + \
                   input_idx[None, :] * input_height * input_width + \
                   offsets_h[:, None] * input_width + offsets_w[:, None]

    # Load input (using 2D indexing via flat offset)
    input_values = tl.load(input_ptr + input_offset, mask=mask, other=0.0)

    # Load weights (O, C_in, 3, 3)
    weight_values = tl.load(weight_ptr + (output_idx[:, None] * input_channels * 9 + input_idx[None, :] * 9 + tl.arange(0, 9)[None, :]), mask=mask, other=0.0)

    # Compute convolution via 3x3 kernel
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Apply kernel with padding
            h_idx = offsets_h + i - padding
            w_idx = offsets_w + j - padding
            h_mask = h_idx >= 0 and h_idx < input_height
            w_mask = w_idx >= 0 and w_idx < input_width
            valid = h_mask & w_mask
            h_offset_val = h_idx - padding
            w_offset_val = w_idx - padding
            # Use 2D indexing for input
            input_flat_idx = (input_batch * input_channels * input_height * input_width +
                              input_idx * input_height * input_width +
                              h_offset_val * input_width + w_offset_val)
            input_val = tl.load(input_ptr + input_flat_idx, mask=valid, other=0.0)
            weight_val = tl.load(weight_ptr + (output_idx * input_channels * 9 + input_idx * 9 + i * 3 + j), mask=valid, other=0.0)
            output += input_val * weight_val

    # Add bias
    bias = tl.load(bias_ptr + output_idx, mask=output_idx < output_channels, other=0.0)
    output = output + bias[None, :]  # Broadcast bias

    # Store output
    output_idx_flat = output_idx[:, None] * BLOCK_SIZE * BLOCK_SIZE + offsets_h[:, None] * BLOCK_SIZE + offsets_w
    tl.store(output_ptr + output_idx_flat, output, mask=mask)


@triton.jit
def conv_bn_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a region of output
    block_id = tl.program_id(0)
    block_start_h = block_id // (input_height // BLOCK_SIZE) * BLOCK_SIZE
    block_start_w = (block_id % (input_height // BLOCK_SIZE)) * BLOCK_SIZE
    h_offset = tl.arange(0, BLOCK_SIZE)
    w_offset = tl.arange(0, BLOCK_SIZE)
    offsets_h = h_offset + block_start_h
    offsets_w = w_offset + block_start_w

    mask_h = offsets_h < input_height
    mask_w = offsets_w < input_width
    mask = mask_h & mask_w

    # Load input
    input_batch = tl.arange(0, batch_size)
    input_idx = tl.arange(0, input_channels)
    input_flat_idx = input_batch[:, None] * input_channels * input_height * input_width + \
                     input_idx[None, :] * input_height * input_width + \
                     offsets_h[:, None] * input_width + offsets_w[:, None]
    input_values = tl.load(input_ptr + input_flat_idx, mask=mask, other=0.0)

    # Load weights (O, C_in, 3, 3)
    weight_values = tl.load(weight_ptr + (tl.arange(0, output_channels)[:, None] * input_channels * 9 + input_idx[None, :] * 9 + tl.arange(0, 9)[None, :]), mask=mask, other=0.0)

    # Convolution
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            h_idx = offsets_h + i - padding
            w_idx = offsets_w + j - padding
            h_mask = h_idx >= 0 and h_idx < input_height
            w_mask = w_idx >= 0 and w_idx < input_width
            valid = h_mask & w_mask
            h_offset_val = h_idx - padding
            w_offset_val = w_idx - padding
            input_val = tl.load(input_ptr + (input_batch * input_channels * input_height * input_width + input_idx * input_height * input_width + h_offset_val * input_width + w_offset_val), mask=valid, other=0.0)
            weight_val = tl.load(weight_ptr + (tl.arange(0, output_channels)[:, None] * input_channels * 9 + input_idx[None, :] * 9 + i * 3 + j), mask=valid, other=0.0)
            output += input_val * weight_val

    # Add bias
    bias = tl.load(bias_ptr + tl.arange(0, output_channels), mask=tl.arange(0, output_channels) < output_channels, other=0.0)
    output = output + bias[None, :]

    # Apply ReLU
    output = tl.where(output > 0, output, 0.0)

    # Store output
    output_flat_idx = tl.arange(0, output_channels)[:, None] * BLOCK_SIZE * BLOCK_SIZE + offsets_h[:, None] * BLOCK_SIZE + offsets_w
    tl.store(output_ptr + output_flat_idx, output, mask=mask)


@triton.jit
def conv_dw_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    input_channels: tl.constexpr,
    output_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block processes a region of output
    block_id = tl.program_id(0)
    block_start_h = block_id // (input_height // BLOCK_SIZE) * BLOCK_SIZE
    block_start_w = (block_id % (input_height // BLOCK_SIZE)) * BLOCK_SIZE
    h_offset = tl.arange(0, BLOCK_SIZE)
    w_offset = tl.arange(0, BLOCK_SIZE)
    offsets_h = h_offset + block_start_h
    offsets_w = w_offset + block_start_w

    mask_h = offsets_h < input_height
    mask_w = offsets_w < input_width
    mask = mask_h & mask_w

    # Load input
    input_batch = tl.arange(0, batch_size)
    input_idx = tl.arange(0, input_channels)
    input_flat_idx = input_batch[:, None] * input_channels * input_height * input_width + \
                     input_idx[None, :] * input_height * input_width + \
                     offsets_h[:, None] * input_width + offsets_w[:, None]
    input_values = tl.load(input_ptr + input_flat_idx, mask=mask, other=0.0)

    # Load weights (C_in, 3, 3)
    weight_values = tl.load(weight_ptr + (input_idx[None, :] * 9 + tl.arange(0, 9)[None, :]), mask=mask, other=0.0)

    # Depthwise convolution
    output = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            h_idx = offsets_h + i - padding
            w_idx = offsets_w + j - padding
            h_mask = h_idx >= 0 and h_idx < input_height
            w_mask = w_idx >= 0 and w_idx < input_width
            valid = h_mask & w_mask
            h_offset_val = h_idx - padding
            w_offset_val = w_idx - padding
            input_val = tl.load(input_ptr + (input_batch * input_channels * input_height * input_width + input_idx * input_height * input_width + h_offset_val * input_width + w_offset_val), mask=valid, other=0.0)
            weight_val = tl.load(weight_ptr + (input_idx[None, :] * 9 + i * 3 + j), mask=valid, other=0.0)
            output += input_val * weight_val

    # Apply ReLU
    output = tl.where(output > 0, output, 0.0)

    # Store output
    output_flat_idx = tl.arange(0, output_channels)[:, None] * BLOCK_SIZE * BLOCK_SIZE + offsets_h[:, None] * BLOCK_SIZE + offsets_w
    tl.store(output_ptr + output_flat_idx, output, mask=mask)


def triton_conv_bn_relu(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    output_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    BLOCK_SIZE: int = 128,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda and bias_tensor.is_cuda, "All tensors must be on CUDA"
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    bias_tensor = bias_tensor.contiguous()

    batch_size, input_channels, input_height, input_width = input_tensor.shape
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    output_tensor = torch.empty(
        (batch_size, output_channels, output_height, output_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    grid = lambda meta: (
        ((input_height * input_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
    )

    conv_bn_relu_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        output_tensor.data_ptr(),
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor


def triton_conv_dw(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    BLOCK_SIZE: int = 128,
):
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "All tensors must be on CUDA"
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    batch_size, input_channels, input_height, input_width = input_tensor.shape
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1

    output_tensor = torch.empty(
        (batch_size, output_channels, output_height, output_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    grid = lambda meta: (
        ((input_height * input_width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
    )

    conv_dw_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        output_tensor.data_ptr(),
        batch_size=batch_size,
        input_channels=input_channels,
        output_channels=output_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_tensor


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            # Use Triton kernel for convolution + BN + ReLU
            weight = torch.randn(oup, inp, 3, 3, dtype=torch.float16).cuda()
            bias = torch.zeros(oup, dtype=torch.float16).cuda()
            return weight, bias
        
        def conv_dw(inp, oup, stride):
            # Use Triton kernel for depthwise convolution + ReLU
            weight = torch.randn(inp, inp, 3, 3, dtype=torch.float16).cuda()
            return weight
        
        # Precompute all weights and biases
        self.model = nn.Sequential(
            # Stage 1
            self._conv_block(3, int(32 * alpha), 2),
            self._conv_block(int(32 * alpha), int(64 * alpha), 1),
            self._conv_block(int(64 * alpha), int(128 * alpha), 2),
            self._conv_block(int(128 * alpha), int(128 * alpha), 1),
            self._conv_block(int(128 * alpha), int(256 * alpha), 2),
            self._conv_block(int(256 * alpha), int(256 * alpha), 1),
            self._conv_block(int(256 * alpha), int(512 * alpha), 2),
            self._conv_block(int(512 * alpha), int(512 * alpha), 1),
            self._conv_block(int(512 * alpha), int(512 * alpha), 1),
            self._conv_block(int(512 * alpha), int(512 * alpha), 1),
            self._conv_block(int(512 * alpha), int(512 * alpha), 1),
            self._conv_block(int(512 * alpha), int(512 * alpha), 1),
            self._conv_block(int(512 * alpha), int(1024 * alpha), 2),
            self._conv_block(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def _conv_block(self, inp, oup, stride):
        # Use Triton kernels for depthwise and regular convs
        if stride == 2:
            return nn.Sequential(
                triton_conv_dw(
                    input_tensor=torch.randn(1, inp, 112, 112, dtype=torch.float16).cuda(),
                    weight_tensor=torch.randn(inp, inp, 3, 3, dtype=torch.float16).cuda(),
                    output_channels=oup,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    BLOCK_SIZE=128
                ),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                triton_conv_dw(
                    input_tensor=torch.randn(1, inp, 112, 112, dtype=torch.float16).cuda(),
                    weight_tensor=torch.randn(inp, inp, 3, 3, dtype=torch.float16).cuda(),
                    output_channels=oup,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    BLOCK_SIZE=128
                ),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
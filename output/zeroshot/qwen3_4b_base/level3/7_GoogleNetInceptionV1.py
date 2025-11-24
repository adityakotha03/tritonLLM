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
    GROUPS: tl.constexpr,
):
    # Input and output dimensions
    batch, in_channels, h, w = input_shape
    out_channels, in_channels_per_group, kh, kw = weight_shape
    # Number of groups
    group_size = in_channels // GROUPS if GROUPS > 0 else in_channels

    # Compute the block indices
    block_id = tl.program_id(0)
    block_h = block_id // (w // BLOCK_SIZE)
    block_w = block_id % (w // BLOCK_SIZE)

    # Compute the starting position in the input and output
    h_start = block_h * BLOCK_SIZE
    w_start = block_w * BLOCK_SIZE

    # Define the range of indices for this block
    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)

    # Compute the output position
    h_out = h_idx // kh
    w_out = w_idx // kw

    # Compute the output indices
    h_out = h_out + h_start
    w_out = w_out + w_start

    # Compute the input indices with padding
    h_in = h_idx + pad_h
    w_in = w_idx + pad_w

    # Compute the output stride
    out_stride = h * w

    # Initialize output accumulator
    out = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Loop over the kernel and compute convolution
    for i in range(kh):
        for j in range(kw):
            # Compute input indices
            h_in_idx = h_in + i
            w_in_idx = w_in + j

            # Compute the input offset
            input_offset = (h_in_idx * w + w_in_idx) * in_channels
            # Compute the weight offset
            weight_offset = (i * kh + j) * in_channels_per_group

            # Load input and weight
            input_val = tl.load(input_ptr + input_offset, mask=(h_in_idx < h and w_in_idx < w), other=0.0)
            weight_val = tl.load(weight_ptr + weight_offset, mask=(i < kh and j < kw), other=0.0)

            # Accumulate output
            out += input_val * weight_val

    # Store output
    tl.store(output_ptr + (h_out * w_out), out, mask=(h_out < h and w_out < w))


@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def maxpool2d_kernel(
    x_ptr,
    y_ptr,
    h, w, kh, kw, pad_h, pad_w, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_h = block_id // (w // BLOCK_SIZE)
    block_w = block_id % (w // BLOCK_SIZE)

    h_start = block_h * BLOCK_SIZE
    w_start = block_w * BLOCK_SIZE

    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)

    h_out = h_idx // kh
    w_out = w_idx // kw

    h_in = h_idx + pad_h
    w_in = w_idx + pad_w

    # Output indices
    h_out_idx = h_out + h_start
    w_out_idx = w_out + w_start

    # Load input values
    input_val = tl.load(x_ptr + (h_in * w + w_in), mask=(h_in < h and w_in < w), other=0.0)

    # Compute max over kernel
    max_val = tl.max(input_val)

    # Store result
    tl.store(y_ptr + (h_out_idx * w_out_idx), max_val, mask=(h_out_idx < h and w_out_idx < w))


@triton.jit
def cat_kernel(
    inputs_ptrs,
    output_ptr,
    num_inputs,
    input_shape,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load from each input
    total_channels = 0
    for i in range(num_inputs):
        input_shape_i = input_shape[i]
        input_ptr = inputs_ptrs[i]
        input_channels = input_shape_i[1]
        output_offset = total_channels * input_shape_i[2] * input_shape_i[3]
        total_channels += input_channels
        # Load input
        x = tl.load(input_ptr + offsets, mask=offsets < input_shape_i[2] * input_shape_i[3], other=0.0)
        # Store to output
        tl.store(output_ptr + (output_offset + offsets), x, mask=offsets < input_shape_i[2] * input_shape_i[3])

    # Final output
    tl.store(output_ptr + offsets, offsets, mask=offsets < total_channels * input_shape[0] * input_shape[1])


def triton_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    groups: int = 1,
) -> torch.Tensor:
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch, in_channels, h, w = input.shape
    out_channels, in_channels_per_group, kh, kw = weight.shape

    # Output shape
    out_h = (h + 2 * pad_h - kh) // stride_h + 1
    out_w = (w + 2 * pad_w - kw) // stride_w + 1

    # Output tensor
    out = torch.empty((batch, out_channels, out_h, out_w), dtype=input.dtype, device=input.device)

    # Grid size
    grid = lambda meta: ((out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch kernel
    conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=out.data_ptr(),
        input_shape=(batch, in_channels, h, w),
        weight_shape=(out_channels, in_channels_per_group, kh, kw),
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUPS=groups,
    )
    return out


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    relu_kernel[lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        x.data_ptr(),
        out.data_ptr(),
        x.numel(),
        BLOCK_SIZE=128,
    )
    return out


def triton_maxpool2d(
    x: torch.Tensor,
    kh: int = 3,
    kw: int = 3,
    pad_h: int = 1,
    pad_w: int = 1,
    stride_h: int = 2,
    stride_w: int = 2,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    h, w = x.shape[2], x.shape[3]
    out_h = (h + 2 * pad_h - kh) // stride_h + 1
    out_w = (w + 2 * pad_w - kw) // stride_w + 1
    out = torch.empty((x.shape[0], x.shape[1], out_h, out_w), dtype=x.dtype, device=x.device)
    grid = lambda meta: ((out_h * out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    maxpool2d_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        h, w, kh, kw, pad_h, pad_w, stride_h, stride_w,
        BLOCK_SIZE=128,
    )
    return out


def triton_cat(tensors: list) -> torch.Tensor:
    assert all(t.is_cuda for t in tensors), "All tensors must be on CUDA"
    shapes = [t.shape for t in tensors]
    total_channels = sum(s[1] for s in shapes)
    out = torch.empty((tensors[0].shape[0], total_channels, tensors[0].shape[2], tensors[0].shape[3]), dtype=tensors[0].dtype, device=tensors[0].device)
    cat_kernel[lambda meta: ((out.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)](
        [t.data_ptr() for t in tensors],
        out.data_ptr(),
        len(tensors),
        shapes,
        BLOCK_SIZE=128,
    )
    return out


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
        x = self.maxpool1(triton_relu(self.conv1(x)))
        x = triton_relu(self.conv2(x))
        x = self.maxpool2(triton_relu(self.conv3(x)))
        
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
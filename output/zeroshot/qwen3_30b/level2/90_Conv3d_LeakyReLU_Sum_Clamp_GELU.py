import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Input tensor pointer
    w_ptr,  # Weight tensor pointer
    out_ptr,  # Output tensor pointer
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,  # Strides for input
    w_stride0, w_stride1, w_stride2, w_stride3, w_stride4,  # Strides for weights
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,  # Strides for output
    batch_size, in_channels, out_channels, depth, height, width, kernel_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Thread block index
    pid = tl.program_id(0)
    block_id = pid // 2  # Each block handles two channels
    channel_id = pid % 2
    if block_id >= out_channels // 2:
        return

    # Calculate the output position for this block
    out_c = block_id * 2 + channel_id
    out_d = tl.arange(0, BLOCK_SIZE) % depth
    out_h = (tl.arange(0, BLOCK_SIZE) // depth) % height
    out_w = (tl.arange(0, BLOCK_SIZE) // (depth * height)) % width
    out_batch = (tl.arange(0, BLOCK_SIZE) // (depth * height * width)) % batch_size

    # Compute the starting position in output
    out_offset = (out_batch * out_stride0 + out_c * out_stride1 + out_d * out_stride2 + out_h * out_stride3 + out_w * out_stride4)

    # Load output data
    out = tl.load(out_ptr + out_offset, mask=out_d < depth, other=0.0)

    # Calculate input and weight indices
    in_c = tl.arange(0, in_channels)
    in_d = out_d[:, None] + tl.arange(0, kernel_size)[None, :]
    in_h = out_h[:, None] + tl.arange(0, kernel_size)[None, :]
    in_w = out_w[:, None] + tl.arange(0, kernel_size)[None, :]

    # Load input data
    x_mask = (in_d >= 0) & (in_d < depth) & (in_h >= 0) & (in_h < height) & (in_w >= 0) & (in_w < width)
    x_offset = (out_batch[:, None] * x_stride0 + in_c[None, :] * x_stride1 + in_d * x_stride2 + in_h * x_stride3 + in_w * x_stride4)
    x_data = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

    # Load weights
    w_offset = (out_c * w_stride0 + in_c * w_stride1 + tl.arange(0, kernel_size)[None, :] * w_stride2 + tl.arange(0, kernel_size)[None, :] * w_stride3 + tl.arange(0, kernel_size)[None, :] * w_stride4)
    w_data = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

    # Perform 3D convolution using tensor cores
    for k in range(kernel_size):
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Compute convolution for this kernel position
                k_idx = k * kernel_size * kernel_size + i * kernel_size + j
                w_val = w_data[:, k_idx]
                x_val = x_data[:, k, i, j]
                out += w_val * x_val

    # Store result
    tl.store(out_ptr + out_offset, out, mask=out_d < depth)


@triton.jit
def leaky_relu_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    negative_slope: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    # Offset within block
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for boundary
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply LeakyReLU
    out = tl.where(x > 0, x, negative_slope * x)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def add_tensor_kernel(
    x_ptr,  # Input pointer
    y_ptr,  # Sum tensor pointer
    out_ptr,  # Output pointer
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
    y_stride0, y_stride1, y_stride2, y_stride3, y_stride4,
    batch_size, out_channels, depth, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    # Compute offsets
    out_offset = block_id * BLOCK_SIZE
    # Stride for each dimension
    stride_d = out_stride2
    stride_h = out_stride3
    stride_w = out_stride4
    stride_c = out_stride1
    stride_b = out_stride0

    # Compute coordinate indices
    offset = out_offset
    b = offset // (stride_b)
    c = (offset // (stride_c)) % (out_channels)
    d = (offset // (stride_d)) % (depth)
    h = (offset // (stride_h)) % (height)
    w = (offset // (stride_w)) % (width)

    # Compute index in output
    idx = b * stride_b + c * stride_c + d * stride_d + h * stride_h + w * stride_w

    # Load input and tensor
    x = tl.load(x_ptr + idx, mask=offset < batch_size * out_channels * depth * height * width, other=0.0)
    y = tl.load(y_ptr + c * y_stride0, mask=offset < batch_size * out_channels * depth * height * width, other=0.0)

    # Add
    out = x + y
    # Store result
    tl.store(out_ptr + idx, out, mask=offset < batch_size * out_channels * depth * height * width)


@triton.jit
def clamp_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    min_val: tl.float32,
    max_val: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    # Offset within block
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for boundary
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Clamp
    out = tl.clamp(x, min_val, max_val)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,  # Input pointer
    out_ptr,  # Output pointer
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    # Block ID
    block_id = tl.program_id(0)
    # Offset within block
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask for boundary
    mask = offsets < n_elements
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_pi = tl.sqrt(2.0 / 3.141592653589793)
    x3 = x * x * x
    inner = sqrt_2_pi * (x + 0.044715 * x3)
    tanh_val = tl.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv3d(x, w, out_shape, kernel_size, stride=1, padding=0):
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    # Parameters
    batch_size, in_channels, depth, height, width = x.shape
    out_channels, _, _, _ = w.shape
    _, _, out_depth, out_height, out_width = out_shape

    # Strides
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3, w_stride4 = w.stride()
    out_stride0, out_stride1, out_stride2, out_stride3, out_stride4 = out.stride()

    # Grid
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tune for A100
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    conv3d_kernel[grid](
        x, w, out,
        x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
        w_stride0, w_stride1, w_stride2, w_stride3, w_stride4,
        out_stride0, out_stride1, out_stride2, out_stride3, out_stride4,
        batch_size, in_channels, out_channels, depth, height, width, kernel_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_leaky_relu(x, negative_slope=0.2):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    leaky_relu_kernel[grid](x, out, n_elements, negative_slope, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_add_tensor(x, sum_tensor):
    assert x.is_cuda and sum_tensor.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, out_channels, depth, height, width = x.shape
    sum_tensor = sum_tensor.contiguous()
    x_stride0, x_stride1, x_stride2, x_stride3, x_stride4 = x.stride()
    y_stride0, y_stride1, y_stride2, y_stride3, y_stride4 = sum_tensor.stride()

    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_tensor_kernel[grid](
        x, sum_tensor, out,
        x_stride0, x_stride1, x_stride2, x_stride3, x_stride4,
        y_stride0, y_stride1, y_stride2, y_stride3, y_stride4,
        batch_size, out_channels, depth, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def triton_clamp(x, min_val=-1.0, max_val=1.0):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    clamp_kernel[grid](x, out, n_elements, min_val, max_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_gelu(x):
    assert x.is_cuda, "Input must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    gelu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        # Replace each operation with Triton kernel
        x = triton_conv3d(x, self.conv.weight, (x.shape[0], self.conv.out_channels, x.shape[2], x.shape[3], x.shape[4]), kernel_size=self.conv.kernel_size[0])
        x = triton_leaky_relu(x, negative_slope=0.2)
        x = triton_add_tensor(x, self.sum_tensor)
        x = triton_clamp(x, min_val=-1.0, max_val=1.0)
        x = triton_gelu(x)
        return x
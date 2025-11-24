import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # pointer to input tensor (B, C_in, D, H, W)
    output_ptr,  # pointer to output tensor (B, C_out, D, H, W)
    input_shape,  # (batch_size, in_channels, depth, height, width)
    output_shape,  # (batch_size, out_channels, depth, height, width)
    kernel,  # (out_channels, in_channels, d_k, h_k, w_k)
    kernel_stride,  # (d_stride, h_stride, w_stride)
    pad,  # (pad_d, pad_h, pad_w)
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Get the program ID
    batch_idx = tl.program_id(0)
    out_channel = tl.program_id(1)
    d_idx = tl.program_id(2)
    h_idx = tl.program_id(3)
    w_idx = tl.program_id(4)

    # Compute the global indices
    batch = batch_idx
    out_c = out_channel
    d = d_idx
    h = h_idx
    w = w_idx

    # Compute the output spatial indices
    d_start = d
    h_start = h
    w_start = w

    # Compute the kernel spatial indices
    d_k = tl.arange(0, TILE_SIZE)
    h_k = tl.arange(0, TILE_SIZE)
    w_k = tl.arange(0, TILE_SIZE)

    # Compute the input spatial indices (with padding)
    d_in = d_start + d_k
    h_in = h_start + h_k
    w_in = w_start + w_k

    # Apply padding bounds
    d_in = d_in + pad[0]
    h_in = h_in + pad[1]
    w_in = w_in + pad[2]

    # Define the valid range for input
    valid_d = (d_in >= 0) & (d_in < input_shape[4])
    valid_h = (h_in >= 0) & (h_in < input_shape[3])
    valid_w = (w_in >= 0) & (w_in < input_shape[2])

    # Create a mask for valid indices
    mask = valid_d & valid_h & valid_w

    # Load kernel values
    kernel_vals = tl.load(kernel + (out_c, tl.arange(0, input_shape[1]), d_k, h_k, w_k), mask=mask, other=0.0)

    # Load input values
    input_vals = tl.load(input_ptr + (batch, tl.arange(0, input_shape[1]), d_in, h_in, w_in), mask=mask, other=0.0)

    # Compute the output
    output = tl.zeros((TILE_SIZE, TILE_SIZE, TILE_SIZE), dtype=tl.float32)
    for i in range(TILE_SIZE):
        for j in range(TILE_SIZE):
            for k in range(TILE_SIZE):
                idx = (i, j, k)
                if mask[i, j, k]:
                    output[i, j, k] = input_vals[i, j, k] * kernel_vals[i, j, k]

    # Accumulate the output
    output = tl.sum(output, axis=(0, 1, 2))

    # Store the result
    tl.store(output_ptr + (batch, out_c, d, h, w), output, mask=mask)


@triton.jit
def leaky_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    negative_slope: tl.constexpr = 0.2,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x >= 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x < -1.0, -1.0, tl.where(x > 1.0, 1.0, x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU: x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    tanh_arg = tl.sqrt(2.0 / tl.pi) * (x + 0.044715 * x3)
    tanh_val = tl.tanh(tanh_arg)
    out = x * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv3d(
    input_tensor: torch.Tensor,
    kernel: torch.Tensor,
    pad: tuple = (1, 1, 1),
    stride: tuple = (1, 1, 1),
    dilation: tuple = (1, 1, 1),
):
    batch_size, in_channels, depth, height, width = input_tensor.shape
    out_channels, in_channels_k, d_k, h_k, w_k = kernel.shape
    d_stride, h_stride, w_stride = stride
    d_pad, h_pad, w_pad = pad

    # Output shape computation
    out_depth = (depth + 2 * d_pad - (d_k - 1) * dilation[0] - 1) // d_stride + 1
    out_height = (height + 2 * h_pad - (h_k - 1) * dilation[1] - 1) // h_stride + 1
    out_width = (width + 2 * w_pad - (w_k - 1) * dilation[2] - 1) // w_stride + 1

    # Prepare output tensor
    output = torch.empty(
        (batch_size, out_channels, out_depth, out_height, out_width),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

    # Define grid
    BLOCK_SIZE = 128
    TILE_SIZE = 16

    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        out_channels,
        out_depth,
        out_height,
        out_width,
    )

    # Launch kernel
    conv3d_kernel[grid](
        input_tensor.data_ptr(),
        output.data_ptr(),
        (batch_size, in_channels, depth, height, width),
        (batch_size, out_channels, out_depth, out_height, out_width),
        kernel.data_ptr(),
        (d_stride, h_stride, w_stride),
        (d_pad, h_pad, w_pad),
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
    )
    return output


def triton_leaky_relu(x: torch.Tensor, negative_slope: float = 0.2):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    leaky_relu_kernel[grid](x.data_ptr(), out.data_ptr(), x.numel(), BLOCK_SIZE=BLOCK_SIZE, negative_slope=negative_slope)
    return out


def triton_clamp(x: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    clamp_kernel[grid](x.data_ptr(), out.data_ptr(), x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda, "Input must be on CUDA"
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK_SIZE = 256
    grid = lambda meta: ((x.numel() + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    gelu_kernel[grid](x.data_ptr(), out.data_ptr(), x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sum_tensor_shape = sum_tensor_shape
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

        # Define kernel shape
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))

        # Initialize padding and stride
        self.pad = (1, 1, 1)
        self.stride = (1, 1, 1)

    def forward(self, x):
        # 1. Apply 3D convolution
        x = triton_conv3d(x, self.kernel, pad=self.pad, stride=self.stride)
        
        # 2. Apply LeakyReLU
        x = triton_leaky_relu(x, negative_slope=0.2)
        
        # 3. Add sum tensor
        x = x + self.sum_tensor
        
        # 4. Clamp to [-1, 1]
        x = triton_clamp(x, min_val=-1.0, max_val=1.0)
        
        # 5. Apply GELU
        x = triton_gelu(x)
        
        return x
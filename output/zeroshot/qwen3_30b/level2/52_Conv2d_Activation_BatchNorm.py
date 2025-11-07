import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    batch_size, in_channels, out_channels, height, width, kernel_size,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Calculate the current block's output indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_k = tl.program_id(3)

    # Calculate the output height and width
    out_h = (height - kernel_size + 1)
    out_w = (width - kernel_size + 1)

    # Offset in the output tensor
    out_h_offset = pid_h * BLOCK_H
    out_w_offset = pid_w * BLOCK_W
    out_c_offset = pid_c * BLOCK_C
    out_k_offset = pid_k * BLOCK_K

    # Load the output tile (shared memory) - only for current block
    # Use shared memory for input and weights to reduce global memory access
    x_block = tl.load(
        x_ptr + (out_h_offset + tl.arange(0, BLOCK_H)[:, None]) * x_stride2 +
        (out_w_offset + tl.arange(0, BLOCK_W)[None, :]) * x_stride3 +
        (tl.arange(0, BLOCK_C)[:, None, None]) * x_stride1 +
        (tl.arange(0, BLOCK_K)[None, None, :]) * x_stride0,
        mask=(
            (out_h_offset + tl.arange(0, BLOCK_H)[:, None]) < out_h
            if BLOCK_H < out_h else True
        ) & (
            (out_w_offset + tl.arange(0, BLOCK_W)[None, :]) < out_w
            if BLOCK_W < out_w else True
        ) & (
            (out_c_offset + tl.arange(0, BLOCK_C)[:, None, None]) < in_channels
            if BLOCK_C < in_channels else True
        ) & (
            (out_k_offset + tl.arange(0, BLOCK_K)[None, None, :]) < kernel_size
            if BLOCK_K < kernel_size else True
        ),
        other=0.0
    )

    # Load weight tile
    w_block = tl.load(
        w_ptr + (tl.arange(0, BLOCK_C)[:, None, None]) * w_stride0 +
        (tl.arange(0, BLOCK_K)[None, None, :]) * w_stride1 +
        (tl.arange(0, BLOCK_H)[None, :, None]) * w_stride2 +
        (tl.arange(0, BLOCK_W)[None, None, :]) * w_stride3,
        mask=(
            (tl.arange(0, BLOCK_C)[:, None, None]) < out_channels
            if BLOCK_C < out_channels else True
        ) & (
            (tl.arange(0, BLOCK_K)[None, None, :]) < kernel_size
            if BLOCK_K < kernel_size else True
        ) & (
            (tl.arange(0, BLOCK_H)[None, :, None]) < kernel_size
            if BLOCK_H < kernel_size else True
        ) & (
            (tl.arange(0, BLOCK_W)[None, None, :]) < kernel_size
            if BLOCK_W < kernel_size else True
        ),
        other=0.0
    )

    # Perform convolution via matmul
    # Output shape: [BLOCK_H, BLOCK_W, BLOCK_C]
    acc = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_C), dtype=tl.float32)
    for k in range(kernel_size):
        # Extract input slice at current kernel position
        x_slice = x_block[:, :, :, k]  # [BLOCK_H, BLOCK_W, BLOCK_C]
        # Extract weight slice at current kernel position
        w_slice = w_block[:, k, :, :]  # [BLOCK_C, BLOCK_H, BLOCK_W]
        # Perform matrix multiplication (contract over k)
        acc += tl.dot(x_slice, w_slice)  # [BLOCK_H, BLOCK_W, BLOCK_C]

    # Store output
    out_ptr_offset = (
        out_h_offset * out_stride2 + out_w_offset * out_stride3 +
        out_c_offset * out_stride1 + (tl.arange(0, BLOCK_C)[None, None, :]) * out_stride0
    )
    tl.store(
        out_ptr + out_ptr_offset,
        acc,
        mask=(
            (out_h_offset + tl.arange(0, BLOCK_H)[:, None]) < out_h
            if BLOCK_H < out_h else True
        ) & (
            (out_w_offset + tl.arange(0, BLOCK_W)[None, :]) < out_w
            if BLOCK_W < out_w else True
        ) & (
            (out_c_offset + tl.arange(0, BLOCK_C)[None, None, :]) < out_channels
            if BLOCK_C < out_channels else True
        )
    )


@triton.jit
def act_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute softplus: log(1 + exp(x))
    x_softplus = tl.log(1.0 + tl.exp(x))
    # Compute tanh(softplus(x))
    x_tanh = tl.tanh(x_softplus)
    # Compute x * tanh(softplus(x))
    out = x * x_tanh
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def bn_kernel(
    x_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    out_ptr,
    n_elements,
    eps: tl.float32,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Load running mean and variance
    mean = tl.load(mean_ptr, mask=mask, other=0.0)
    var = tl.load(var_ptr, mask=mask, other=0.0)
    # Load gamma and beta
    gamma = tl.load(gamma_ptr, mask=mask, other=0.0)
    beta = tl.load(beta_ptr, mask=mask, other=0.0)
    # Normalize: (x - mean) / sqrt(var + eps)
    x_norm = (x - mean) * tl.rsqrt(var + eps)
    # Scale and shift
    out = gamma * x_norm + beta
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x, w, out_shape, stride_h, stride_w, padding_h, padding_w):
    # Ensure contiguous tensors on GPU
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape
    out_h = (height + 2 * padding_h - kernel_size) // stride_h + 1
    out_w = (width + 2 * padding_w - kernel_size) // stride_w + 1

    # Define block sizes
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 32
    BLOCK_K = 3

    # Grid dimensions
    grid_h = (out_h + BLOCK_H - 1) // BLOCK_H
    grid_w = (out_w + BLOCK_W - 1) // BLOCK_W
    grid_c = (out_channels + BLOCK_C - 1) // BLOCK_C
    grid_k = (kernel_size + BLOCK_K - 1) // BLOCK_K
    grid = (grid_h, grid_w, grid_c, grid_k)

    # Launch kernel
    conv2d_kernel[grid](
        x, w, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, in_channels, out_channels, height, width, kernel_size,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K
    )
    return out


def triton_act(x):
    # Ensure contiguous tensors on GPU
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()

    # Define block size
    BLOCK_SIZE = 128

    # Grid
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch kernel
    act_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def triton_bn(x, gamma, beta, mean, var, eps):
    # Ensure contiguous tensors on GPU
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()

    # Define block size
    BLOCK_SIZE = 128

    # Grid
    grid = lambda meta: (n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch kernel
    bn_kernel[grid](
        x, gamma, beta, mean, var, out, n_elements, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Step 1: Convolution with Triton
        x = triton_conv2d(
            x,
            self.conv.weight,
            out_shape=(x.shape[0], self.conv.out_channels, x.shape[2] - self.conv.kernel_size[0] + 1, x.shape[3] - self.conv.kernel_size[1] + 1),
            stride_h=self.conv.stride[0],
            stride_w=self.conv.stride[1],
            padding_h=self.conv.padding[0],
            padding_w=self.conv.padding[1]
        )

        # Step 2: Apply activation function with Triton
        x = triton_act(x)

        # Step 3: BatchNorm with Triton
        x = triton_bn(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        return x
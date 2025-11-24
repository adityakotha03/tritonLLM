import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    output_ptr,  # pointer to output tensor (batch, out_channels, H, W)
    input_shape,  # (batch, in_channels, H, W)
    output_shape,  # (batch, out_channels, H, W)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # pointer to bias tensor (out_channels,)
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    GROUPS: tl.constexpr,
):
    # Program ID for block
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h = tl.program_id(2)
    out_w = tl.program_id(3)

    # Compute the block boundaries
    batch_size, in_channels, H, W = input_shape
    out_channels, _, kernel_size, _ = weight_ptr.shape
    h_start = out_h * BLOCK_SIZE_H
    h_end = h_start + BLOCK_SIZE_H
    w_start = out_w * BLOCK_SIZE_W
    w_end = w_start + BLOCK_SIZE_W

    # Define the range of indices for this block
    h_indices = tl.arange(0, BLOCK_SIZE_H)
    w_indices = tl.arange(0, BLOCK_SIZE_W)

    # Compute the valid region (within input bounds)
    mask_h = h_indices < H
    mask_w = w_indices < W
    mask_hw = mask_h[:, None] & mask_w[None, :]

    # Load input features (batch, in_channels, H, W)
    # We assume input is stored in (batch, in_channels, H, W) format
    # For each output channel, we compute the convolution across input channels
    # Use shared memory to reduce global memory access
    # We will compute the output at (batch_idx, out_channel_idx, h, w)

    # Load weights (out_channels, in_channels, kernel_size, kernel_size)
    # We will use a tiled approach to avoid loading entire weight matrix
    # For simplicity, we assume we are computing one output channel at a time
    # and use a single block to compute the convolution

    # Compute output value at (batch_idx, out_channel_idx, h, w)
    out_val = 0.0
    for in_channel in range(in_channels):
        # Load input patch at (batch_idx, in_channel, h, w)
        # We use a loop over the kernel size
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Compute input indices
                input_h = h_indices + k_h
                input_w = w_indices + k_w
                # Mask to avoid out-of-bounds
                valid_h = input_h < H
                valid_w = input_w < W
                valid_hw = valid_h & valid_w
                # Load input value
                input_val = tl.load(
                    input_ptr + batch_idx * in_channels * H * W +
                    in_channel * H * W + input_h * W + input_w,
                    mask=valid_hw,
                    other=0.0
                )
                # Load weight value
                weight_val = tl.load(
                    weight_ptr + out_channel_idx * in_channels * kernel_size * kernel_size +
                    in_channel * kernel_size * kernel_size + k_h * kernel_size + k_w,
                    mask=valid_hw,
                    other=0.0
                )
                out_val += input_val * weight_val

    # Add bias
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx, mask=1, other=0.0)
        out_val += bias_val

    # Store output
    tl.store(
        output_ptr + batch_idx * out_channels * H * W +
        out_channel_idx * H * W + out_h * W + out_w,
        out_val,
        mask=mask_hw
    )


@triton.jit
def subtract_kernel(
    x_ptr,  # pointer to input tensor
    subtract_val,  # scalar value to subtract
    out_ptr,  # pointer to output tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x - subtract_val
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def mish_kernel(
    x_ptr,  # pointer to input tensor
    out_ptr,  # pointer to output tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Mish activation: x * tanh(ln(1 + exp(x)))
    log_exp_x = tl.math.log(tl.math.exp(x) + 1.0)
    tanh_log_exp_x = tl.math.tanh(log_exp_x)
    out = x * tanh_log_exp_x
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(
    input_tensor,  # (batch, in_channels, H, W)
    weight_tensor,  # (out_channels, in_channels, kernel_size, kernel_size)
    bias_tensor=None,  # (out_channels,)
):
    """
    Conv2D using Triton kernel with block-level parallelization.
    """
    assert input_tensor.is_cuda and weight_tensor.is_cuda, "Tensors must be on CUDA."
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()

    batch_size, in_channels, H, W = input_tensor.shape
    out_channels, _, kernel_size, _ = weight_tensor.shape

    # Output tensor
    output_shape = (batch_size, out_channels, H, W)
    output_tensor = torch.empty(output_shape, device=input_tensor.device)

    # Grid and block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    GROUPS = 1

    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_channels + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        (H + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (W + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
    )

    # Launch the kernel
    conv2d_kernel[grid](
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        (batch_size, in_channels, H, W),
        (batch_size, out_channels, H, W),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr() if bias_tensor is not None else None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        GROUPS=GROUPS,
    )
    return output_tensor


def triton_subtract(x: torch.Tensor, subtract_val: float):
    """
    Element-wise subtraction using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    subtract_kernel[grid](
        x.data_ptr(),
        subtract_val,
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def triton_mish(x: torch.Tensor):
    """
    Mish activation using Triton kernel.
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    mish_kernel[grid](
        x.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value_1, subtract_value_2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.subtract_value_1 = subtract_value_1
        self.subtract_value_2 = subtract_value_2

        # Initialize convolution weights and bias
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
        self.bias = torch.randn(out_channels).cuda()

    def forward(self, x):
        # Convolution using Triton kernel
        x = triton_conv2d(x, self.weight, self.bias)
        # Subtract values using Triton kernels
        x = triton_subtract(x, self.subtract_value_1)
        x = triton_subtract(x, self.subtract_value_2)
        # Apply Mish activation
        x = triton_mish(x)
        return x
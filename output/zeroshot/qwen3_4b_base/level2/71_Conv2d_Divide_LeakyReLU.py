import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, H, W)
    weight_ptr,  # pointer to convolution weights (out_channels, in_channels, kernel_size, kernel_size)
    bias_ptr,  # pointer to bias (out_channels)
    output_ptr,  # pointer to output tensor (batch, out_channels, H, W)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the grid size
    pid = tl.program_id(0)
    batch_idx = pid // (height * width // BLOCK_SIZE)
    if batch_idx >= batch_size:
        return

    # Compute the block's starting position in the spatial dimensions
    h_start = (pid % (height * width // BLOCK_SIZE)) // width
    w_start = (pid % (height * width // BLOCK_SIZE)) % width

    # Compute the block of output indices
    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    h_idx = h_idx + h_start
    w_idx = w_idx + w_start

    # Compute valid bounds
    h_valid = (h_idx >= 0) & (h_idx < height)
    w_valid = (w_idx >= 0) & (w_idx < width)
    mask = h_valid & w_valid

    # Compute the output channel and input channel indices
    output_h = h_idx
    output_w = w_idx
    output_idx = output_h * width + output_w

    # Compute the input indices (with padding)
    input_h = h_idx - padding
    input_w = w_idx - padding
    input_h = input_h + tl.arange(0, kernel_size) // kernel_size
    input_w = input_w + tl.arange(0, kernel_size) % kernel_size

    # Expand the kernel and input dimensions
    # We use a tile-based approach to compute the convolution
    # We assume kernel_size is odd, and padding is symmetric
    kernel_h = tl.arange(0, kernel_size)
    kernel_w = tl.arange(0, kernel_size)

    # Compute the convolution sum
    # We will use a 2D convolution via a loop over kernel positions
    # But we use a more efficient approach: loop over kernel positions and accumulate
    # We assume the kernel is symmetric and use a block-wise tiling
    # For simplicity, we compute the full convolution in a single kernel
    # We will compute the output for each spatial position in the block

    # We restructure the kernel to avoid nested loops
    # We use a single loop over kernel positions and accumulate
    # We will compute the convolution using a tiled approach

    # Load input and weights
    # We assume input is (batch, in_channels, H, W)
    # We assume weights are (out_channels, in_channels, k, k)

    # Compute the output for each output channel
    for out_c in tl.arange(0, out_channels):
        # Load bias
        bias = tl.load(bias_ptr + out_c, mask=(out_c < out_channels), other=0.0)

        # Initialize output
        out_val = 0.0

        # Loop over kernel positions
        for k_h in tl.arange(0, kernel_size):
            for k_w in tl.arange(0, kernel_size):
                # Compute input indices
                i_h = input_h + k_h
                i_w = input_w + k_w

                # Compute input channel index
                for in_c in tl.arange(0, in_channels):
                    # Load input
                    input_val = tl.load(
                        input_ptr + batch_idx * in_channels * height * width +
                        in_c * height * width + i_h * width + i_w,
                        mask=(i_h >= 0) & (i_h < height) & (i_w >= 0) & (i_w < width),
                        other=0.0
                    )
                    # Load weight
                    weight_val = tl.load(
                        weight_ptr + out_c * in_channels * kernel_size * kernel_size +
                        in_c * kernel_size * kernel_size + k_h * kernel_size + k_w,
                        mask=(k_h < kernel_size) & (k_w < kernel_size),
                        other=0.0
                    )
                    out_val += input_val * weight_val

        # Apply leaky ReLU with slope 0.01
        # We do this in the kernel to avoid memory transfer
        out_val = out_val * (1.0 if out_val >= 0 else 0.01)
        out_val = out_val + bias

        # Store output
        tl.store(
            output_ptr + batch_idx * out_channels * height * width + out_c * height * width + output_idx,
            out_val,
            mask=mask
        )


@triton.jit
def conv2d_leaky_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # We fuse convolution and leaky ReLU into a single kernel
    pid = tl.program_id(0)
    batch_idx = pid // (height * width // BLOCK_SIZE)
    if batch_idx >= batch_size:
        return

    h_start = (pid % (height * width // BLOCK_SIZE)) // width
    w_start = (pid % (height * width // BLOCK_SIZE)) % width

    h_idx = tl.arange(0, BLOCK_SIZE)
    w_idx = tl.arange(0, BLOCK_SIZE)
    h_idx = h_idx + h_start
    w_idx = w_idx + w_start

    h_valid = (h_idx >= 0) & (h_idx < height)
    w_valid = (w_idx >= 0) & (w_idx < width)
    mask = h_valid & w_valid

    # Compute output indices
    output_idx = h_idx * width + w_idx

    # Compute input indices with padding
    input_h = h_idx - padding
    input_w = w_idx - padding

    # Compute kernel indices
    kernel_h = tl.arange(0, kernel_size)
    kernel_w = tl.arange(0, kernel_size)

    # Accumulate convolution
    out_val = 0.0
    for out_c in tl.arange(0, out_channels):
        bias = tl.load(bias_ptr + out_c, mask=(out_c < out_channels), other=0.0)

        # Convolution sum
        for k_h in kernel_h:
            for k_w in kernel_w:
                i_h = input_h + k_h
                i_w = input_w + k_w
                for in_c in tl.arange(0, in_channels):
                    input_val = tl.load(
                        input_ptr + batch_idx * in_channels * height * width +
                        in_c * height * width + i_h * width + i_w,
                        mask=(i_h >= 0) & (i_h < height) & (i_w >= 0) & (i_w < width),
                        other=0.0
                    )
                    weight_val = tl.load(
                        weight_ptr + out_c * in_channels * kernel_size * kernel_size +
                        in_c * kernel_size * kernel_size + k_h * kernel_size + k_w,
                        mask=(k_h < kernel_size) & (k_w < kernel_size),
                        other=0.0
                    )
                    out_val += input_val * weight_val

        # Apply leaky ReLU
        out_val = out_val * (1.0 if out_val >= 0 else 0.01) + bias

        # Store output
        tl.store(
            output_ptr + batch_idx * out_channels * height * width + out_c * height * width + output_idx,
            out_val,
            mask=mask
        )


def triton_conv2d_leaky_relu(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    bias_tensor: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    padding: int = 1,
    divisor: float = 1.0,
) -> torch.Tensor:
    """
    Custom Triton kernel that performs convolution with LeakyReLU activation.
    """
    assert input_tensor.is_cuda, "Input tensor must be on CUDA."
    assert weight_tensor.is_cuda, "Weight tensor must be on CUDA."
    assert bias_tensor.is_cuda, "Bias tensor must be on CUDA."

    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight_tensor.shape[0]

    # Ensure tensors are contiguous
    input_tensor = input_tensor.contiguous()
    weight_tensor = weight_tensor.contiguous()
    bias_tensor = bias_tensor.contiguous()

    # Output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Define block size
    BLOCK_SIZE = 128

    # Grid size
    grid = lambda meta: (
        (batch_size * height * width + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_leaky_relu_kernel[
        grid
    ](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        output_tensor.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_tensor


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor

        # Initialize convolution weights and bias
        self.weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size).cuda()
        self.bias = torch.zeros(out_channels).cuda()

    def forward(self, x):
        # Perform convolution with leaky ReLU directly via Triton kernel
        return triton_conv2d_leaky_relu(
            x,
            self.weight,
            self.bias,
            kernel_size=self.kernel_size,
            stride=1,
            padding=1,
            divisor=self.divisor
        )
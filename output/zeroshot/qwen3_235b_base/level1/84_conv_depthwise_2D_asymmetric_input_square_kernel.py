import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,          # pointer to input tensor (batch_size, in_channels, height_in, width_in)
    w_ptr,          # pointer to weights (out_channels, 1, kernel_size, kernel_size)
    y_ptr,          # pointer to output tensor (batch_size, out_channels, height_out, width_out)
    batch_size,
    in_channels,
    out_channels,
    height_in,
    width_in,
    height_out,
    width_out,
    kernel_size,
    stride,
    padding,
    eps: tl.constexpr,  # block size padding for divisibility
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(axis=0)  # batch dimension
    pid_c = tl.program_id(axis=1)  # channel dimension
    pid_hw = tl.program_id(axis=2)  # spatial dimension (H_out * W_out)

    # Compute output spatial indices
    hw_offset = pid_hw * BLOCK_SIZE_HW
    h_out = (hw_offset // width_out) % height_out
    w_out = hw_offset % width_out

    # Input spatial start (accounting for stride and padding)
    h_in_start = h_out * stride - padding
    w_in_start = w_out * stride - padding

    # Pointers to input and output
    x_offset = pid_b * in_channels * height_in * width_in + pid_c * height_in * width_in
    y_offset = pid_b * out_channels * height_out * width_out + pid_c * height_out * width_out + h_out * width_out + w_out

    # Weight pointer (assuming grouped conv: one kernel per output channel)
    w_offset = pid_c * kernel_size * kernel_size

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_HW,), dtype=tl.float32)

    # Loop over kernel window
    for kh in range(0, kernel_size):
        for kw in range(0, kernel_size):
            # Input coordinate
            h_in = h_in_start + kh
            w_in = w_in_start + kw

            # Check bounds
            mask_kh = (h_in >= 0) & (h_in < height_in)
            mask_kw = (w_in >= 0) & (w_in < width_in)
            mask_hw = (hw_offset + tl.arange(0, BLOCK_SIZE_HW)) < height_out * width_out
            valid_hw = mask_hw
            valid = valid_hw & mask_kh & mask_kw

            # Input index
            x_index = x_offset + (h_in * width_in + w_in) + tl.arange(0, BLOCK_SIZE_HW)
            x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)

            # Weight value
            w_val = tl.load(w_ptr + w_offset + kh * kernel_size + kw)

            # Multiply-accumulate
            acc += x_val * w_val

    # Store output
    y_index = y_offset + tl.arange(0, BLOCK_SIZE_HW)
    tl.store(y_ptr + y_index, acc, mask=(hw_offset + tl.arange(0, BLOCK_SIZE_HW)) < height_out * width_out)


def triton_depthwise_conv2d(x, weight, bias, stride, padding, groups):
    # Assume groups == in_channels == out_channels for depthwise
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size, _ = weight.shape
    assert in_channels == out_channels, "Depthwise conv requires in_channels == out_channels"
    assert groups == in_channels, "Groups must equal in_channels for depthwise"

    # Compute output spatial dimensions
    height_out = (height_in + 2 * padding - kernel_size) // stride + 1
    width_out = (width_in + 2 * padding - kernel_size) // stride + 1

    # Output tensor
    y = torch.zeros(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

    # Launch kernel
    def grid(META):
        return (
            batch_size,
            in_channels,
            triton.cdiv(height_out * width_out, META["BLOCK_SIZE_HW"]),
        )

    # Heuristics for block sizes
    BLOCK_SIZE_M = 1
    BLOCK_SIZE_HW = 64
    BLOCK_SIZE_K = 16

    depthwise_conv2d_kernel[grid](
        x,
        weight,
        y,
        batch_size,
        in_channels,
        out_channels,
        height_in,
        width_in,
        height_out,
        width_out,
        kernel_size,
        stride,
        padding,
        eps=16,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    # Add bias if present
    if bias is not None:
        y += bias.view(1, -1, 1, 1)

    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = in_channels  # depthwise

        # Depthwise weights and optional bias
        self.weight = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.groups)
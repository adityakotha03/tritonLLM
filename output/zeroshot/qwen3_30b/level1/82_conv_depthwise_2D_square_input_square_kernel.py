import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Thread indices
    pid_batch = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Compute output indices
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Compute input indices
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Mask out-of-bounds indices
    h_mask = (out_h < out_height) & (in_h >= 0) & (in_h < height)
    w_mask = (out_w < out_width) & (in_w >= 0) & (in_w < width)

    # Expand masks to 4D
    h_mask_4d = h_mask[:, None, None, None]
    w_mask_4d = w_mask[None, :, None, None]

    # Compute offsets for input, weight, and output
    input_offset = (pid_batch * in_channels + pid_c) * height * width
    weight_offset = pid_c * kernel_size * kernel_size
    output_offset = (pid_batch * in_channels + pid_c) * out_height * out_width

    # Load input values
    input_ptrs = input_ptr + input_offset + in_h[:, None] * width + in_w[None, :]
    input_data = tl.load(input_ptrs, mask=h_mask_4d & w_mask_4d, other=0.0)

    # Load weight values
    weight_ptrs = weight_ptr + weight_offset + tl.arange(0, kernel_size)[:, None] * kernel_size + tl.arange(0, kernel_size)[None, :]
    weight_data = tl.load(weight_ptrs, mask=tl.arange(0, kernel_size)[:, None] < kernel_size, other=0.0)

    # Perform convolution
    # Inner loop over kernel size
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for kh in range(0, kernel_size, BLOCK_SIZE_K):
        for kw in range(0, kernel_size, BLOCK_SIZE_K):
            # Compute current kernel block
            kkh = tl.arange(0, BLOCK_SIZE_K)
            kkw = tl.arange(0, BLOCK_SIZE_K)
            k_mask = (kkh < kernel_size - kh) & (kkw < kernel_size - kw)

            # Load kernel values
            k_ptrs = weight_ptr + weight_offset + (kh + kkh)[:, None] * kernel_size + (kw + kkw)[None, :]
            k_vals = tl.load(k_ptrs, mask=k_mask[:, None] & k_mask[None, :], other=0.0)

            # Compute input block
            i_h = in_h[:, None] + kkh[None, :]
            i_w = in_w[None, :] + kkw[:, None]
            i_mask = (i_h < height) & (i_w < width)

            # Load input block
            i_ptrs = input_ptr + input_offset + i_h[:, :, None, None] * width + i_w[None, None, :, :]
            i_vals = tl.load(i_ptrs, mask=i_mask[:, :, None, :], other=0.0)

            # Accumulate
            output += tl.sum(k_vals[None, None, :, :] * i_vals, axis=(2, 3))

    # Store output
    out_ptrs = output_ptr + output_offset + out_h[:, None] * out_width + out_w[None, :]
    tl.store(out_ptrs, output, mask=h_mask_4d & w_mask_4d)


def triton_depthwise_conv2d(input: torch.Tensor, weight: torch.Tensor, stride: int, padding: int, kernel_size: int):
    """
    Wrapper function for Triton-based depthwise 2D convolution.

    Args:
        input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        weight (torch.Tensor): Weight tensor of shape (in_channels, kernel_size, kernel_size).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        kernel_size (int): Size of the convolution kernel.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, in_channels, out_height, out_width).
    """
    batch_size, in_channels, height, width = input.shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    # Ensure inputs are contiguous
    input = input.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor
    output = torch.empty(batch_size, in_channels, out_height, out_width, dtype=input.dtype, device=input.device)

    # Configure kernel launch parameters
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_C = 16

    # Grid dimensions
    grid = (
        batch_size,
        in_channels,
        (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )

    # Launch kernel
    depthwise_conv2d_kernel[grid](
        input,
        weight,
        output,
        batch_size,
        in_channels,
        height,
        width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, kernel_size, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(x, self.weight, self.stride, self.padding, self.kernel_size)
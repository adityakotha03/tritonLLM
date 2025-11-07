import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    out_height: tl.constexpr,
    out_width: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Calculate the global indices for the current thread block
    pid_h = tl.program_id(0)  # Height dimension
    pid_w = tl.program_id(1)  # Width dimension
    pid_c = tl.program_id(2)  # Channel dimension

    # Calculate the output location
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    out_h = tl.clamp(out_h, 0, out_height - 1)
    out_w = tl.clamp(out_w, 0, out_width - 1)

    # Calculate the corresponding input locations
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Apply dilation
    in_h = in_h * dilation
    in_w = in_w * dilation

    # Define a mask to avoid out-of-bounds access
    mask_h = (in_h >= 0) & (in_h < height)
    mask_w = (in_w >= 0) & (in_w < width)

    # Flatten the output indices
    out_offset = (pid_h * out_width + pid_w) * BLOCK_SIZE_H * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_H)[:, None] * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)[None, :]
    out_offset = out_offset.flatten()

    # Input and weight pointers
    in_ptr = x_ptr + (pid_c * height * width) + (in_h[:, None] * width + in_w[None, :]).flatten()
    weight_ptr = w_ptr + (pid_c * kernel_size)

    # Load the input data with masking
    in_data = tl.load(in_ptr, mask=(mask_h[:, None] & mask_w[None, :]).flatten(), other=0.0)
    weights = tl.load(weight_ptr, mask=tl.arange(0, kernel_size) < kernel_size, other=0.0)

    # Reshape for computation
    in_data = in_data.reshape(BLOCK_SIZE_H, BLOCK_SIZE_W, kernel_size)
    weights = weights[None, None, :]

    # Perform convolution: element-wise multiplication and sum over kernel size
    out = tl.sum(in_data * weights, axis=2)

    # Write output back
    out_ptr = out_ptr + (pid_c * out_height * out_width) + out_offset
    tl.store(out_ptr, out, mask=tl.arange(0, BLOCK_SIZE_H * BLOCK_SIZE_W) < out_height * out_width)


def triton_depthwise_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, kernel_size: int):
    """
    Wrapper function for the Triton depthwise 2D convolution kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        w (torch.Tensor): Weight tensor of shape (in_channels, kernel_size, 1).
        stride (int): Stride of the convolution.
        padding (int): Padding applied to the input.
        dilation (int): Spacing between kernel elements.
        kernel_size (int): Size of the kernel.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, in_channels, out_height, out_width).
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Ensure out_height and out_width are positive
    assert out_height > 0 and out_width > 0, "Output dimensions must be positive."

    # Create output tensor
    out = torch.empty(batch_size, in_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Define block sizes for the kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16

    # Grid dimensions: (out_height // BLOCK_SIZE_H, out_width // BLOCK_SIZE_W, in_channels)
    grid = lambda meta: (
        (out_height + meta["BLOCK_SIZE_H"] - 1) // meta["BLOCK_SIZE_H"],
        (out_width + meta["BLOCK_SIZE_W"] - 1) // meta["BLOCK_SIZE_W"],
        in_channels,
    )

    # Launch the Triton kernel
    depthwise_conv2d_kernel[grid](
        x, w, out,
        batch_size, in_channels, height, width,
        kernel_size, stride, padding, dilation,
        out_height, out_width,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        # Use the custom Triton kernel instead of nn.Conv2d
        # Initialize the weight tensor
        self.weight = nn.Parameter(torch.randn(in_channels, kernel_size, 1, device='cuda'))

        # Store parameters for use in forward
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_depthwise_conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=self.kernel_size
        )
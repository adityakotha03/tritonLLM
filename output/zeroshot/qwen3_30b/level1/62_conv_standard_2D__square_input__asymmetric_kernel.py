import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X,  # Input pointer
    W,  # Weight pointer
    B,  # Bias pointer
    Out,  # Output pointer
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    out_height,
    out_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread indices
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_h = tl.program_id(2)
    pid_out_w = tl.program_id(3)

    # Calculate output indices
    out_h = pid_out_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_w = pid_out_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Compute input indices with padding
    in_h = out_h * stride_h - pad_h
    in_w = out_w * stride_w - pad_w

    # Get input channel and batch index
    in_c = tl.arange(0, in_channels)
    batch = pid_batch

    # Compute output channel
    out_c = pid_out_c

    # Compute weights indices
    w_h = tl.arange(0, kernel_h)
    w_w = tl.arange(0, kernel_w)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Iterate over input channels and weights
    for ic in range(0, in_channels, BLOCK_SIZE):
        # Load input tile
        in_h_offset = in_h[:, None] + w_h[None, :] * dilation_h
        in_w_offset = in_w[None, :] + w_w[None, :] * dilation_w

        # Mask to handle padding
        h_mask = (in_h_offset >= 0) & (in_h_offset < height)
        w_mask = (in_w_offset >= 0) & (in_w_offset < width)

        # Apply dilation and offset
        in_h_offset = in_h_offset * (h_mask & w_mask)
        in_w_offset = in_w_offset * (h_mask & w_mask)

        # Load input data
        x = tl.load(
            X + batch * in_channels * height * width +
            (ic + in_c[None, None, :]) * height * width +
            in_h_offset[:, :, None] * width + in_w_offset[:, :, None],
            mask=(in_h_offset < height) & (in_w_offset < width),
            other=0.0
        )

        # Load weights
        w = tl.load(
            W + (out_c // groups) * in_channels * kernel_h * kernel_w +
            (ic + in_c[None, None, :]) * kernel_h * kernel_w +
            w_h[None, :, None] * kernel_w + w_w[None, None, :],
            mask=(in_c < in_channels),
            other=0.0
        )

        # Compute dot product
        acc += tl.dot(x, w)

    # Handle bias
    if B is not None:
        bias = tl.load(B + out_c, mask=out_c < out_channels, other=0.0)
        acc += bias

    # Write output
    out_offset = batch * out_channels * out_height * out_width + \
                 out_c * out_height * out_width + \
                 out_h[:, None] * out_width + out_w[None, :]

    tl.store(
        Out + out_offset,
        acc,
        mask=(out_h[:, None] < out_height) & (out_w[None, :] < out_width)
    )


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, padding: int, dilation: int, groups: int):
    """
    Conv2d operation implemented using Triton kernel.

    Args:
        x: Input tensor of shape (batch_size, in_channels, height, width)
        w: Weight tensor of shape (out_channels, in_channels // groups, kernel_h, kernel_w)
        b: Bias tensor of shape (out_channels,) or None
        stride: Stride value (int or tuple)
        padding: Padding value (int or tuple)
        dilation: Dilation value (int or tuple)
        groups: Number of groups for grouped convolution

    Returns:
        Output tensor of shape (batch_size, out_channels, out_height, out_width)
    """
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    w = w.contiguous()
    if b is not None:
        b = b.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = w.shape
    out_height = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Handle stride and dilation as tuples
    stride_h = stride if isinstance(stride, int) else stride[0]
    stride_w = stride if isinstance(stride, int) else stride[1]
    pad_h = padding if isinstance(padding, int) else padding[0]
    pad_w = padding if isinstance(padding, int) else padding[1]
    dilation_h = dilation if isinstance(dilation, int) else dilation[0]
    dilation_w = dilation if isinstance(dilation, int) else dilation[1]

    # Allocate output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Define BLOCK_SIZE for tiling
    BLOCK_SIZE = 32  # Optimized for A100, power of 2

    # Define grid dimensions
    grid = (
        batch_size,
        (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (out_height + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (out_width + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )

    # Launch Triton kernel
    conv2d_kernel[
        grid,
        1024,  # Shared memory
        tl.cuda.Stream(0),
    ](
        x,
        w,
        b,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
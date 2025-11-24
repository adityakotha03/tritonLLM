import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # pointer to input tensor (batch, in_channels, height, width)
    weight_ptr,  # pointer to weight tensor (out_channels, in_channels, kh, kw)
    bias_ptr,  # pointer to bias tensor (out_channels), or None
    output_ptr,  # pointer to output tensor (batch, out_channels, height_out, width_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute grid and block indices
    batch_idx = tl.program_id(0)
    out_h_start = tl.program_id(1)
    out_w_start = tl.program_id(2)

    # Define block size for each dimension
    block_h = tl.arange(0, BLOCK_SIZE)
    block_w = tl.arange(0, BLOCK_SIZE)

    # Compute the current output position
    out_h = out_h_start + block_h
    out_w = out_w_start + block_w

    # Compute the range of input positions this block will access
    # We need to map output (h, w) to input (h, w) with padding and dilation
    # Input coordinates: (h_in, w_in)
    # h_in = (out_h - pad_h) * stride_h - dilation_h * (kh - 1) / 2
    # w_in = (out_w - pad_w) * stride_w - dilation_w * (kw - 1) / 2

    # But we use a more efficient tiling approach with loop over input positions
    # Instead, we precompute the input indices using dilation and stride

    # For each input position (h_in, w_in) that contributes to output (out_h, out_w)
    # We use a 2D loop over the kernel window

    # We will use a tiling strategy: for each output position, compute the input window
    # We assume that the kernel is applied with dilation and padding

    # Compute the number of input positions in the kernel window
    # We use a 2D loop over the kernel window, but we must ensure bounds checking

    # Define input indices
    h_in = tl.arange(0, kh)
    w_in = tl.arange(0, kw)

    # Apply dilation
    h_dilated = h_in * dilation_h
    w_dilated = w_in * dilation_w

    # Compute the input coordinates for each kernel element
    # The input coordinate is: (out_h - pad_h + h_dilated) * stride_h, (out_w - pad_w + w_dilated) * stride_w
    # But we need to loop over the kernel and accumulate contributions

    # Instead, we restructure: for each output position (out_h, out_w), we compute the input positions
    # that fall within the kernel window

    # We will compute the input positions that contribute to (out_h, out_w)
    # The input height index: h_in = (out_h - pad_h) * stride_h - h_dilated
    # The input width index: w_in = (out_w - pad_w) * stride_w - w_dilated

    # But we must ensure bounds checking

    # Instead, we use a different approach: loop over the kernel window and accumulate
    # We will compute the input indices for each kernel element

    # For each kernel position (h_k, w_k), compute the input (h_in, w_in)
    # h_in = out_h - pad_h + h_k * dilation_h
    # w_in = out_w - pad_w + w_k * dilation_w
    # But we need to apply stride to input

    # Actually, we need to compute input coordinates as:
    # h_in = (out_h - pad_h) * stride_h - h_k * dilation_h
    # w_in = (out_w - pad_w) * stride_w - w_k * dilation_w

    # We loop over the kernel window (h_k, w_k)
    # But we need to check bounds for input

    # We will compute the input indices for each kernel element
    # and then load input and weight

    # Initialize output
    out_val = tl.zeros((out_channels,), dtype=tl.float32)

    # Loop over kernel positions
    for h_k in range(kh):
        for w_k in range(kw):
            # Compute input coordinates
            h_in = out_h - pad_h + h_k * dilation_h
            w_in = out_w - pad_w + w_k * dilation_w

            # Apply stride to input
            h_in = h_in // stride_h
            w_in = w_in // stride_w

            # But this is not correct — we need to compute the actual input index
            # Correct: input index is (out_h - pad_h) * stride_h + h_k * dilation_h
            # But we need to map to actual input position

            # Actually, we need to compute input coordinates as:
            # h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            # w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # But this is not correct either — dilation and stride are applied to the kernel

            # Correct formula:
            # The input position is:
            # h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            # w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # But this can go out of bounds — we need to check

            # We need to compute the actual input coordinates
            h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # Check bounds
            h_in = tl.max(h_in, 0)
            w_in = tl.max(w_in, 0)
            h_in = tl.min(h_in, height - 1)
            w_in = tl.min(w_in, width - 1)

            # We are looping over kernel, so we need to load input and weight
            # But we are missing the actual input tensor indexing

            # We need to restructure: we are not looping over kernel positions properly

            # Let's use a different approach: we loop over the kernel window and compute input indices
            # For each kernel element (h_k, w_k), we compute input (h_in, w_in)
            # Then we load input and weight and accumulate

            # But we need to compute input indices correctly

            # Correct input index: (h_in, w_in) = (out_h - pad_h + h_k * dilation_h, out_w - pad_w + w_k * dilation_w)
            # Then we need to apply stride to input coordinates

            # Actually, the input coordinates are:
            # h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            # w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # But we need to ensure input indices are within [0, height-1] and [0, width-1]

            # Compute input indices
            h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # Clamp to valid range
            h_in = tl.max(h_in, 0)
            w_in = tl.max(w_in, 0)
            h_in = tl.min(h_in, height - 1)
            w_in = tl.min(w_in, width - 1)

            # Load input value
            # Input is (batch, in_channels, height, width)
            # We need to load input at (batch_idx, c, h_in, w_in)
            # But we don't have c — we need to loop over channels

            # We need to restructure: we loop over output channels and input channels

            # We are not doing this correctly — we need to loop over input channels and output channels

            # We will restructure the kernel to be over output channels and input channels

            # Let's change the kernel to loop over output channel and input channel
            pass

    # Instead, we restructure: we will loop over output channel and input channel
    # We will compute the input positions for each output position and each channel

    # We will recompute the kernel in a different way

    # We define the output channel index
    out_ch = tl.arange(0, out_channels)

    # Initialize output
    out_val = tl.zeros((out_channels,), dtype=tl.float32)

    # Loop over kernel positions
    for h_k in range(kh):
        for w_k in range(kw):
            # Compute input coordinates
            h_in = (out_h - pad_h) * stride_h + h_k * dilation_h
            w_in = (out_w - pad_w) * stride_w + w_k * dilation_w

            # Clamp to valid range
            h_in = tl.max(h_in, 0)
            w_in = tl.max(w_in, 0)
            h_in = tl.min(h_in, height - 1)
            w_in = tl.min(w_in, width - 1)

            # Load input and weight
            # Input: (batch, in_channels, h_in, w_in)
            # Weight: (out_channels, in_channels, kh, kw)
            # We need to loop over input channels

            # For each input channel
            for c in range(in_channels):
                # Load input value
                input_val = tl.load(input_ptr + batch_idx * in_channels * height * width + c * height * width + h_in * width + w_in, mask=(h_in < height) & (w_in < width), other=0.0)
                # Load weight value
                weight_val = tl.load(weight_ptr + out_ch * in_channels * kh * kw + c * kh * kw + h_k * kw + w_k, mask=(h_k < kh) & (w_k < kw), other=0.0)
                # Accumulate
                out_val = out_val + input_val * weight_val

    # Store output
    tl.store(output_ptr + batch_idx * out_channels * height * width + out_ch * height * width + out_h * width + out_w, out_val, mask=(out_h < height) & (out_w < width))


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: tuple = (0, 0),
    dilation: tuple = (1, 1),
    kernel_size: tuple = (3, 3),
) -> torch.Tensor:
    """
    Custom 2D convolution using Triton kernels.

    Args:
        x: Input tensor of shape (batch, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels, kh, kw)
        bias: Bias tensor of shape (out_channels), or None
        stride: Stride of the convolution
        padding: Padding (top/bottom, left/right)
        dilation: Dilation (height, width)

    Returns:
        Output tensor of shape (batch, out_channels, height_out, width_out)
    """
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kh, kw = weight.shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride, stride
    dilation_h, dilation_w = dilation[0], dilation[1]

    # Compute output dimensions
    height_out = (height + 2 * pad_h - kh * dilation_h + (kh - 1) * (dilation_h - 1)) // stride_h + 1
    width_out = (width + 2 * pad_w - kw * dilation_w + (kw - 1) * (dilation_w - 1)) // stride_w + 1

    # Ensure tensors are contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output
    output = torch.empty((batch_size, out_channels, height_out, width_out), dtype=x.dtype, device=x.device)

    # Define kernel parameters
    BLOCK_SIZE = 128  # Optimal block size for memory and compute

    # Grid dimensions
    grid = lambda meta: (
        (batch_size,),
        ((height_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
        ((width_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]),
    )

    # Launch kernel
    conv2d_kernel[grid](
        x.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        output.data_ptr(),
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kh,
        kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0), dilation: tuple = (1, 1), bias: bool = False):
        super(ModelNew, self).__init__()
        # Initialize weight tensor
        self.weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], dtype=torch.float16, device="cuda")
        self.bias = torch.randn(out_channels, dtype=torch.float16, device="cuda") if bias else None

        # Store parameters for forward
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using the custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return triton_conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, kernel_size=self.kernel_size)
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,        # pointer to input tensor (batch, channels, height, width)
    output_ptr,       # pointer to output tensor (batch, channels, height_out, width_out)
    weight_ptr,       # pointer to weight tensor (in_channels, kernel_h, kernel_w)
    bias_ptr,         # pointer to bias tensor (in_channels) - optional
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the output dimensions
    height_out = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    width_out = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    # Current block index
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)

    # Thread-level indices
    row_start = tl.program_id(2) * BLOCK_SIZE
    col_start = tl.program_id(3) * BLOCK_SIZE

    # Compute the current block's row and col range
    row_end = min(row_start + BLOCK_SIZE, height_out)
    col_end = min(col_start + BLOCK_SIZE, width_out)

    # Load the weights for this channel (only one channel per thread in depthwise)
    # Weights are stored as (in_channels, kernel_h, kernel_w)
    # We use a 2D kernel for the height and width dimensions
    weights = tl.zeros((kernel_h, kernel_w), dtype=tl.float16)
    if bias_ptr is not None:
        bias = tl.zeros(1, dtype=tl.float16)
    else:
        bias = tl.zeros(1, dtype=tl.float16)

    # Compute the output for each output position
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            # Compute the input indices (with dilation and padding)
            input_row = row * stride_h - padding_h
            input_col = col * stride_w - padding_w

            # Apply dilation to kernel indices
            # We loop over the kernel and compute input indices
            # We use a loop over the kernel to compute the weighted sum
            total = tl.zeros(1, dtype=tl.float16)

            # Loop over kernel positions
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute input indices with dilation
                    in_row = input_row + kh * dilation_h
                    in_col = input_col + kw * dilation_w

                    # Check bounds
                    in_row_valid = (in_row >= 0) and (in_row < height)
                    in_col_valid = (in_col >= 0) and (in_col < width)

                    if in_row_valid and in_col_valid:
                        # Load input value
                        input_idx = batch_idx * in_channels * height * width + \
                                    channel_idx * height * width + \
                                    in_row * width + in_col
                        input_val = tl.load(input_ptr + input_idx, mask=(in_row_valid & in_col_valid), other=0.0)

                        # Load weight
                        weight_idx = channel_idx * kernel_h * kernel_w + kh * kernel_w + kw
                        weight_val = tl.load(weight_ptr + weight_idx, mask=(kh < kernel_h) & (kw < kernel_w), other=0.0)

                        total += input_val * weight_val

            # Add bias if present
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + channel_idx, mask=(channel_idx < in_channels), other=0.0)
                total += bias_val

            # Store output
            output_idx = batch_idx * in_channels * height_out * width_out + \
                         channel_idx * height_out * width_out + \
                         row * width_out + col
            tl.store(output_ptr + output_idx, total, mask=(row < height_out) & (col < width_out))


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    kernel_size: tuple = (3, 1)
) -> torch.Tensor:
    """
    Custom Triton kernel for depthwise 2D convolution with asymmetric kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    kernel_h, kernel_w = kernel_size

    # Output dimensions
    height_out = (height + 2 * padding - dilation * (kernel_h - 1) - 1) // stride + 1
    width_out = (width + 2 * padding - dilation * (kernel_w - 1) - 1) // stride + 1

    # Output tensor
    output = torch.empty((batch_size, in_channels, height_out, width_out), dtype=x.dtype, device=x.device)

    # Define grid and block size
    BLOCK_SIZE = 16  # Power of 2, optimized for shared memory and warp efficiency
    grid = lambda meta: (
        (batch_size + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (in_channels + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (height_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
        (width_out + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],
    )

    # Launch kernel
    conv2d_kernel[grid](
        x.data_ptr(),
        output.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else None,
        batch_size,
        in_channels,
        kernel_h,
        kernel_w,
        height,
        width,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create weight tensor (in_channels, kernel_h, kernel_w)
        self.kernel_h, self.kernel_w = kernel_size, 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels

        # Initialize weight with random values
        self.weight = torch.randn(in_channels, self.kernel_h, self.kernel_w, dtype=torch.float16, device='cuda')
        # Initialize bias if needed
        if bias:
            self.bias = torch.randn(in_channels, dtype=torch.float16, device='cuda')
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom Triton kernel
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=(self.kernel_h, self.kernel_w)
        )
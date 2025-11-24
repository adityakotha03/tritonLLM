import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    bias_ptr,
    batch_size, in_channels, out_channels,
    input_height, input_width,
    output_height, output_width,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    has_bias: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)
    pid_m = tl.program_id(3)

    # Calculate output spatial position
    oh = pid_oh
    ow = pid_ow

    # Pointers for output
    output_offset = pid_b * out_channels * output_height * output_width + \
                    pid_m * output_height * output_width + oh * output_width + ow
    output_mask = (pid_b < batch_size) & (pid_m < out_channels) & (oh < output_height) & (ow < output_width)
    
    # Accumulate result for this output element
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Loop over input channels and kernel space
    for ic in range(0, in_channels):
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                # Calculate input position
                ih = oh * stride_h - padding_h + kh * dilation_h
                iw = ow * stride_w - padding_w + kw * dilation_w

                # Check bounds
                ih_in_bounds = (ih >= 0) & (ih < input_height)
                iw_in_bounds = (iw >= 0) & (iw < input_width)

                # Input pointer and mask
                input_offset = pid_b * in_channels * input_height * input_width + \
                               ic * input_height * input_width + ih * input_width + iw
                input_mask = (pid_b < batch_size) & (ic < in_channels) & ih_in_bounds & iw_in_bounds
                input_val = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

                # Weight pointer
                weight_offset = pid_m * in_channels * kernel_h * kernel_w + \
                                ic * kernel_h * kernel_w + kh * kernel_w + kw
                weight_val = tl.load(weight_ptr + weight_offset)

                # Multiply and accumulate
                acc += input_val.to(tl.float32) * weight_val.to(tl.float32)

    # Add bias if present
    if has_bias:
        bias_val = tl.load(bias_ptr + pid_m) if has_bias else 0.0
        acc += bias_val

    # Store output
    output_val = acc.to(tl.float16)
    tl.store(output_ptr + output_offset, output_val, mask=output_mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    dilation: tuple,
    groups: int
):
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output spatial dimensions
    output_height = (input_height - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_h - 1) + output_padding[0] + 1
    output_width = (input_width - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_w - 1) + output_padding[1] + 1

    # Create output tensor
    out = torch.zeros(batch_size, out_channels, output_height, output_width, device=x.device, dtype=torch.float16)

    # Flatten weight for ease of indexing
    weight = weight.view(out_channels, in_channels, kernel_h, kernel_w)

    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # Grid: (batch_size, output_height, output_width, out_channels)
    grid = (batch_size, output_height, output_width, out_channels)

    # Launch kernel
    conv_transpose2d_kernel[grid](
        x, weight, out, bias,
        batch_size, in_channels, out_channels,
        input_height, input_width,
        output_height, output_width,
        kernel_h, kernel_w,
        stride[0], stride[1],
        padding[0], padding[1],
        dilation[0], dilation[1],
        bias is not None,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = (1, 1)
        self.groups = 1
        self.output_padding = (0, 0)

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Transpose weight layout for direct use in kernel
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else None

        # Cast input to float16 for faster computation on Tensor Cores
        x = x.to(torch.float16)

        # Call Triton kernel
        out = triton_conv_transpose2d(
            x, weight, bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out
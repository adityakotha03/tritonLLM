import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    # Compute the output position
    out_h = pid // (out_channels * width_in)
    out_w = (pid % (out_channels * width_in)) // out_channels
    out_c = pid % out_channels

    # Compute the input position
    in_h = out_h * stride - padding
    in_w = out_w * stride - padding

    # Iterate over the output block
    for o_h in range(kernel_size):
        for o_w in range(kernel_size):
            # Compute the input position for this kernel element
            in_h_current = in_h + o_h
            in_w_current = in_w + o_w

            # Iterate over input channels
            for in_c in range(in_channels):
                # Compute the input offset
                in_offset = (
                    batch_size * in_channels * height_in * width_in * out_c +
                    in_c * height_in * width_in +
                    in_h_current * width_in + in_w_current
                )

                # Load input value
                input_val = tl.load(input_ptr + in_offset, mask=in_offset < (batch_size * in_channels * height_in * width_in * out_channels), other=0.0)

                # Load weight value
                weight_offset = (
                    in_channels * out_channels * kernel_size * kernel_size +
                    in_c * out_channels * kernel_size * kernel_size +
                    out_c * kernel_size * kernel_size +
                    o_h * kernel_size + o_w
                )
                weight_val = tl.load(weight_ptr + weight_offset, mask=weight_offset < (in_channels * out_channels * kernel_size * kernel_size), other=0.0)

                # Accumulate the result
                output_offset = (
                    batch_size * out_channels * height_in * width_in +
                    out_c * height_in * width_in +
                    out_h * width_in + out_w
                )
                output_val = tl.load(output_ptr + output_offset, mask=output_offset < (batch_size * out_channels * height_in * width_in), other=0.0)
                output_val += input_val * weight_val
                tl.store(output_ptr + output_offset, output_val, mask=output_offset < (batch_size * out_channels * height_in * width_in))


def triton_depthwise_conv(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, batch_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
    # Ensure input and weight are on GPU
    assert input.is_cuda and weight.is_cuda, "Input and weight must be on CUDA."

    # Prepare output tensor
    output = torch.empty(
        batch_size,
        out_channels,
        (height_in + 2 * padding - kernel_size) // stride + 1,
        (width_in + 2 * padding - kernel_size) // stride + 1,
        device=input.device,
        dtype=input.dtype
    )

    # Compute the number of output elements
    n_elements = output.numel()

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    depthwise_conv_kernel[grid](input, weight, output, batch_size, in_channels, out_channels, kernel_size, stride, padding, BLOCK_SIZE=128)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform depthwise convolution using Triton kernel
        output = triton_depthwise_conv(
            x,
            self.weight,
            self.bias,
            x.size(0),
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding
        )
        return output
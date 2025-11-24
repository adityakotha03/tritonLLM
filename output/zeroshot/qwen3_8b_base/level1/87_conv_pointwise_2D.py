import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID
    pid = tl.program_id(0)
    # Compute the offset for the current program
    offset = pid * BLOCK_SIZE
    # Compute the index in the output tensor
    out_idx = offset + tl.arange(0, BLOCK_SIZE)
    # Compute the corresponding input indices
    in_idx = out_idx
    # Compute the weight indices
    weight_idx = tl.arange(0, BLOCK_SIZE)
    # Load input values
    input_val = tl.load(input_ptr + in_idx, mask=out_idx < (batch_size * in_channels * height * width), other=0.0)
    # Load weight values
    weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < out_channels, other=0.0)
    # Compute the product
    output_val = input_val * weight_val
    # Store the result
    tl.store(output_ptr + out_idx, output_val, mask=out_idx < (batch_size * out_channels * height * width))


def triton_conv1d(input: torch.Tensor, weight: torch.Tensor):
    """
    This function wraps the Triton kernel call for the 1D convolution.
    """
    assert input.is_cuda and weight.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()

    # Prepare output tensor
    output = torch.empty((input.size(0), weight.size(0), input.size(2)), device=input.device, dtype=input.dtype)

    # Parameters
    batch_size = input.size(0)
    in_channels = input.size(1)
    out_channels = weight.size(0)
    height = input.size(2)
    width = input.size(3)

    # Choose block size
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    num_blocks = (batch_size * in_channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    conv1d_kernel[ num_blocks ](input, weight, output, batch_size, in_channels, out_channels, height, width, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, 1, 1).cuda())
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_channels, 1, 1).cuda())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the pointwise 2D convolution using a custom Triton kernel.
        """
        # Perform the convolution using the Triton kernel
        output = triton_conv1d(x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output
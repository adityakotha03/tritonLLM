import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    kernel_size,  # Kernel size (assumed square)
    num_channels,  # Number of input channels
    out_channels,  # Number of output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    pid = tl.program_id(0)
    # Get the thread ID within the block
    tid = tl.program_id(1)
    # Each block processes a block of output data
    # Compute the output position
    out_h = pid // (out_channels // BLOCK_SIZE)
    out_w = pid % (out_channels // BLOCK_SIZE)
    # Compute the output index
    out_idx = out_h * out_channels * output_stride_0 + out_w * output_stride_1
    # Compute the input and weight indices
    for i in range(BLOCK_SIZE):
        # Compute the input position
        input_idx = out_idx + i * output_stride_2
        # Compute the weight position
        weight_idx = out_w * weight_stride_0 + i * weight_stride_1
        # Load input and weight
        input_val = tl.load(input_ptr + input_idx, mask=input_idx < input_ptr.shape[0], other=0.0)
        weight_val = tl.load(weight_ptr + weight_idx, mask=weight_idx < weight_ptr.shape[0], other=0.0)
        # Perform the convolution
        output_val = input_val * weight_val
        # Store the result
        tl.store(output_ptr + out_idx + i * output_stride_2, output_val, mask=out_idx + i * output_stride_2 < output_ptr.shape[0])


def triton_conv2d(input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and output.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    output = output.contiguous()

    # Calculate the grid size
    num_output_channels = weight.shape[0]
    num_blocks = (num_output_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, 1)

    # Launch the Triton kernel
    conv2d_kernel[grid](input, weight, output, input.stride(0), input.stride(1), input.stride(2), input.stride(3),
                        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                        kernel_size=11, num_channels=3, out_channels=num_output_channels, BLOCK_SIZE=128)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(96, 3, 11, 11).cuda())

    def forward(self, x):
        # Allocate output tensor
        output = torch.empty((x.size(0), 96, x.size(2) - 11 + 1, x.size(3) - 11 + 1), device=x.device)
        # Call the Triton kernel
        triton_conv2d(x, self.weight, output)
        return output
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    input_ptr,  # Pointer to input tensor
    weight_ptr,  # Pointer to weight tensor
    output_ptr,  # Pointer to output tensor
    input_shape,  # (batch, in_channels, width, height, depth)
    kernel_size,  # (kW, kH, kD)
    stride,  # (sW, sH, sD)
    padding,  # (pW, pH, pD)
    dilation,  # (dW, dH, dD)
    BLOCK_SIZE: tl.constexpr,
):
    # Extract input dimensions
    batch, in_channels, width, height, depth = input_shape
    kW, kH, kD = kernel_size
    sW, sH, sD = stride
    pW, pH, pD = padding
    dW, dH, dD = dilation

    # Compute output dimensions
    out_width = (width + 2 * pW - dW * (kW - 1) - kW) // sW + 1
    out_height = (height + 2 * pH - dH * (kH - 1) - kH) // sH + 1
    out_depth = (depth + 2 * pD - dD * (kD - 1) - kD) // sD + 1

    # Determine the current block's position
    batch_idx = tl.program_id(0)
    out_depth_idx = tl.program_id(1)
    out_height_idx = tl.program_id(2)
    out_width_idx = tl.program_id(3)

    # Compute the output position
    out_depth_start = out_depth_idx * sD
    out_height_start = out_height_idx * sH
    out_width_start = out_width_idx * sW

    # Compute the input position
    in_depth_start = out_depth_start - pD + dD * (kD - 1)
    in_height_start = out_height_start - pH + dH * (kH - 1)
    in_width_start = out_width_start - pW + dW * (kW - 1)

    # Compute the input range for this block
    in_depth_range = tl.arange(0, kW)
    in_height_range = tl.arange(0, kH)
    in_width_range = tl.arange(0, kW)

    # Adjust for dilation
    in_depth_range = in_depth_range * dD
    in_height_range = in_height_range * dH
    in_width_range = in_width_range * dW

    # Compute the offset for each input position
    in_depth_offsets = in_depth_start + in_depth_range
    in_height_offsets = in_height_start + in_height_range
    in_width_offsets = in_width_start + in_width_range

    # Compute the input indices
    in_idx = (batch_idx * in_channels * width * height * depth +
              tl.arange(0, BLOCK_SIZE) // (width * height * depth) * in_channels +
              tl.arange(0, BLOCK_SIZE) % (width * height * depth) // (height * depth) * width +
              tl.arange(0, BLOCK_SIZE) % (height * depth) // depth * height +
              tl.arange(0, BLOCK_SIZE) % depth)

    # Compute the weight indices
    weight_idx = (tl.arange(0, BLOCK_SIZE) // (width * height * depth) * in_channels +
                  tl.arange(0, BLOCK_SIZE) % (width * height * depth) // (height * depth) * width +
                  tl.arange(0, BLOCK_SIZE) % (height * depth) // depth * height +
                  tl.arange(0, BLOCK_SIZE) % depth)

    # Load input and weight
    input = tl.load(input_ptr + in_idx, mask=in_idx < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4], other=0.0)
    weight = tl.load(weight_ptr + weight_idx, mask=weight_idx < in_channels * width * height * depth * depth, other=0.0)

    # Compute the output
    output = tl.sum(input * weight, axis=0)

    # Store the result
    out_idx = (batch_idx * out_channels * out_width * out_height * out_depth +
               tl.arange(0, BLOCK_SIZE) // (out_width * out_height * out_depth) * out_channels +
               tl.arange(0, BLOCK_SIZE) % (out_width * out_height * out_depth) // (out_height * out_depth) * out_width +
               tl.arange(0, BLOCK_SIZE) % (out_height * out_depth) // out_depth * out_height +
               tl.arange(0, BLOCK_SIZE) % out_depth)
    tl.store(output_ptr + out_idx, output)


def triton_conv3d(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert input.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    # Compute output dimensions
    batch, in_channels, width, height, depth = input.shape
    kW, kH, kD = weight.shape[2:]
    sW, sH, sD = stride
    pW, pH, pD = padding
    dW, dH, dD = dilation

    out_width = (width + 2 * pW - dW * (kW - 1) - kW) // sW + 1
    out_height = (height + 2 * pH - dH * (kH - 1) - kH) // sH + 1
    out_depth = (depth + 2 * pD - dD * (kD - 1) - kD) // sD + 1

    output = torch.empty((batch, weight.shape[0], out_width, out_height, out_depth), device=input.device, dtype=input.dtype)

    # Determine the number of blocks needed
    BLOCK_SIZE = 128
    num_blocks = (batch * out_channels * out_width * out_height * out_depth + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the Triton kernel
    grid = (num_blocks,)
    conv3d_kernel[grid](input, weight, output, input.shape, weight.shape[2:], stride, padding, dilation, BLOCK_SIZE=BLOCK_SIZE)
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *kernel_size, device='cuda', dtype=torch.float16))
        self.bias = nn.Parameter(torch.empty(out_channels, device='cuda', dtype=torch.float16) if bias else None)

        # Initialize weights using Kaiming initialization
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        if bias:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        # Ensure input is on GPU
        x = x.to('cuda')

        # Compute output dimensions
        batch, in_channels, width, height, depth = x.shape
        kW, kH, kD = self.kernel_size
        sW, sH, sD = self.stride, self.stride, self.stride
        pW, pH, pD = self.padding, self.padding, self.padding
        dW, dH, dD = self.dilation, self.dilation, self.dilation

        out_width = (width + 2 * pW - dW * (kW - 1) - kW) // sW + 1
        out_height = (height + 2 * pH - dH * (kH - 1) - kH) // sH + 1
        out_depth = (depth + 2 * pD - dD * (kD - 1) - kD) // sD + 1

        # Apply convolution
        output = triton_conv3d(x, self.weight, self.bias, (sW, sH, sD), (pW, pH, pD), (dW, dH, dD))

        return output
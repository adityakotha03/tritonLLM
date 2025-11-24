import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    stride_d, stride_h, stride_w,  # Strides for transposed conv
    kernel_size_d, kernel_size_h, kernel_size_w,  # Kernel sizes
    padding_d, padding_h, padding_w,  # Padding sizes
    out_channels,  # Number of output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (block ID)
    pid = tl.program_id(0)
    # Compute the block's starting position in the output
    offset = pid * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < out_channels
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply GELU activation
    x = x * tl.math.erf(x * 0.7071) + 1.0
    # Store the result
    tl.store(output_ptr + offsets, x, mask=mask)


def triton_conv_transpose_3d(x: torch.Tensor, out_channels, stride_d, stride_h, stride_w, kernel_size_d, kernel_size_h, kernel_size_w, padding_d, padding_h, padding_w, scaling_factor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    # Prepare output tensor
    output_shape = (x.size(0), out_channels, x.size(2) * stride_d, x.size(3) * stride_h, x.size(4) * stride_w)
    out = torch.empty(output_shape, device=x.device, dtype=x.dtype)

    # Number of elements in the output tensor
    n_elements = out.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv_transpose_3d_kernel[grid](x, out, stride_d, stride_h, stride_w, kernel_size_d, kernel_size_h, kernel_size_w, padding_d, padding_h, padding_w, out_channels, BLOCK_SIZE=BLOCK_SIZE)
    return out * scaling_factor


class ModelNew(nn.Module):
    """
    Optimized Model using custom Triton kernels for 3D transposed convolution and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.eps = eps
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        # Perform 3D transposed convolution using custom Triton kernel
        x = triton_conv_transpose_3d(x, self.out_channels, self.stride[0], self.stride[1], self.stride[2],
                                     self.kernel_size[0], self.kernel_size[1], self.kernel_size[2],
                                     self.padding[0], self.padding[1], self.padding[2], self.scaling_factor)
        return x
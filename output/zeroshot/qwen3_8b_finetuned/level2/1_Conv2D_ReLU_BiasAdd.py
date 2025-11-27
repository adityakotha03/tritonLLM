import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_bias_kernel(
    output_ptr,  # Pointer to the flattened convolution output (B*C_out*H*W)
    bias_ptr,    # Pointer to the bias vector (C_out)
    out_ptr,     # Pointer to the output after bias addition
    num_elements,  # Total number of elements in the flattened output
    C_out: tl.constexpr,  # Number of output channels (bias length)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a contiguous block of the flattened output
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xoffset = xoffset + tl.arange(0, BLOCK_SIZE)[:, None]
    xoffset = xoffset % C_out  # Compute channel index via modulo
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[None, :]
    xindex = xindex % num_elements  # Ensure index stays within bounds

    # Load the original value (convolution output) and the corresponding bias
    x = tl.load(output_ptr + xindex, mask=xindex < num_elements, other=0.0)
    y = tl.load(bias_ptr + xoffset, mask=xoffset < C_out, other=0.0)

    # Perform elementwise addition
    tmp0 = x + y

    # Store the result back
    tl.store(out_ptr + xindex, tmp0, mask=xindex < num_elements)


def triton_add_bias(output, bias):
    """
    Adds a 1D bias tensor to a 4D convolution output tensor using a Triton kernel.
    The bias is broadcast across the spatial dimensions of the output.
    """
    assert output.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert output.dim() == 4 and bias.dim() == 1, "Output must be 4D, bias 1D."
    assert bias.shape[0] == output.shape[1], "Bias length must match output channels."

    # Flatten the output to a 1D view for contiguous memory access
    output = output.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(output)

    num_elements = output.numel()
    C_out = output.shape[1]
    BLOCK_SIZE = 128  # Tunable power-of-two block size

    # Compute grid: ceil(num_elements / BLOCK_SIZE)
    grid = lambda meta: ((num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_bias_kernel[grid](output, bias, out, num_elements, C_out=C_out, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized version of the original model that replaces the final addition
    with a Triton kernel that adds a 1D bias tensor to the convolution output
    while preserving the convolution and ReLU operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))  # (out_channels, 1, 1)

    def forward(self, x):
        x = self.conv(x)  # (B, C_out, H, W)
        x = torch.relu(x)  # Apply ReLU
        x = triton_add_bias(x, self.bias)  # Add bias (broadcast across spatial dimensions)
        return x
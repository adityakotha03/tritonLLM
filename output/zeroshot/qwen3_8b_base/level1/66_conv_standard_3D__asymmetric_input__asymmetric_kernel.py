import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    stride_d, stride_h, stride_w,  # Stride values
    padding_d, padding_h, padding_w,  # Padding values
    dilation_d, dilation_h, dilation_w,  # Dilation values
    out_channels,  # Number of output channels
    in_channels,  # Number of input channels
    kernel_d, kernel_h, kernel_w,  # Kernel size
    BLOCK_SIZE: tl.constexpr,
):
    # Get the thread index
    pid = tl.program_id(0)
    # Get the block index in the output
    block_d = pid // (BLOCK_SIZE // stride_d)
    block_h = (pid // BLOCK_SIZE) % (BLOCK_SIZE // stride_h)
    block_w = pid % (BLOCK_SIZE // stride_w)
    # Compute the offset in the output
    offset_d = block_d * stride_d
    offset_h = block_h * stride_h
    offset_w = block_w * stride_w
    # Compute the start and end indices in the input
    start_d = offset_d - padding_d
    start_h = offset_h - padding_h
    start_w = offset_w - padding_w
    end_d = start_d + kernel_d
    end_h = start_h + kernel_h
    end_w = start_w + kernel_w
    # Compute the input indices for each channel
    for c in range(in_channels):
        # Compute the input offset for this channel
        input_offset = c * (depth * height * width)
        # Compute the output offset for this channel
        out_offset = (block_d * stride_d) * (height * width) * out_channels + (block_h * stride_h) * width * out_channels + (block_w * stride_w) * out_channels + c
        # Iterate over the kernel
        for kd in range(kernel_d):
            for kh in range(kernel_h):
                for kw in range(kernel_w):
                    # Compute the input index
                    input_idx = (start_d + kd) * height * width + (start_h + kh) * width + (start_w + kw)
                    input_idx += input_offset
                    # Load the input value
                    x = tl.load(x_ptr + input_idx, mask=input_idx < x.size(0), other=0.0)
                    # Load the weight value
                    weight_idx = (c * kernel_d * kernel_h * kernel_w) + (kd * kernel_h * kernel_w) + (kh * kernel_w) + kw
                    weight = tl.load(w_ptr + weight_idx, mask=weight_idx < w.size(0), other=0.0)
                    # Multiply and accumulate
                    out = tl.load(out_ptr + out_offset, mask=out_offset < out.size(0), other=0.0)
                    out += x * weight
                    tl.store(out_ptr + out_offset, out, mask=out_offset < out.size(0))
    # Return the output
    return out


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    # Prepare output tensor
    out_channels = w.size(0)
    in_channels = w.size(1)
    kernel_d = w.size(2)
    kernel_h = w.size(3)
    kernel_w = w.size(4)
    depth = x.size(2)
    height = x.size(3)
    width = x.size(4)
    out_depth = (depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) // stride_d + 1
    out_height = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    out = torch.zeros((x.size(0), out_channels, out_depth, out_height, out_width), device=x.device)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    conv3d_kernel[grid](x, w, out, stride_d, stride_h, stride_w, padding_d, padding_h, padding_w, dilation_d, dilation_h, dilation_w, out_channels, in_channels, kernel_d, kernel_h, kernel_w, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Apply the Triton-based convolution
        conv_out = triton_conv3d(
            x,
            self.weight,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2]
        )
        if self.bias is not None:
            conv_out += self.bias.view(1, -1, 1, 1, 1)
        return conv_out
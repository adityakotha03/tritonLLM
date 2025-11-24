import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor (batch, in_channels, height, width)
    w_ptr,  # Pointer to weight tensor (out_channels, in_channels // groups, kernel_h, kernel_w)
    bias_ptr,  # Pointer to bias tensor (out_channels,)
    out_ptr,  # Pointer to output tensor (batch, out_channels, height, width)
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    kernel_h,  # Height of kernel
    kernel_w,  # Width of kernel
    stride_h,  # Stride in height direction
    stride_w,  # Stride in width direction
    padding_h,  # Padding in height direction
    padding_w,  # Padding in width direction
    dilation_h,  # Dilation in height direction
    dilation_w,  # Dilation in width direction
    groups,  # Number of groups
    BLOCK_SIZE: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
):
    # Each thread handles one output element
    # Compute the output position (out_h, out_w)
    out_h = tl.program_id(0)
    out_w = tl.program_id(1)
    # Compute the input position (in_h, in_w)
    in_h = out_h * stride_h - padding_h
    in_w = out_w * stride_w - padding_w
    # Compute the number of threads per block
    num_threads = tl.num_programs(0) * tl.num_programs(1)
    # Compute the number of blocks per output
    num_blocks_h = (tl.num_programs(0) + 1) // 2
    num_blocks_w = (tl.num_programs(1) + 1) // 2
    # Compute the output shape
    out_h_total = (in_channels + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w_total = (in_channels + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    # Check if current output position is valid
    if out_h >= out_h_total or out_w >= out_w_total:
        return
    # Compute the input shape
    in_h_total = in_channels + 2 * padding_h
    in_w_total = in_channels + 2 * padding_w
    # Compute the input position for each channel group
    group_size = in_channels // groups
    # Iterate over channel groups
    for g in range(groups):
        # Compute the input channel start and end
        in_ch_start = g * group_size
        in_ch_end = (g + 1) * group_size
        # Iterate over output channels
        for out_ch in range(out_channels):
            # Compute the weight offset
            w_offset = out_ch * in_channels * kernel_h * kernel_w + in_ch_start * kernel_h * kernel_w
            # Compute the bias offset
            bias_offset = out_ch
            # Compute the input offset
            in_offset = (in_ch_start + in_h * in_channels + in_w) * in_channels * in_channels
            # Compute the output offset
            out_offset = (out_h * out_channels + out_ch) * out_channels * out_h_total * out_w_total + out_w
            # Initialize accumulator
            acc = tl.zeros((KERNEL_H, KERNEL_W), dtype=tl.float32)
            # Iterate over kernel height
            for kh in range(KERNEL_H):
                # Compute the input height position
                in_h_k = in_h + kh * dilation_h
                # Check if input height is within bounds
                if in_h_k < 0 or in_h_k >= in_h_total:
                    continue
                # Iterate over kernel width
                for kw in range(KERNEL_W):
                    # Compute the input width position
                    in_w_k = in_w + kw * dilation_w
                    # Check if input width is within bounds
                    if in_w_k < 0 or in_w_k >= in_w_total:
                        continue
                    # Compute the input offset
                    in_offset_k = (in_ch_start + in_h_k * in_channels + in_w_k) * in_channels
                    # Load input value
                    x = tl.load(x_ptr + in_offset_k + in_ch_start, mask=in_offset_k + in_ch_start < x_ptr + in_offset_k + in_ch_start + in_channels, other=0.0)
                    # Load weight value
                    w = tl.load(w_ptr + w_offset + kh * kernel_w + kw, mask=w_offset + kh * kernel_w + kw < w_ptr + w_offset + kh * kernel_w + kw + kernel_h * kernel_w, other=0.0)
                    # Multiply and accumulate
                    acc[kh, kw] += x * w
            # Add bias
            if bias_ptr is not None:
                acc += tl.load(bias_ptr + bias_offset, mask=bias_offset < bias_ptr + bias_offset + out_channels, other=0.0)
            # Store result
            tl.store(out_ptr + out_offset, acc, mask=out_offset < out_ptr + out_offset + out_channels * out_h_total * out_w_total, other=0.0)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, out_channels: int, kernel_size: tuple, stride: int, padding: int, dilation: int, groups: int):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and w.is_cuda and (bias.is_cuda if bias is not None else True), "Tensors must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    # Prepare output tensor
    out = torch.empty((x.size(0), out_channels, x.size(2), x.size(3)), dtype=x.dtype, device=x.device)
    # Compute the output shape
    out_h = (x.size(2) + 2 * padding - dilation * (kernel_size[0] - 1) - 1) // stride + 1
    out_w = (x.size(3) + 2 * padding - dilation * (kernel_size[1] - 1) - 1) // stride + 1
    # Compute the number of blocks needed
    grid = lambda meta: (out_h + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], (out_w + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]
    # Launch the Triton kernel
    conv2d_kernel[grid](x, w, bias, out, x.size(0), x.size(1), out_channels, kernel_size[0], kernel_size[1], stride, stride, padding, padding, dilation, dilation, groups, BLOCK_SIZE=128, KERNEL_H=kernel_size[0], KERNEL_W=kernel_size[1])
    return out


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using a custom Triton kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Create weight and bias tensors
        weight = torch.nn.Parameter(torch.randn(self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        if self.bias:
            bias = torch.nn.Parameter(torch.randn(self.out_channels))
        else:
            bias = None
        # Perform the convolution using the Triton kernel
        return triton_conv2d(x, weight, bias, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
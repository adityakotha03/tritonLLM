import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    X_ptr,  # Pointer to input tensor
    W_ptr,  # Pointer to weight tensor
    B_ptr,  # Pointer to bias tensor (optional)
    Out_ptr,  # Pointer to output tensor
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    height,  # Input height
    width,  # Input width
    out_height,  # Output height
    out_width,  # Output width
    kernel_h,  # Kernel height
    kernel_w,  # Kernel width
    stride_h,  # Stride height
    stride_w,  # Stride width
    pad_h,  # Padding height
    pad_w,  # Padding width
    dilation_h,  # Dilation height
    dilation_w,  # Dilation width
    groups,  # Number of groups
    has_bias,  # Whether bias is used
    BLOCK_SIZE_H: tl.constexpr,  # Tile size for height dimension
    BLOCK_SIZE_W: tl.constexpr,  # Tile size for width dimension
    BLOCK_SIZE_C: tl.constexpr,  # Tile size for channel dimension (in/out)
    TILE_SIZE: tl.constexpr,  # Tile size for inner loop (for shared memory)
):
    # Each block processes one output element
    pid_b = tl.program_id(0)  # Batch ID
    pid_c_out = tl.program_id(1)  # Output channel ID
    pid_h = tl.program_id(2)  # Output height ID
    pid_w = tl.program_id(3)  # Output width ID

    # Calculate output offset
    out_offset = pid_b * out_channels * out_height * out_width + \
                 pid_c_out * out_height * out_width + \
                 pid_h * out_width + pid_w

    # Compute input indices with padding
    start_h = pid_h * stride_h - pad_h
    start_w = pid_w * stride_w - pad_w

    # Channel offset for grouped convolutions
    group_size = out_channels // groups
    group_id = pid_c_out // group_size
    in_channel_offset = (pid_c_out % group_size) * (in_channels // groups)

    # Loop over input channels and kernel elements
    acc = tl.zeros((1,), dtype=tl.float32)
    for c_in in tl.arange(0, in_channels):
        # Compute input channel index within group
        c_in_group = c_in % (in_channels // groups)
        if c_in_group != c_in:
            continue  # Skip if not in current group

        # Compute input position
        inp_h = start_h + c_in * dilation_h * kernel_h
        inp_w = start_w + c_in * dilation_w * kernel_w

        # Loop over kernel height and width
        for kh in tl.arange(0, kernel_h):
            for kw in tl.arange(0, kernel_w):
                # Compute input coordinates
                h_idx = start_h + kh * dilation_h
                w_idx = start_w + kw * dilation_w

                # Check bounds
                if h_idx < 0 or h_idx >= height or w_idx < 0 or w_idx >= width:
                    continue

                # Compute input index
                inp_idx = pid_b * in_channels * height * width + \
                          (c_in + group_id * (in_channels // groups)) * height * width + \
                          h_idx * width + w_idx

                # Compute weight index
                w_idx = (pid_c_out % group_size) * in_channels * kernel_h * kernel_w + \
                        c_in * kernel_h * kernel_w + kh * kernel_w + kw

                # Load input and weight
                x_val = tl.load(X_ptr + inp_idx, mask=(h_idx < height) & (w_idx < width), other=0.0)
                w_val = tl.load(W_ptr + w_idx, mask=True)

                acc += x_val * w_val

    # Add bias if present
    if has_bias:
        bias_val = tl.load(B_ptr + pid_c_out, mask=True)
        acc += bias_val

    # Store output
    tl.store(Out_ptr + out_offset, acc, mask=True)


def triton_conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: tuple, padding: tuple, dilation: tuple, groups: int):
    """
    Optimized 2D convolution using Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda), "Tensors must be on CUDA."
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    # Compute output dimensions
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1

    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, dtype=x.dtype, device=x.device)

    # Launch configuration
    grid = (
        batch_size,
        out_channels,
        out_height,
        out_width
    )

    # Use FP16/BF16 if available to leverage Tensor Cores
    dtype = x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_C = 32
    else:
        BLOCK_SIZE_H = 8
        BLOCK_SIZE_W = 8
        BLOCK_SIZE_C = 16

    # Ensure tile sizes are compatible with shared memory and registers
    TILE_SIZE = 16

    # Launch Triton kernel
    conv2d_kernel[grid](
        x,
        weight,
        bias,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        out_height,
        out_width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        bias is not None,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        TILE_SIZE=TILE_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
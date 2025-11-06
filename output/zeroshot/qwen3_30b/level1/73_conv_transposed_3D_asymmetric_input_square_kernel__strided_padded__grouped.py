import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    batch_size: tl.int32,
    in_channels: tl.int32,
    out_channels: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    padding: tl.int32,
    groups: tl.int32,
    depth: tl.int32,
    height: tl.int32,
    width: tl.int32,
    out_depth: tl.int32,
    out_height: tl.int32,
    out_width: tl.int32,
    in_channel_per_group: tl.int32,
    out_channel_per_group: tl.int32,
    kernel_size_sq: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    USE_SPARSE: tl.constexpr = False,
):
    # Shared memory for tiles of input and weight
    pid = tl.program_id(0)  # Block ID in grid
    num_blocks = tl.num_programs(0)

    # Compute output block indices
    out_idx = pid * BLOCK_SIZE
    out_d = out_idx // (out_height * out_width * out_channels)
    out_idx = out_idx % (out_height * out_width * out_channels)
    out_h = out_idx // (out_width * out_channels)
    out_idx = out_idx % (out_width * out_channels)
    out_w = out_idx // out_channels
    out_c = out_idx % out_channels

    # Output channel group and index within group
    group_id = out_c // out_channel_per_group
    c_in_group = out_c % out_channel_per_group

    # Compute input indices: mapped via transposed convolution
    i_d = out_d * stride - padding
    i_h = out_h * stride - padding
    i_w = out_w * stride - padding

    # Iterate over kernel size
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Input position after padding
                i_d_k = i_d + k_d
                i_h_k = i_h + k_h
                i_w_k = i_w + k_w

                # Check bounds
                if i_d_k < 0 or i_d_k >= depth or i_h_k < 0 or i_h_k >= height or i_w_k < 0 or i_w_k >= width:
                    continue

                # Input index: (batch, in_c, d, h, w)
                # in_c = out_c (grouped) + group offset
                in_c = group_id * in_channel_per_group + c_in_group

                # Compute input and weight offsets
                input_idx = ((batch_size * in_channels * depth * height * width) + (in_c * depth * height * width) + (i_d_k * height * width) + (i_h_k * width) + i_w_k)
                weight_idx = (out_c * in_channel_per_group * kernel_size_sq) + (c_in_group * kernel_size_sq) + (k_d * kernel_size * kernel_size) + (k_h * kernel_size) + k_w

                # Load input and weight
                x_val = tl.load(x_ptr + input_idx, mask=(i_d_k >= 0) & (i_d_k < depth) & (i_h_k >= 0) & (i_h_k < height) & (i_w_k >= 0) & (i_w_k < width), other=0.0)
                w_val = tl.load(w_ptr + weight_idx, mask=True, other=0.0)

                # Accumulate into output
                output_idx = (out_d * out_height * out_width * out_channels) + (out_h * out_width * out_channels) + (out_w * out_channels) + out_c
                tl.atomic_add(out_ptr + output_idx, x_val * w_val)


@triton.jit
def conv_transpose3d_kernel_fused(
    x_ptr,  # Input pointer
    w_ptr,  # Weight pointer
    out_ptr,  # Output pointer
    batch_size: tl.int32,
    in_channels: tl.int32,
    out_channels: tl.int32,
    kernel_size: tl.int32,
    stride: tl.int32,
    padding: tl.int32,
    groups: tl.int32,
    depth: tl.int32,
    height: tl.int32,
    width: tl.int32,
    out_depth: tl.int32,
    out_height: tl.int32,
    out_width: tl.int32,
    in_channel_per_group: tl.int32,
    out_channel_per_group: tl.int32,
    kernel_size_sq: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each block computes one output element
    pid = tl.program_id(0)
    num_blocks = tl.num_programs(0)

    # Compute output indices
    out_idx = pid * BLOCK_SIZE
    if out_idx >= out_depth * out_height * out_width * out_channels:
        return

    # Compute 3D coordinates
    out_d = out_idx // (out_height * out_width * out_channels)
    out_idx = out_idx % (out_height * out_width * out_channels)
    out_h = out_idx // (out_width * out_channels)
    out_idx = out_idx % (out_width * out_channels)
    out_w = out_idx // out_channels
    out_c = out_idx % out_channels

    # Group index and channel within group
    group_id = out_c // out_channel_per_group
    c_in_group = out_c % out_channel_per_group

    # Compute input coordinates (with stride and padding)
    i_d = out_d * stride - padding
    i_h = out_h * stride - padding
    i_w = out_w * stride - padding

    # Accumulator
    acc = tl.zeros((1,), dtype=tl.float32)

    # Iterate over kernel size
    for k_d in range(kernel_size):
        for k_h in range(kernel_size):
            for k_w in range(kernel_size):
                # Input position after padding
                i_d_k = i_d + k_d
                i_h_k = i_h + k_h
                i_w_k = i_w + k_w

                # Skip out-of-bounds
                if i_d_k < 0 or i_d_k >= depth or i_h_k < 0 or i_h_k >= height or i_w_k < 0 or i_w_k >= width:
                    continue

                # Compute input index
                in_c = group_id * in_channel_per_group + c_in_group
                input_idx = (batch_size * in_channels * depth * height * width) + (in_c * depth * height * width) + (i_d_k * height * width) + (i_h_k * width) + i_w_k
                weight_idx = (out_c * in_channel_per_group * kernel_size_sq) + (c_in_group * kernel_size_sq) + (k_d * kernel_size * kernel_size) + (k_h * kernel_size) + k_w

                # Load input and weight
                x_val = tl.load(x_ptr + input_idx, mask=(i_d_k >= 0) & (i_d_k < depth) & (i_h_k >= 0) & (i_h_k < height) & (i_w_k >= 0) & (i_w_k < width), other=0.0)
                w_val = tl.load(w_ptr + weight_idx, mask=True, other=0.0)

                acc += x_val * w_val

    # Output index
    out_idx = (out_d * out_height * out_width * out_channels) + (out_h * out_width * out_channels) + (out_w * out_channels) + out_c
    tl.store(out_ptr + out_idx, acc)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, groups: int, out_channels: int, in_channels: int, kernel_size: int) -> torch.Tensor:
    """
    Custom Triton kernel for 3D transposed convolution with fusing input and kernel access.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, depth, height, width = x.shape
    out_depth = (depth - 1) * stride - 2 * padding + kernel_size
    out_height = (height - 1) * stride - 2 * padding + kernel_size
    out_width = (width - 1) * stride - 2 * padding + kernel_size

    out_shape = (batch_size, out_channels, out_depth, out_height, out_width)
    out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)

    # Parameters
    in_channel_per_group = in_channels // groups
    out_channel_per_group = out_channels // groups
    kernel_size_sq = kernel_size * kernel_size
    num_elements = out.numel()

    # Tune block size
    BLOCK_SIZE = 128
    grid = lambda meta: (num_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"]

    # Launch kernel
    conv_transpose3d_kernel_fused[grid](
        x, w, out,
        batch_size, in_channels, out_channels, kernel_size, stride, padding,
        groups, depth, height, width, out_depth, out_height, out_width,
        in_channel_per_group, out_channel_per_group, kernel_size_sq,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias

        # Initialize weight
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous input and weight
        x = x.contiguous()
        w = self.weight.contiguous()

        # Run Triton kernel
        out = triton_conv_transpose3d(x, w, self.stride, self.padding, self.groups, self.out_channels, self.in_channels, self.kernel_size)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, self.out_channels, 1, 1, 1)

        return out
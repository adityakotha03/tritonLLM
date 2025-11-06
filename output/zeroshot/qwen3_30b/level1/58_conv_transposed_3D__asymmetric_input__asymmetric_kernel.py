import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to weight tensor
    out_ptr,  # Pointer to output tensor
    bias_ptr,  # Pointer to bias tensor (if exists)
    batch_size: tl.int32,
    in_channels: tl.int32,
    out_channels: tl.int32,
    depth_in: tl.int32,
    height_in: tl.int32,
    width_in: tl.int32,
    depth_out: tl.int32,
    height_out: tl.int32,
    width_out: tl.int32,
    kernel_depth: tl.int32,
    kernel_height: tl.int32,
    kernel_width: tl.int32,
    stride_depth: tl.int32,
    stride_height: tl.int32,
    stride_width: tl.int32,
    padding_depth: tl.int32,
    padding_height: tl.int32,
    padding_width: tl.int32,
    output_padding_depth: tl.int32,
    output_padding_height: tl.int32,
    output_padding_width: tl.int32,
    groups: tl.int32,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE_D: tl.constexpr,
    TILE_SIZE_H: tl.constexpr,
    TILE_SIZE_W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Define program ID for current block
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_group = tl.program_id(2)

    # Compute global indices for output (batch, out_ch, depth, height, width)
    batch_idx = pid_batch
    out_ch_idx = pid_out_ch
    group_idx = pid_group

    # Calculate output channel offset within the group
    out_ch_offset = out_ch_idx * groups + group_idx

    # Calculate the output shape per group
    out_ch_per_group = out_channels // groups
    if out_ch_idx >= out_ch_per_group:
        return

    # Compute output spatial coordinates
    depth_offset = tl.load(tl.make_block_ptr(
        base=tl.arange(0, TILE_SIZE_D),
        shape=(TILE_SIZE_D,),
        strides=(1,),
        offsets=(0,),
        block_shape=(TILE_SIZE_D,),
        order=(0,)
    ) + tl.arange(0, TILE_SIZE_D) * BLOCK_SIZE_D)

    height_offset = tl.load(tl.make_block_ptr(
        base=tl.arange(0, TILE_SIZE_H),
        shape=(TILE_SIZE_H,),
        strides=(1,),
        offsets=(0,),
        block_shape=(TILE_SIZE_H,),
        order=(0,)
    ) + tl.arange(0, TILE_SIZE_H) * BLOCK_SIZE_H)

    width_offset = tl.load(tl.make_block_ptr(
        base=tl.arange(0, TILE_SIZE_W),
        shape=(TILE_SIZE_W,),
        strides=(1,),
        offsets=(0,),
        block_shape=(TILE_SIZE_W,),
        order=(0,)
    ) + tl.arange(0, TILE_SIZE_W) * BLOCK_SIZE_W)

    # Calculate global output indices (depth, height, width)
    depth_out_global = depth_offset
    height_out_global = height_offset
    width_out_global = width_offset

    # Ensure valid output indices
    mask_d = depth_out_global < depth_out
    mask_h = height_out_global < height_out
    mask_w = width_out_global < width_out
    mask = mask_d & mask_h & mask_w

    # Compute input indices
    depth_in_start = (depth_out_global - padding_depth) * stride_depth - output_padding_depth
    height_in_start = (height_out_global - padding_height) * stride_height - output_padding_height
    width_in_start = (width_out_global - padding_width) * stride_width - output_padding_width

    # Apply output padding
    depth_in_start += output_padding_depth
    height_in_start += output_padding_height
    width_in_start += output_padding_width

    # Clamp to input bounds
    depth_in_start = tl.max(tl.zeros((), dtype=tl.int32), depth_in_start)
    height_in_start = tl.max(tl.zeros((), dtype=tl.int32), height_in_start)
    width_in_start = tl.max(tl.zeros((), dtype=tl.int32), width_in_start)

    # Compute end indices (clamped to input bounds)
    depth_in_end = depth_in_start + kernel_depth
    height_in_end = height_in_start + kernel_height
    width_in_end = width_in_start + kernel_width

    depth_in_end = tl.min(depth_in_end, depth_in)
    height_in_end = tl.min(height_in_end, height_in)
    width_in_end = tl.min(width_in_end, width_in)

    # Iterate over input channels
    accum = tl.zeros((TILE_SIZE_D, TILE_SIZE_H, TILE_SIZE_W), dtype=tl.float32)

    for c_in in range(in_channels):
        # Load input data (TILE_SIZE_D x TILE_SIZE_H x TILE_SIZE_W) in a blocked fashion
        x_block_ptrs = tl.make_block_ptr(
            base=x_ptr + (batch_idx * in_channels + c_in) * depth_in * height_in * width_in,
            shape=(depth_in, height_in, width_in),
            strides=(height_in * width_in, width_in, 1),
            offsets=(depth_in_start, height_in_start, width_in_start),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W),
            order=(0, 1, 2)
        )

        # Load weights (kernel depth x height x width)
        w_block_ptrs = tl.make_block_ptr(
            base=w_ptr + (out_ch_offset * in_channels + c_in) * kernel_depth * kernel_height * kernel_width,
            shape=(kernel_depth, kernel_height, kernel_width),
            strides=(kernel_height * kernel_width, kernel_width, 1),
            offsets=(0, 0, 0),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W),
            order=(0, 1, 2)
        )

        # Load input and weights (with masking)
        x_vals = tl.load(x_block_ptrs, mask=tl.load(tl.make_block_ptr(
            base=tl.arange(0, BLOCK_SIZE_D),
            shape=(BLOCK_SIZE_D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_D,),
            order=(0,)
        ) < depth_in_end - depth_in_start, other=0.0), other=0.0)
        w_vals = tl.load(w_block_ptrs, mask=tl.load(tl.make_block_ptr(
            base=tl.arange(0, BLOCK_SIZE_D),
            shape=(BLOCK_SIZE_D,),
            strides=(1,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_D,),
            order=(0,)
        ) < kernel_depth, other=0.0), other=0.0)

        # Perform reduction over kernel dimensions (convolution in transposed mode)
        for k_d in range(BLOCK_SIZE_D):
            for k_h in range(BLOCK_SIZE_H):
                for k_w in range(BLOCK_SIZE_W):
                    d_in = k_d
                    h_in = k_h
                    w_in = k_w
                    d_out = depth_out_global - (depth_in_start + d_in)
                    h_out = height_out_global - (height_in_start + h_in)
                    w_out = width_out_global - (width_in_start + w_in)
                    # Check bounds and accumulate
                    if (d_in < kernel_depth and h_in < kernel_height and w_in < kernel_width and
                        d_out >= 0 and h_out >= 0 and w_out >= 0):
                        accum += x_vals[k_d, k_h, k_w] * w_vals[k_d, k_h, k_w]

    # Apply bias if exists
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_offset)
        accum += bias_val

    # Write output
    out_block_ptrs = tl.make_block_ptr(
        base=out_ptr + (batch_idx * out_channels + out_ch_offset) * depth_out * height_out * width_out,
        shape=(depth_out, height_out, width_out),
        strides=(height_out * width_out, width_out, 1),
        offsets=(depth_out_global, height_out_global, width_out_global),
        block_shape=(TILE_SIZE_D, TILE_SIZE_H, TILE_SIZE_W),
        order=(0, 1, 2)
    )

    tl.store(out_block_ptrs, accum, mask=mask)


def triton_conv_transpose3d(x, w, bias, stride, padding, output_padding, groups):
    # Ensure inputs are contiguous and on GPU
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Input and output shapes
    batch_size, in_channels, depth_in, height_in, width_in = x.shape
    out_channels, _, kernel_depth, kernel_height, kernel_width = w.shape
    stride_depth, stride_height, stride_width = stride
    padding_depth, padding_height, padding_width = padding
    output_padding_depth, output_padding_height, output_padding_width = output_padding

    # Compute output dimensions
    depth_out = (depth_in - 1) * stride_depth - 2 * padding_depth + kernel_depth + output_padding_depth
    height_out = (height_in - 1) * stride_height - 2 * padding_height + kernel_height + output_padding_height
    width_out = (width_in - 1) * stride_width - 2 * padding_width + kernel_width + output_padding_width

    # Prepare output tensor
    out = torch.empty(batch_size, out_channels, depth_out, height_out, width_out, dtype=x.dtype, device=x.device)

    # Determine grid and block sizes
    # Use a fixed tile size to fit in shared memory and maximize occupancy
    TILE_SIZE_D = 8
    TILE_SIZE_H = 8
    TILE_SIZE_W = 8
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8

    # Number of output channels per group
    out_ch_per_group = out_channels // groups

    # Grid: (batch, out_channels, groups)
    grid = (
        batch_size,
        out_ch_per_group,
        groups
    )

    # Launch kernel
    conv_transpose3d_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        out_ptr=out,
        bias_ptr=bias,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depth_in=depth_in,
        height_in=height_in,
        width_in=width_in,
        depth_out=depth_out,
        height_out=height_out,
        width_out=width_out,
        kernel_depth=kernel_depth,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        stride_depth=stride_depth,
        stride_height=stride_height,
        stride_width=stride_width,
        padding_depth=padding_depth,
        padding_height=padding_height,
        padding_width=padding_width,
        output_padding_depth=output_padding_depth,
        output_padding_height=output_padding_height,
        output_padding_width=output_padding_width,
        groups=groups,
        n_elements=out.numel(),
        BLOCK_SIZE=128,
        TILE_SIZE_D=TILE_SIZE_D,
        TILE_SIZE_H=TILE_SIZE_H,
        TILE_SIZE_W=TILE_SIZE_W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias

        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Triton kernel for transposed 3D convolution
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )
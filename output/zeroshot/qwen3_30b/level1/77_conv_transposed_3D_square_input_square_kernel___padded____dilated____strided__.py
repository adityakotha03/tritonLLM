import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,  # Pointer to input tensor (batch, in_channels, depth, height, width)
    weight_ptr,  # Pointer to weight tensor (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    output_ptr,  # Pointer to output tensor (batch, out_channels, out_depth, out_height, out_width)
    batch_size, in_channels, out_channels, depth, height, width,
    kernel_size, stride, padding, dilation,
    output_depth, output_height, output_width,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Block indices
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    out_d_id = tl.program_id(2)
    out_h_id = tl.program_id(3)
    out_w_id = tl.program_id(4)

    # Calculate output indices
    out_d_start = out_d_id * TILE_SIZE
    out_h_start = out_h_id * TILE_SIZE
    out_w_start = out_w_id * TILE_SIZE

    # Define tile size
    tile_size = TILE_SIZE

    # Define block size for inner loop
    block_size = BLOCK_SIZE

    # Define output offset
    output_offset = (batch_id * out_channels * output_depth * output_height * output_width +
                     out_ch_id * output_depth * output_height * output_width +
                     out_d_start * output_height * output_width + 
                     out_h_start * output_width +
                     out_w_start)

    # Shared memory for input tiles
    input_tile = tl.zeros((in_channels, kernel_size, kernel_size, kernel_size), dtype=tl.float32)
    weight_tile = tl.zeros((in_channels, kernel_size, kernel_size, kernel_size), dtype=tl.float32)

    # Compute input spatial offsets
    input_d_start = out_d_start * stride - padding
    input_h_start = out_h_start * stride - padding
    input_w_start = out_w_start * stride - padding

    # Initialize output tile
    output_tile = tl.zeros((block_size, block_size, block_size), dtype=tl.float32)

    # Loop over input channels
    for ic in range(0, in_channels, block_size):
        ic_end = min(ic + block_size, in_channels)

        # Load input tile
        input_offsets = (
            batch_id * in_channels * depth * height * width +
            ic * depth * height * width +
            (input_d_start + tl.arange(0, kernel_size))[:, None, None] * height * width +
            (input_h_start + tl.arange(0, kernel_size))[None, :, None] * width +
            (input_w_start + tl.arange(0, kernel_size))[None, None, :]
        )
        input_mask = (
            (input_d_start + tl.arange(0, kernel_size))[:, None, None] < depth
            & (input_h_start + tl.arange(0, kernel_size))[None, :, None] < height
            & (input_w_start + tl.arange(0, kernel_size))[None, None, :] < width
        )
        input_tile = tl.load(
            input_ptr + input_offsets,
            mask=input_mask[:, None, None],
            other=0.0
        )
        input_tile = tl.trans(input_tile)  # (kernel_size, kernel_size, kernel_size, in_channels)

        # Load weight tile
        weight_offsets = (
            out_ch_id * in_channels * kernel_size * kernel_size * kernel_size +
            ic * kernel_size * kernel_size * kernel_size +
            tl.arange(0, kernel_size)[:, None, None] * kernel_size * kernel_size +
            tl.arange(0, kernel_size)[None, :, None] * kernel_size +
            tl.arange(0, kernel_size)[None, None, :]
        )
        weight_mask = (ic < in_channels)
        weight_tile = tl.load(
            weight_ptr + weight_offsets,
            mask=weight_mask[:, None, None],
            other=0.0
        )
        weight_tile = tl.trans(weight_tile)  # (kernel_size, kernel_size, kernel_size, in_channels)

        # Compute convolution using tensor core-friendly matmul
        # Use tiling to reduce register pressure
        for di in range(0, tile_size, block_size):
            for hi in range(0, tile_size, block_size):
                for wi in range(0, tile_size, block_size):
                    # Compute output index
                    out_d = out_d_start + di
                    out_h = out_h_start + hi
                    out_w = out_w_start + wi

                    # Compute input indices
                    input_d = out_d * stride - padding
                    input_h = out_h * stride - padding
                    input_w = out_w * stride - padding

                    # Compute output value
                    val = 0.0
                    for kd in range(kernel_size):
                        for kh in range(kernel_size):
                            for kw in range(kernel_size):
                                d_idx = input_d + kd * dilation
                                h_idx = input_h + kh * dilation
                                w_idx = input_w + kw * dilation

                                # Check bounds
                                valid = (
                                    d_idx >= 0 and d_idx < depth and
                                    h_idx >= 0 and h_idx < height and
                                    w_idx >= 0 and w_idx < width
                                )
                                if valid:
                                    val += input_tile[kd, kh, kw] * weight_tile[kd, kh, kw]

                    # Store result
                    output_tile[di, hi, wi] = val

    # Store output tile
    output_offsets = output_offset + tl.arange(0, tile_size)[:, None, None] * output_height * output_width + \
                     tl.arange(0, tile_size)[None, :, None] * output_width + \
                     tl.arange(0, tile_size)[None, None, :]
    output_mask = (
        (out_d_start + tl.arange(0, tile_size))[:, None, None] < output_depth
        & (out_h_start + tl.arange(0, tile_size))[None, :, None] < output_height
        & (out_w_start + tl.arange(0, tile_size))[None, None, :] < output_width
    )
    tl.store(
        output_ptr + output_offsets,
        output_tile,
        mask=output_mask[:, None, None]
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_buffer('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate output dimensions
        batch_size, _, depth, height, width = x.shape
        output_depth = (depth - 1) * self.stride - 2 * self.padding + self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        output_height = (height - 1) * self.stride - 2 * self.padding + self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)
        output_width = (width - 1) * self.stride - 2 * self.padding + self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1)

        # Allocate output tensor
        output = torch.empty(
            batch_size, self.out_channels, output_depth, output_height, output_width,
            dtype=x.dtype, device=x.device
        )

        # Ensure contiguous inputs
        x = x.contiguous()
        weight = self.weight.contiguous()

        # Define grid
        grid = lambda meta: (
            batch_size,
            self.out_channels,
            (output_depth + meta['TILE_SIZE'] - 1) // meta['TILE_SIZE'],
            (output_height + meta['TILE_SIZE'] - 1) // meta['TILE_SIZE'],
            (output_width + meta['TILE_SIZE'] - 1) // meta['TILE_SIZE'],
        )

        # Launch kernel
        conv_transpose3d_kernel[
            grid
        ](
            x, weight, output,
            batch_size, self.in_channels, self.out_channels, depth, height, width,
            self.kernel_size, self.stride, self.padding, self.dilation,
            output_depth, output_height, output_width,
            BLOCK_SIZE=16, TILE_SIZE=8
        )

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, self.out_channels, 1, 1, 1)

        return output
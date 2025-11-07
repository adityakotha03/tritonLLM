import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose_3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_size,
    stride,
    padding,
    min_value,
    divisor,
    input_stride_0,  # batch
    input_stride_1,  # in_channels
    input_stride_2,  # depth
    input_stride_3,  # height
    input_stride_4,  # width
    weight_stride_0,  # out_channels
    weight_stride_1,  # in_channels
    weight_stride_2,  # kernel_depth
    weight_stride_3,  # kernel_height
    weight_stride_4,  # kernel_width
    output_stride_0,  # batch
    output_stride_1,  # out_channels
    output_stride_2,  # out_depth
    output_stride_3,  # out_height
    output_stride_4,  # out_width
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
    BLOCK_SIZE_IC: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    TILE_OC: tl.constexpr,
    TILE_IC: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    # Define grid for the kernel
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_w = tl.program_id(4)

    # Compute output tensor indices
    out_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    out_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    out_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    # Output coordinate mask to avoid out-of-bounds
    mask_d = out_d < depth
    mask_h = out_h < height
    mask_w = out_w < width
    mask = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # Compute input indices (output -> input mapping via transposed conv)
    input_d = out_d * stride - padding
    input_h = out_h * stride - padding
    input_w = out_w * stride - padding

    # Kernel indices
    kernel_d = tl.arange(0, kernel_size)[:, None, None]
    kernel_h = tl.arange(0, kernel_size)[None, :, None]
    kernel_w = tl.arange(0, kernel_size)[None, None, :]

    # Compute input coordinates
    input_d = input_d[None, None, None, :] + kernel_d
    input_h = input_h[None, None, :, None] + kernel_h
    input_w = input_w[None, :, None, None] + kernel_w

    # Flatten coordinates for indexing
    input_d = input_d.view(-1)
    input_h = input_h.view(-1)
    input_w = input_w.view(-1)

    # Compute global offsets for input and weight
    input_offsets = (
        pid_b * input_stride_0 +
        tl.arange(0, in_channels)[None, :] * input_stride_1 +
        input_d[:, None] * input_stride_2 +
        input_h[:, None] * input_stride_3 +
        input_w[:, None] * input_stride_4
    )

    weight_offsets = (
        pid_o * weight_stride_0 +
        tl.arange(0, in_channels)[:, None] * weight_stride_1 +
        kernel_d * weight_stride_2 +
        kernel_h * weight_stride_3 +
        kernel_w * weight_stride_4
    )

    # Load input and weight (use shared memory for input and weight)
    # Use shared memory to cache input patches and weight
    shared_input = tl.load(
        input_ptr + input_offsets,
        mask=(input_d[:, None] < depth) & (input_h[:, None] < height) & (input_w[:, None] < width),
        other=0.0,
        cache_type=tl.load.cache_type.shared
    )
    shared_weight = tl.load(
        weight_ptr + weight_offsets,
        cache_type=tl.load.cache_type.shared
    )

    # Initialize output accumulator
    output = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_OC), dtype=tl.float32)

    # Tile over input channels and output channels
    for ic in range(0, in_channels, BLOCK_SIZE_IC):
        for oc in range(0, out_channels, BLOCK_SIZE_OC):
            # Load input chunk
            ic_mask = tl.arange(0, BLOCK_SIZE_IC) < (in_channels - ic)
            input_chunk = tl.load(
                shared_input + ic * input_stride_1 + tl.arange(0, BLOCK_SIZE_IC) * input_stride_1,
                mask=ic_mask[None, None, None, :],
                other=0.0
            )

            # Load weight chunk
            oc_mask = tl.arange(0, BLOCK_SIZE_OC) < (out_channels - oc)
            weight_chunk = tl.load(
                shared_weight + oc * weight_stride_0 + ic * weight_stride_1 + tl.arange(0, BLOCK_SIZE_OC)[:, None, None, None] * weight_stride_0,
                mask=oc_mask[None, :, None, None],
                other=0.0
            )

            # Perform matrix multiplication: (DHW x IC) @ (IC x OC) -> (DHW x OC)
            # Use tensor cores if enabled
            if USE_TENSOR_CORES:
                out_tile = tl.dot(input_chunk, weight_chunk, acc=output)
            else:
                out_tile = tl.dot(input_chunk, weight_chunk, acc=output)

            output = out_tile

    # Apply clamp and divide
    output = tl.clamp(output, min=min_value) / divisor

    # Store output
    output_offsets = (
        pid_b * output_stride_0 +
        pid_o * output_stride_1 +
        out_d[:, None, None] * output_stride_2 +
        out_h[None, :, None] * output_stride_3 +
        out_w[None, None, :] * output_stride_4
    )

    tl.store(
        output_ptr + output_offsets,
        output,
        mask=mask[:, :, :, None] & (tl.arange(0, BLOCK_SIZE_OC)[None, None, None, :] < out_channels)
    )


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Register weight buffer
        self.register_buffer("weight", self.conv_transpose.weight.data)

    def forward(self, x):
        # Ensure contiguous input
        x = x.contiguous()

        # Prepare output tensor
        out_shape = (x.size(0), self.conv_transpose.out_channels, x.size(2) * self.stride - 2 * self.padding + self.kernel_size - 1,
                      x.size(3) * self.stride - 2 * self.padding + self.kernel_size - 1,
                      x.size(4) * self.stride - 2 * self.padding + self.kernel_size - 1)

        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)

        # Get strides
        input_stride_0, input_stride_1, input_stride_2, input_stride_3, input_stride_4 = x.stride()
        weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3, weight_stride_4 = self.weight.stride()
        output_stride_0, output_stride_1, output_stride_2, output_stride_3, output_stride_4 = output.stride()

        # Determine block sizes and grid
        BLOCK_SIZE_D, BLOCK_SIZE_H, BLOCK_SIZE_W = 8, 8, 8
        BLOCK_SIZE_OC = 16
        BLOCK_SIZE_IC = 16
        TILE_D, TILE_H, TILE_W = 8, 8, 8
        TILE_OC, TILE_IC = 16, 16

        # Use autotuning to find best configuration
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE_D': 8, 'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8,
                               'BLOCK_SIZE_OC': 16, 'BLOCK_SIZE_IC': 16,
                               'TILE_D': 8, 'TILE_H': 8, 'TILE_W': 8,
                               'TILE_OC': 16, 'TILE_IC': 16,
                               'USE_TENSOR_CORES': True}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_SIZE_D': 16, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16,
                               'BLOCK_SIZE_OC': 32, 'BLOCK_SIZE_IC': 16,
                               'TILE_D': 16, 'TILE_H': 16, 'TILE_W': 16,
                               'TILE_OC': 32, 'TILE_IC': 16,
                               'USE_TENSOR_CORES': True}, num_stages=3, num_warps=8),
            ],
            key=['in_channels', 'out_channels', 'depth', 'height', 'width', 'kernel_size']
        )
        @triton.jit
        def fused_conv_transpose_kernel(
            input_ptr,
            weight_ptr,
            output_ptr,
            batch_size,
            in_channels,
            out_channels,
            depth,
            height,
            width,
            kernel_size,
            stride,
            padding,
            min_value,
            divisor,
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            input_stride_4,
            weight_stride_0,
            weight_stride_1,
            weight_stride_2,
            weight_stride_3,
            weight_stride_4,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            output_stride_4,
            BLOCK_SIZE_D: tl.constexpr,
            BLOCK_SIZE_H: tl.constexpr,
            BLOCK_SIZE_W: tl.constexpr,
            BLOCK_SIZE_OC: tl.constexpr,
            BLOCK_SIZE_IC: tl.constexpr,
            TILE_D: tl.constexpr,
            TILE_H: tl.constexpr,
            TILE_W: tl.constexpr,
            TILE_OC: tl.constexpr,
            TILE_IC: tl.constexpr,
            USE_TENSOR_CORES: tl.constexpr,
        ):
            # Call the kernel
            conv_transpose_3d_kernel[
                (batch_size, out_channels, (depth + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
                 (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
                 (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
            ](
                input_ptr,
                weight_ptr,
                output_ptr,
                batch_size,
                in_channels,
                out_channels,
                depth,
                height,
                width,
                kernel_size,
                stride,
                padding,
                min_value,
                divisor,
                input_stride_0,
                input_stride_1,
                input_stride_2,
                input_stride_3,
                input_stride_4,
                weight_stride_0,
                weight_stride_1,
                weight_stride_2,
                weight_stride_3,
                weight_stride_4,
                output_stride_0,
                output_stride_1,
                output_stride_2,
                output_stride_3,
                output_stride_4,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
                BLOCK_SIZE_W=BLOCK_SIZE_W,
                BLOCK_SIZE_OC=BLOCK_SIZE_OC,
                BLOCK_SIZE_IC=BLOCK_SIZE_IC,
                TILE_D=TILE_D,
                TILE_H=TILE_H,
                TILE_W=TILE_W,
                TILE_OC=TILE_OC,
                TILE_IC=TILE_IC,
                USE_TENSOR_CORES=USE_TENSOR_CORES,
            )

        # Launch autotuned kernel
        fused_conv_transpose_kernel[
            (x.size(0), self.conv_transpose.out_channels,
             (out_shape[2] + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D,
             (out_shape[3] + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
             (out_shape[4] + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
        ](
            x,
            self.weight,
            output,
            x.size(0),
            x.size(1),
            self.conv_transpose.out_channels,
            out_shape[2],
            out_shape[3],
            out_shape[4],
            self.kernel_size,
            self.stride,
            self.padding,
            self.min_value,
            self.divisor,
            input_stride_0,
            input_stride_1,
            input_stride_2,
            input_stride_3,
            input_stride_4,
            weight_stride_0,
            weight_stride_1,
            weight_stride_2,
            weight_stride_3,
            weight_stride_4,
            output_stride_0,
            output_stride_1,
            output_stride_2,
            output_stride_3,
            output_stride_4,
        )

        return output
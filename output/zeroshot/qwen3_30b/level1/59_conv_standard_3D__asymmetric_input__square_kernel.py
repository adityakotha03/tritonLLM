import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr,  # Pointer to input tensor (B, C_in, H, W, D)
    w_ptr,  # Pointer to kernel weights (C_out, C_in, KH, KW, 1)
    out_ptr,  # Pointer to output tensor (B, C_out, H_out, W_out, D_out)
    batch_size,  # Number of batches
    in_channels,  # Number of input channels
    out_channels,  # Number of output channels
    height,  # Height of input
    width,  # Width of input
    depth,  # Depth of input
    kernel_size,  # Size of kernel (KH, KW, 1)
    stride,  # Stride for height and width
    padding,  # Padding applied to input
    dilation,  # Dilation of kernel
    groups,  # Number of groups
    H_out,  # Output height
    W_out,  # Output width
    D_out,  # Output depth
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    TILE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    pid_d = tl.program_id(4)

    # Calculate output position in batch, channels, height, width, depth
    batch_start = pid_b * BLOCK_H * H_out
    c_out_start = pid_c_out * BLOCK_C_OUT
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    d_start = pid_d * BLOCK_D

    # Offset for output
    output_offset = (pid_b * out_channels * H_out * W_out * D_out +
                     c_out_start * H_out * W_out * D_out +
                     h_start * W_out * D_out +
                     w_start * D_out +
                     d_start)

    # Output pointer
    out_ptr_block = out_ptr + output_offset

    # Shared memory for tiles of input and weights
    # We tile the input along C_in and kernel dimensions
    # Shared memory layout: [BLOCK_C_IN, TILE_K, BLOCK_H, BLOCK_W, BLOCK_D]
    shmem_input = tl.make_block_ptr(
        base=x_ptr,
        shape=(batch_size, in_channels, height, width, depth),
        strides=(in_channels * height * width * depth, height * width * depth, width * depth, depth, 1),
        offsets=(pid_b, 0, 0, 0, 0),
        block_shape=(1, BLOCK_C_IN, BLOCK_H, BLOCK_W, BLOCK_D),
        order=(0, 1, 2, 3, 4)
    )

    # Shared memory for kernel weights: [BLOCK_C_OUT, BLOCK_C_IN, kernel_size, kernel_size, 1]
    shmem_weight = tl.make_block_ptr(
        base=w_ptr,
        shape=(out_channels, in_channels, kernel_size, kernel_size, 1),
        strides=(in_channels * kernel_size * kernel_size, kernel_size * kernel_size, kernel_size, 1, 1),
        offsets=(c_out_start, 0, 0, 0, 0),
        block_shape=(BLOCK_C_OUT, BLOCK_C_IN, kernel_size, kernel_size, 1),
        order=(0, 1, 2, 3, 4)
    )

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_H, BLOCK_W, BLOCK_D, BLOCK_C_OUT), dtype=tl.float32)

    # Iterate over input channel groups
    for c_in_block in range(0, in_channels, BLOCK_C_IN):
        c_in_start = c_in_block
        # Load input tile
        input_tile = tl.load(
            shmem_input,
            boundary_check=(0, 1, 2, 3, 4),
            padding_option="zero"
        )

        # Load kernel tile
        weight_tile = tl.load(
            shmem_weight,
            boundary_check=(0, 1, 2, 3, 4),
            padding_option="zero"
        )

        # Broadcast input tile to match kernel shape
        # Expand input: (1, BLOCK_C_IN, BLOCK_H, BLOCK_W, BLOCK_D) -> (BLOCK_C_IN, BLOCK_H, BLOCK_W, BLOCK_D)
        # Expand kernel: (BLOCK_C_OUT, BLOCK_C_IN, kernel_size, kernel_size, 1) -> (BLOCK_C_OUT, BLOCK_C_IN, kernel_size, kernel_size, 1)
        # Compute conv operation: input * weight
        for h_k in range(kernel_size):
            for w_k in range(kernel_size):
                # Compute actual input coordinates for convolution
                h_input = h_start + h_k - padding
                w_input = w_start + w_k - padding

                # Skip if outside valid input region
                if h_input < 0 or h_input >= height or w_input < 0 or w_input >= width:
                    continue

                # Stride and dilation
                h_input = h_input + (h_k - padding) * dilation
                w_input = w_input + (w_k - padding) * dilation

                # Check bounds
                h_in_valid = (h_input >= 0) & (h_input < height)
                w_in_valid = (w_input >= 0) & (w_input < width)

                # Only process valid positions
                if not h_in_valid or not w_in_valid:
                    continue

                # Load input at (h_input, w_input)
                input_val = tl.load(
                    x_ptr + pid_b * in_channels * height * width * depth +
                    c_in_start * height * width * depth +
                    h_input * width * depth +
                    w_input * depth,
                    mask=(tl.arange(0, BLOCK_H) < BLOCK_H) & (tl.arange(0, BLOCK_W) < BLOCK_W) & (tl.arange(0, BLOCK_D) < BLOCK_D),
                    other=0.0
                )

                # Expand to tile size
                input_expanded = tl.broadcast_to(input_val, (BLOCK_C_IN, BLOCK_H, BLOCK_W, BLOCK_D))

                # Load weight at (h_k, w_k)
                weight_val = tl.load(
                    w_ptr + (c_out_start * in_channels + c_in_start) * kernel_size * kernel_size +
                    h_k * kernel_size + w_k,
                    mask=(tl.arange(0, BLOCK_C_OUT) < BLOCK_C_OUT) & (tl.arange(0, BLOCK_C_IN) < BLOCK_C_IN),
                    other=0.0
                )

                # Expand weight to (BLOCK_C_OUT, BLOCK_C_IN, 1, 1, 1)
                weight_expanded = tl.broadcast_to(weight_val, (BLOCK_C_OUT, BLOCK_C_IN, 1, 1, 1))

                # Compute partial dot product
                accumulator += tl.dot(input_expanded, weight_expanded)

    # Store result
    tl.store(
        out_ptr_block,
        accumulator,
        boundary_check=(0, 1, 2, 3)
    )


def triton_conv3d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, groups: int, out_channels: int, kernel_size: int, height: int, width: int, depth: int) -> torch.Tensor:
    """
    Custom Triton-based 3D convolution with optimized memory access and tiling.
    Supports asymmetric input and square kernel (KH, KW, 1).
    Uses shared memory, tiling, and fused operations.
    """
    # Ensure inputs are contiguous and on CUDA
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()

    # Extract dimensions
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    H_out = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    W_out = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    D_out = depth  # No change in depth since kernel is (1, 1, 1) in depth

    # Output tensor
    out = torch.empty(batch_size, out_channels, H_out, W_out, D_out, dtype=x.dtype, device=x.device)

    # Define block sizes (tunable)
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_D = 16
    BLOCK_C_OUT = 32
    BLOCK_C_IN = 16
    TILE_K = 8  # Unused but kept for flexibility

    # Grid definition: (batch, C_out, H_out, W_out, D_out)
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["BLOCK_C_OUT"] - 1) // meta["BLOCK_C_OUT"],
        (H_out + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W_out + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
        (D_out + meta["BLOCK_D"] - 1) // meta["BLOCK_D"]
    )

    # Launch kernel
    conv3d_kernel[grid](
        x, w, out,
        batch_size, in_channels, out_channels, height, width, depth,
        kernel_size, stride, padding, dilation, groups, H_out, W_out, D_out,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, BLOCK_D=BLOCK_D,
        BLOCK_C_OUT=BLOCK_C_OUT, BLOCK_C_IN=BLOCK_C_IN, TILE_K=TILE_K
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, 1))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run custom Triton 3D convolution
        out = triton_conv3d(
            x, self.weight, self.stride, self.padding, self.dilation, self.groups,
            out_channels=self.weight.shape[0], kernel_size=self.weight.shape[2],
            height=x.shape[2], width=x.shape[3], depth=x.shape[4]
        )

        # Add bias if applicable
        if self.bias is not None:
            out = out + self.bias.view(1, self.bias.shape[0], 1, 1, 1)

        return out
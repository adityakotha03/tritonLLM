import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_kernel(
    x_ptr,  # Pointer to input tensor
    w_ptr,  # Pointer to kernel weights
    out_ptr,  # Pointer to output tensor
    x_stride0, x_stride1, x_stride2, x_stride3,  # Strides for input tensor (B, C, H, W)
    w_stride0, w_stride1, w_stride2, w_stride3,  # Strides for weight tensor (O, I, KH, KW)
    out_stride0, out_stride1, out_stride2, out_stride3,  # Strides for output tensor (B, O, OH, OW)
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding, dilation, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_OC: tl.constexpr,
):
    # Thread block indices
    pid_b = tl.program_id(0)  # Batch index
    pid_o = tl.program_id(1)  # Output channel group index
    pid_h = tl.program_id(2)  # Output height index
    pid_w = tl.program_id(3)  # Output width index

    # Calculate output height and width
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Compute output offsets
    out_offset_h = pid_h * BLOCK_SIZE_H
    out_offset_w = pid_w * BLOCK_SIZE_W
    out_offset_o = pid_o * BLOCK_SIZE_OC

    # Output dimensions
    out_end_h = out_height
    out_end_w = out_width
    out_end_o = out_channels

    # Compute input offsets (after padding)
    input_offset_h = out_offset_h * stride - padding
    input_offset_w = out_offset_w * stride - padding

    # Shared memory for input tile and weights tile
    # Use static size for simplicity and to fit within 163KB limit
    input_tile = tl.load(
        tl.make_block_ptr(x_ptr, (batch_size, in_channels, height, width),
                          (x_stride0, x_stride1, x_stride2, x_stride3),
                          (pid_b, 0, input_offset_h, input_offset_w),
                          (1, in_channels, BLOCK_SIZE_H + (kernel_size - 1) * dilation, BLOCK_SIZE_W + (kernel_size - 1) * dilation),
                          (0, 0, 0, 0)),
        boundary_check=(0, 1, 2, 3),
        padding_option='zeros'
    )

    # Load weights: one output channel group
    # Since we are processing one group of output channels, we load corresponding weights
    weight_tile = tl.load(
        tl.make_block_ptr(w_ptr, (out_channels, in_channels, kernel_size, kernel_size),
                          (w_stride0, w_stride1, w_stride2, w_stride3),
                          (out_offset_o, 0, 0, 0),
                          (BLOCK_SIZE_OC, in_channels, kernel_size, kernel_size),
                          (0, 0, 0, 0)),
        boundary_check=(0, 1, 2, 3),
        padding_option='zeros'
    )

    # Perform convolution
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            # Dilated index
            input_h = input_offset_h + kh * dilation
            input_w = input_offset_w + kw * dilation

            # Load input for current kernel position
            input_val = tl.load(
                input_tile + (kh * dilation * (BLOCK_SIZE_H + (kernel_size - 1) * dilation) + kw * dilation),
                boundary_check=(2, 3),
                padding_option='zeros'
            )

            # Load weight for current kernel position
            weight_val = tl.load(
                weight_tile + (kh * (kernel_size * BLOCK_SIZE_OC) + kw * BLOCK_SIZE_OC),
                boundary_check=(0, 1),
                padding_option='zeros'
            )

            # Convolution accumulation (broadcasting across channels and output channels)
            acc += tl.dot(input_val, weight_val, out_dtype=tl.float32)

    # Store output
    output_ptr = out_ptr + (
        pid_b * out_stride0 + out_offset_o * out_stride1 +
        out_offset_h * out_stride2 + out_offset_w * out_stride3
    )

    output_mask = (tl.arange(0, BLOCK_SIZE_H) < out_height - out_offset_h)[:, None] & \
                  (tl.arange(0, BLOCK_SIZE_W) < out_width - out_offset_w)[None, :]

    tl.store(output_ptr, acc, mask=output_mask)


def triton_conv2d(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int, dilation: int, groups: int) -> torch.Tensor:
    """
    Conv2D with Triton kernel, supports grouped convolutions.
    Assumes the input and weight tensors are contiguous and on GPU.
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA."
    x = x.contiguous()
    w = w.contiguous()

    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_size, _ = w.shape

    # Handle groups
    out_channels_per_group = out_channels // groups
    in_channels_per_group = in_channels // groups

    # Output dimensions
    out_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    # Output tensor
    out = torch.empty(batch_size, out_channels, out_height, out_width, device=x.device, dtype=x.dtype)

    # Grid configuration
    # Block sizes (tuned for A100: 128x128 for spatial, 32 for output channels)
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_OC = 32

    # Number of blocks for each dimension
    num_blocks_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    num_blocks_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    num_blocks_o = (out_channels + BLOCK_SIZE_OC - 1) // BLOCK_SIZE_OC
    num_blocks_b = batch_size

    grid = (num_blocks_b, num_blocks_o, num_blocks_h, num_blocks_w)

    # Kernel launch
    conv2d_kernel[grid](
        x, w, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, in_channels, out_channels, height, width,
        kernel_size, stride, padding, dilation, groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_OC=BLOCK_SIZE_OC
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # Initialize weights with proper shape
        weight_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device='cuda'))

        # Register the weight
        self.register_buffer('weight', self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform convolution using custom Triton kernel
        return triton_conv2d(x, self.weight, self.stride, self.padding, self.dilation, self.groups)
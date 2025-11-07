import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_bn_scale_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    s_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride,
    pad,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
    TILE_HEIGHT: tl.constexpr,
    TILE_WIDTH: tl.constexpr,
    TILE_OUT_CHANNELS: tl.constexpr,
):
    # Define thread indices
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Calculate output coordinate
    h = pid_h * TILE_HEIGHT + tl.arange(0, TILE_HEIGHT)
    w = pid_w * TILE_WIDTH + tl.arange(0, TILE_WIDTH)
    h_offset = h + pad
    w_offset = w + pad

    # Create masks for valid output positions
    h_mask = h < output_height
    w_mask = w < output_width
    h_valid = h_offset < height
    w_valid = w_offset < width
    valid_mask = h_mask & w_mask & h_valid & w_valid

    # Load input block (B, C_in, H, W) -> (TILE_HEIGHT, TILE_WIDTH) per thread
    input_ptr = x_ptr + pid_batch * in_channels * height * width
    # Reorder to (C_in, H, W) for the current batch
    input_block = tl.load(
        input_ptr + in_channels * height * width * pid_batch,
        shape=(in_channels, height, width),
        dtype=tl.float16,
        mask=True,
    )

    # Compute output block: (TILE_HEIGHT, TILE_WIDTH) -> (TILE_OUT_CHANNELS, TILE_HEIGHT, TILE_WIDTH)
    output = tl.zeros((TILE_OUT_CHANNELS, TILE_HEIGHT, TILE_WIDTH), dtype=tl.float32)

    # Iterate over input channels and kernel weights
    for c_in in range(0, in_channels, BLOCK_SIZE):
        c_in_end = min(c_in + BLOCK_SIZE, in_channels)
        c_in_range = c_in_end - c_in

        # Load kernel weights for current input channel block
        w_block = tl.load(
            w_ptr + pid_out_c * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size,
            shape=(c_in_range, kernel_size, kernel_size),
            dtype=tl.float16,
            mask=(c_in_range, kernel_size, kernel_size),
        )

        # Load input patch for current block
        input_patch = tl.load(
            input_ptr + c_in * height * width,
            shape=(c_in_range, height, width),
            dtype=tl.float16,
            mask=(c_in_range, height, width),
        )

        # Perform conv: (c_in_range, H, W) x (c_in_range, k, k) -> (TILE_HEIGHT, TILE_WIDTH)
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Load input region at (h_offset + i, w_offset + j)
                h_idx = h_offset + i
                w_idx = w_offset + j
                valid = (h_idx < height) & (w_idx < width)
                patch = tl.load(
                    input_patch + h_idx * width + w_idx,
                    shape=(c_in_range,),
                    dtype=tl.float16,
                    mask=valid,
                )
                output += tl.dot(w_block[i, j], patch, out_dtype=tl.float32)

    # Load batch norm stats: (out_channels,)
    bn_weight = tl.load(b_ptr + pid_out_c, dtype=tl.float16, mask=pid_out_c < out_channels)
    bn_bias = tl.load(b_ptr + out_channels + pid_out_c, dtype=tl.float16, mask=pid_out_c < out_channels)
    scale = tl.load(s_ptr, dtype=tl.float16)

    # Apply batch norm and scaling
    # Output is (TILE_OUT_CHANNELS, TILE_HEIGHT, TILE_WIDTH)
    out_data = tl.load(
        out_ptr + pid_batch * out_channels * output_height * output_width + pid_out_c * output_height * output_width,
        shape=(TILE_HEIGHT, TILE_WIDTH),
        dtype=tl.float32,
        mask=valid_mask,
    )

    # Apply BN + scaling
    out_data = (out_data - bn_bias) * (bn_weight) + scale * out_data

    # Store result
    tl.store(
        out_ptr + pid_batch * out_channels * output_height * output_width + pid_out_c * output_height * output_width,
        out_data,
        mask=valid_mask,
    )


def triton_conv_bn_scale(x, w, b, s, kernel_size=3, stride=1, pad=1):
    # Ensure inputs are on GPU and contiguous
    x = x.contiguous().to(torch.float16)
    w = w.contiguous().to(torch.float16)
    b = b.contiguous().to(torch.float16)
    s = s.contiguous().to(torch.float16)

    # Extract dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels, _, _, _ = w.shape
    output_height = (height + 2 * pad - kernel_size) // stride + 1
    output_width = (width + 2 * pad - kernel_size) // stride + 1

    # Prepare output
    out = torch.empty(batch_size, out_channels, output_height, output_width, device=x.device, dtype=torch.float16)

    # Define block sizes
    BLOCK_SIZE = 16  # Input channel tiling
    TILE_HEIGHT = 16
    TILE_WIDTH = 16
    TILE_OUT_CHANNELS = 16

    # Grid: (batch, out_c, out_h, out_w)
    grid = lambda meta: (
        batch_size,
        (out_channels + meta["TILE_OUT_CHANNELS"] - 1) // meta["TILE_OUT_CHANNELS"],
        (output_height + meta["TILE_HEIGHT"] - 1) // meta["TILE_HEIGHT"],
        (output_width + meta["TILE_WIDTH"] - 1) // meta["TILE_WIDTH"],
    )

    # Launch kernel
    conv_bn_scale_kernel[grid](
        x,
        w,
        b,
        s,
        out,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size,
        stride,
        pad,
        output_height,
        output_width,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_HEIGHT=TILE_HEIGHT,
        TILE_WIDTH=TILE_WIDTH,
        TILE_OUT_CHANNELS=TILE_OUT_CHANNELS,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Fuse conv, bn, and scale into a single Triton kernel
        return triton_conv_bn_scale(
            x,
            self.conv.weight,
            torch.cat((self.bn.weight, self.bn.bias), dim=0),
            torch.tensor([self.scaling_factor], device=x.device),
            kernel_size=self.conv.kernel_size[0],
            stride=self.conv.stride[0],
            pad=self.conv.padding[0],
        )
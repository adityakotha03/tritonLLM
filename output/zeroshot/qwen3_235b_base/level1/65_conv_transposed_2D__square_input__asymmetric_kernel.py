import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose2d_kernel(
    x_ptr, weight_ptr, out_ptr,
    batch, out_channels, in_channels, height_out, width_out, height_in, width_in,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    output_padding_h, output_padding_w,
    groups,
    TILE_N: tl.constexpr, TILE_C: tl.constexpr, TILE_H: tl.constexpr, TILE_W: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Compute group indices
    group_out_ch = out_channels // groups
    group_in_ch = in_channels // groups

    # Offset weight pointer to current group
    weight_ptr += pid_g * group_out_ch * group_in_ch * kernel_h * kernel_w

    # Compute output spatial tile
    hw_tile_id = pid_hw
    out_h = height_out
    out_w = width_out
    hw_tiles_per_row = (out_w + TILE_W - 1) // TILE_W
    pid_h = hw_tile_id // hw_tiles_per_row
    pid_w = hw_tile_id % hw_tiles_per_row

    # Define offsets for the tile
    h_off = pid_h * TILE_H + tl.arange(0, TILE_H)
    w_off = pid_w * TILE_W + tl.arange(0, TILE_W)

    # Masks for spatial bounds
    h_mask = h_off < height_out
    w_mask = w_off < width_out
    hw_mask = h_mask[:, None] & w_mask[None, :]

    # Initialize output accumulator for this tile
    acc = tl.zeros((TILE_N, TILE_C, TILE_H, TILE_W), dtype=tl.float32)

    # Loop over input channels and kernel
    for ic in range(0, group_in_ch, TILE_C):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input spatial coordinates
                ih = (h_off - padding_h + kh * 1) // stride_h
                iw = (w_off - padding_w + kw * 1) // stride_w
                # Validity mask for input access
                ih_valid = (ih >= 0) & (ih < height_in)
                iw_valid = (iw >= 0) & (iw < width_in)
                input_mask = ih_valid[:, None] & iw_valid[None, :] & hw_mask

                # Load input: [batch, TILE_N, group_in_ch, TILE_H, TILE_W]
                c_off = ic + tl.arange(0, TILE_C)
                c_mask = c_off < group_in_ch
                c_mask = c_mask[:, None, None]

                input_ptrs = x_ptr + \
                    pid_n * in_channels * height_in * width_in + \
                    (pid_g * group_in_ch + c_off[None, :, None, None]) * height_in * width_in + \
                    ih[None, None, :, None] * width_in + \
                    iw[None, None, None, :]

                x_tile = tl.load(input_ptrs, mask=input_mask[None, None, :, :] & c_mask, other=0.0)

                # Load weights: [group_out_ch, TILE_C, 1, 1]
                w_ptrs = weight_ptr + \
                    (tl.arange(0, TILE_C)[None, :, None, None] * kernel_h * kernel_w * group_out_ch) + \
                    (kh * kernel_w + kw) * group_out_ch + \
                    tl.arange(0, group_out_ch)[:, None, None, None]

                w_mask = c_off < group_in_ch
                w_tile = tl.load(w_ptrs, mask=w_mask[None, :, None, None], other=0.0)

                # Perform outer product and accumulate: [TILE_N, group_out_ch, TILE_H, TILE_W]
                w_tile = w_tile[:, :, None, None]
                outer_prod = tl.dot(w_tile, x_tile, out_dtype=tl.float32)
                acc += outer_prod

    # Store output
    o_ptrs = out_ptr + \
        pid_n * out_channels * out_h * out_w + \
        (pid_g * group_out_ch + tl.arange(0, group_out_ch)[:, None, None, None]) * out_h * out_w + \
        h_off[None, None, :, None] * out_w + \
        w_off[None, None, None, :]

    o_mask = hw_mask[None, None, :, :] & (tl.arange(0, group_out_ch)[:, None, None, None] < group_out_ch)
    tl.store(o_ptrs, acc, mask=o_mask)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    output_padding: tuple,
    groups: int
):
    # Input shapes
    batch, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_pad_h, out_pad_w = output_padding

    # Compute output spatial dimensions
    height_out = (height_in - 1) * stride_h - 2 * pad_h + kernel_h + out_pad_h
    width_out = (width_in - 1) * stride_w - 2 * pad_w + kernel_w + out_pad_w

    # Output buffer
    out = torch.zeros(batch, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

    # Tile sizes
    TILE_N = 1
    TILE_C = 16
    TILE_H = 32
    TILE_W = 32

    # Grid dimensions
    hw_tiles = ((height_out + TILE_H - 1) // TILE_H) * ((width_out + TILE_W - 1) // TILE_W)
    grid = (batch, groups, hw_tiles)

    # Launch kernel
    _conv_transpose2d_kernel[grid](
        x, weight, out,
        batch, out_channels, in_channels,
        height_out, width_out, height_in, width_in,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        out_pad_h, out_pad_w,
        groups,
        TILE_N, TILE_C, TILE_H, TILE_W
    )

    # Add bias if provided
    if bias is not None:
        out += bias.view(1, out_channels, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.groups = groups

        # Initialize weight and optional bias
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose weight to match expected shape: (out_channels, in_channels // groups, k_h, k_w)
        weight_t = self.weight.permute(1, 0, 2, 3).contiguous()
        return triton_conv_transpose2d(
            x, weight_t, self.bias,
            self.stride, self.padding, self.output_padding, self.groups
        )
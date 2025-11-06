import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    out_depth,
    out_height,
    out_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Shared memory for storing tiles of input and weights
    # We will tile both input and weight tensors to fit in shared memory
    # Use the same tile size for all dimensions
    tile_size = TILE_SIZE
    # Compute indices for this block
    pid_batch = tl.program_id(0)
    pid_out_c = tl.program_id(1)
    pid_out_d = tl.program_id(2)
    pid_out_h = tl.program_id(3)
    pid_out_w = tl.program_id(4)

    # Offset within output tensor
    off_d = pid_out_d * tile_size
    off_h = pid_out_h * tile_size
    off_w = pid_out_w * tile_size

    # Initialize output tile
    out = tl.zeros((tile_size, tile_size, tile_size), dtype=tl.float32)

    # Loop over input channels and groups
    for c in range(0, in_channels, groups):
        # Calculate input channel offset
        ic = c

        # Load input tile: (tile_size, tile_size, tile_size)
        # Each thread handles one output element in the tile
        # We will use the same tiling strategy: tiles of size TILE_SIZE x TILE_SIZE x TILE_SIZE
        # Offset for input tensor
        input_off_d = off_d - padding
        input_off_h = off_h - padding
        input_off_w = off_w - padding

        # Determine which input elements are valid
        mask_d = input_off_d + tl.arange(0, tile_size) < out_depth
        mask_h = input_off_h + tl.arange(0, tile_size) < out_height
        mask_w = input_off_w + tl.arange(0, tile_size) < out_width

        # Check if the output position is within bounds
        out_d = off_d + tl.arange(0, tile_size)
        out_h = off_h + tl.arange(0, tile_size)
        out_w = off_w + tl.arange(0, tile_size)

        # Compute input indices for the current output tile
        input_d = out_d * stride - padding
        input_h = out_h * stride - padding
        input_w = out_w * stride - padding

        # Apply dilation
        input_d = input_d * dilation
        input_h = input_h * dilation
        input_w = input_w * dilation

        # Check for valid input positions (within input bounds)
        valid_d = (input_d >= 0) & (input_d < depth)
        valid_h = (input_h >= 0) & (input_h < height)
        valid_w = (input_w >= 0) & (input_w < width)

        # Create valid mask for input data
        valid_mask = valid_d[:, None, None] & valid_h[None, :, None] & valid_w[None, None, :]

        # Create offsets into input tensor
        input_offsets = (input_d[:, None, None] * height * width +
                         input_h[None, :, None] * width +
                         input_w[None, None, :])

        # Load input tile (only if valid)
        x_tile = tl.load(x_ptr + (pid_batch * in_channels * depth * height * width +
                                  ic * depth * height * width + input_offsets),
                         mask=valid_mask, other=0.0)

        # Now load weight tile: (out_channels, in_channels_per_group, kernel_size, kernel_size, kernel_size)
        # We need to transpose this to match layout of weights
        w_tile = tl.load(w_ptr + (pid_out_c * groups * in_channels * kernel_size * kernel_size * kernel_size +
                                  (c // groups) * in_channels * kernel_size * kernel_size * kernel_size +
                                  (tl.arange(0, kernel_size)[:, None, None] * kernel_size * kernel_size +
                                   tl.arange(0, kernel_size)[None, :, None] * kernel_size +
                                   tl.arange(0, kernel_size)[None, None, :]) * in_channels),
                         mask=(tl.arange(0, kernel_size)[:, None, None] < kernel_size) &
                               (tl.arange(0, kernel_size)[None, :, None] < kernel_size) &
                               (tl.arange(0, kernel_size)[None, None, :] < kernel_size),
                         other=0.0)

        # Compute output tile
        # Convolution over input tile and weight tile
        # Use fused reduction over kernel size dimensions
        for k_d in range(kernel_size):
            for k_h in range(kernel_size):
                for k_w in range(kernel_size):
                    out += x_tile * w_tile[k_d, k_h, k_w]

        # Accumulate into output tile
        out += tl.load(out_ptr + (pid_batch * out_channels * out_depth * out_height * out_width +
                                  pid_out_c * out_depth * out_height * out_width +
                                  out_d[:, None, None] * out_height * out_width +
                                  out_h[None, :, None] * out_width +
                                  out_w[None, None, :]),
                       mask=valid_mask, other=0.0)

    # Store the output
    out_offsets = (pid_batch * out_channels * out_depth * out_height * out_width +
                   pid_out_c * out_depth * out_height * out_width +
                   out_d[:, None, None] * out_height * out_width +
                   out_h[None, :, None] * out_width +
                   out_w[None, None, :])

    tl.store(out_ptr + out_offsets, out, mask=valid_mask)


def triton_conv_transpose3d(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, stride: int, padding: int, dilation: int, output_padding: int, groups: int):
    """
    Custom Triton-based transposed 3D convolution with automatic tuning.

    Args:
        x: Input tensor (B, C_in, D, H, W)
        w: Weight tensor (C_out, C_in, K, K, K)
        bias: Optional bias (C_out,)
        stride: Stride value
        padding: Padding value
        dilation: Dilation value
        output_padding: Output padding value
        groups: Number of groups

    Returns:
        Output tensor (B, C_out, D_out, H_out, W_out)
    """
    assert x.is_cuda and w.is_cuda, "Tensors must be on CUDA"
    x = x.contiguous()
    w = w.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    B, C_in, D, H, W = x.shape
    C_out, _, K, _, _ = w.shape

    # Compute output dimensions
    D_out = (D - 1) * stride - 2 * padding + K + output_padding
    H_out = (H - 1) * stride - 2 * padding + K + output_padding
    W_out = (W - 1) * stride - 2 * padding + K + output_padding

    # Initialize output tensor
    out = torch.empty(B, C_out, D_out, H_out, W_out, device=x.device, dtype=x.dtype)

    # Grid: (batch_size, out_channels, out_depth // TILE_SIZE, out_height // TILE_SIZE, out_width // TILE_SIZE)
    grid = lambda meta: (B, C_out, (D_out + meta["TILE_SIZE"] - 1) // meta["TILE_SIZE"],
                         (H_out + meta["TILE_SIZE"] - 1) // meta["TILE_SIZE"],
                         (W_out + meta["TILE_SIZE"] - 1) // meta["TILE_SIZE"])

    # Launch kernel
    # Use TILE_SIZE = 32 for good balance between shared memory usage and occupancy
    conv_transpose3d_kernel[grid](
        x,
        w,
        out,
        B,
        C_in,
        C_out,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        K,
        stride,
        padding,
        dilation,
        groups,
        BLOCK_SIZE=128,
        TILE_SIZE=32
    )

    # Add bias if present
    if bias is not None:
        out += bias.view(1, C_out, 1, 1, 1)

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 dilation: int = 1, groups: int = 1, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.output_padding, self.groups
        )
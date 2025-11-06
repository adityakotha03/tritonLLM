import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_transpose_kernel(
    x_ptr,  # Pointer to input tensor (batch, in_channels, length)
    w_ptr,  # Pointer to weight tensor (out_channels, in_channels, kernel_size)
    out_ptr,  # Pointer to output tensor (batch, out_channels, length_out)
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    length: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    output_padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Compute the output length
    length_out = (length - 1) * stride + kernel_size - 2 * padding + output_padding

    # Define thread indices
    pid_batch = tl.program_id(0)
    pid_out_ch = tl.program_id(1)
    pid_x = tl.program_id(2)

    # Block offsets
    block_offset_x = pid_x * BLOCK_SIZE
    block_offset_out_ch = pid_out_ch * TILE_SIZE

    # Compute thread indices within block
    off_x = block_offset_x + tl.arange(0, BLOCK_SIZE)
    off_out_ch = block_offset_out_ch + tl.arange(0, TILE_SIZE)

    # Create masks for boundary conditions
    mask_x = off_x < length_out
    mask_out_ch = off_out_ch < out_channels

    # Compute input length and other parameters
    input_length = length
    pad_start = padding

    # Compute input channel block
    block_in_ch = 32  # Coalesced tiling for in_channels
    off_in_ch = tl.arange(0, block_in_ch)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, TILE_SIZE), dtype=tl.float32)

    # Loop over input channels and kernel
    for idx_in_ch in range(0, in_channels, block_in_ch):
        # Load input data (block of in_channels)
        off_in_ch_block = idx_in_ch + off_in_ch
        mask_in_ch = off_in_ch_block < in_channels

        # Load input data: (BLOCK_SIZE, block_in_ch)
        x_load = tl.load(
            x_ptr + pid_batch * in_channels * input_length + off_in_ch_block[:, None] * input_length + off_x[None, :],
            mask=(mask_in_ch[:, None] & mask_x[None, :]),
            other=0.0,
        )

        # Load weight: (TILE_SIZE, block_in_ch, kernel_size)
        w_load = tl.load(
            w_ptr + pid_out_ch * in_channels * kernel_size + idx_in_ch * kernel_size + off_in_ch_block[:, None, None] * kernel_size + tl.arange(0, kernel_size)[None, None, :],
            mask=(mask_out_ch[:, None, None] & mask_in_ch[:, None, None] & (tl.arange(0, kernel_size) < kernel_size)),
            other=0.0,
        )

        # Compute stride-adjusted output index
        out_indices = off_x[:, None] - tl.arange(0, kernel_size)[None, :] * stride
        out_mask = (out_indices >= 0) & (out_indices < input_length)

        # Multiply and accumulate
        for k in range(kernel_size):
            # Extract valid output indices
            idx = out_indices[:, k]
            mask = out_mask[:, k]

            # Load input for this kernel shift
            x_val = tl.load(
                x_ptr + pid_batch * in_channels * input_length + off_in_ch_block[:, None] * input_length + idx[None, :],
                mask=(mask_in_ch[:, None] & mask[None, :]),
                other=0.0,
            )

            # Multiply with weight
            w_val = w_load[:, :, k]
            acc += tl.dot(x_val, w_val.T)  # (BLOCK_SIZE, TILE_SIZE)

    # Clamp output to valid length
    out_mask_x = off_x < length_out

    # Store output
    out = acc
    tl.store(
        out_ptr + pid_batch * out_channels * length_out + off_out_ch[None, :] * length_out + off_x[:, None],
        out,
        mask=(mask_out_ch[None, :] & out_mask_x[:, None]),
    )


def triton_conv1d_transpose(x: torch.Tensor, w: torch.Tensor, stride: int = 1, padding: int = 0, output_padding: int = 0) -> torch.Tensor:
    """
    Performs transposed 1D convolution using Triton kernel.
    Input: (batch, in_channels, length)
    Weight: (out_channels, in_channels, kernel_size)
    Output: (batch, out_channels, length_out)
    """
    batch_size, in_channels, length = x.shape
    out_channels, _, kernel_size = w.shape

    # Compute output length
    length_out = (length - 1) * stride + kernel_size - 2 * padding + output_padding

    # Ensure input is contiguous
    x = x.contiguous()
    w = w.contiguous()

    # Allocate output
    out = torch.empty(batch_size, out_channels, length_out, device=x.device, dtype=x.dtype)

    # Configure kernel launch parameters
    BLOCK_SIZE = 128  # Tile size along output length
    TILE_SIZE = 32   # Tile size along out_channels

    # Grid dimensions
    grid = lambda meta: (
        batch_size,
        triton.cdiv(out_channels, meta["TILE_SIZE"]),
        triton.cdiv(length_out, meta["BLOCK_SIZE"]),
    )

    # Launch kernel
    conv1d_transpose_kernel[grid](
        x,
        w,
        out,
        batch_size,
        in_channels,
        out_channels,
        length,
        kernel_size,
        stride,
        padding,
        output_padding,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # Use non-learnable weight (for Triton kernel compatibility)
        self.register_buffer("weight", torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float16))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = bias
        if bias:
            self.register_buffer("bias", torch.randn(out_channels, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float16 to leverage Tensor Cores
        x = x.to(torch.float16)

        # Apply Triton-based conv1d transpose
        out = triton_conv1d_transpose(x, self.weight, self.stride, self.padding, self.output_padding)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)

        # Cast back to float32 if needed (for downstream models)
        if x.dtype == torch.float32:
            out = out.to(torch.float32)

        return out
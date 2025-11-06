import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    batch_size, in_channels, out_channels, height_in, width_in,
    kernel_size, stride, padding, output_padding,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Block indices
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    pid_c = tl.program_id(3)

    # Block offsets
    block_h_start = pid_h * BLOCK_SIZE_H
    block_w_start = pid_w * BLOCK_SIZE_W
    block_c_start = pid_c * BLOCK_SIZE_C

    # Output dimensions
    height_out = (height_in - 1) * stride + kernel_size + output_padding
    width_out = (width_in - 1) * stride + kernel_size + output_padding

    # Load input (x) block
    x_offsets_h = block_h_start + tl.arange(0, BLOCK_SIZE_H)[:, None]
    x_offsets_w = block_w_start + tl.arange(0, BLOCK_SIZE_W)[None, :]
    x_mask = (x_offsets_h < height_in) & (x_offsets_w < width_in)
    x_offsets = (
        pid_batch * x_stride0 +
        tl.arange(0, BLOCK_SIZE_H)[:, None] * x_stride2 +
        tl.arange(0, BLOCK_SIZE_W)[None, :] * x_stride3
    )
    x_data = tl.load(
        x_ptr + x_offsets,
        mask=x_mask[:, :, None],
        other=0.0
    )  # Shape: (BLOCK_SIZE_H, BLOCK_SIZE_W, in_channels)

    # Load weights (w) for the output channel block
    w_offsets_h = tl.arange(0, kernel_size)[:, None]
    w_offsets_w = tl.arange(0, kernel_size)[None, :]
    w_offsets_c = tl.arange(0, BLOCK_SIZE_C)[:, None, None]  # Output channel block
    w_offsets = (
        w_stride0 * tl.arange(0, out_channels)[:, None, None] +
        w_stride1 * tl.arange(0, in_channels)[None, :, None] +
        w_stride2 * w_offsets_h +
        w_stride3 * w_offsets_w
    )
    w_data = tl.load(
        w_ptr + w_offsets,
        mask=(tl.arange(0, out_channels)[:, None, None] == pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)[:, None, None]) &
              (tl.arange(0, in_channels)[None, :, None] < in_channels) &
              (w_offsets_h < kernel_size) &
              (w_offsets_w < kernel_size),
        other=0.0
    )  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

    # Accumulate output
    out_offsets_h = block_h_start * stride + w_offsets_h
    out_offsets_w = block_w_start * stride + w_offsets_w
    out_offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)[:, None, None]

    # Output offsets
    out_offsets = (
        pid_batch * out_stride0 +
        out_offsets_c * out_stride1 +
        out_offsets_h * out_stride2 +
        out_offsets_w * out_stride3
    )
    out_mask = (out_offsets_h < height_out) & (out_offsets_w < width_out)

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)

    # Compute output for the current block
    for c_in in range(0, in_channels, 1):
        x_c = x_data[:, :, c_in : c_in + 1]  # (BLOCK_SIZE_H, BLOCK_SIZE_W, 1)
        w_c = w_data[:, c_in : c_in + 1, :, :]  # (out_channels, 1, kernel_size, kernel_size)
        # Conv transpose: elementwise multiply and sum over kernel_size and in_channels
        # Use broadcasting for matmul-like operation
        out_temp = tl.sum(x_c * w_c, axis=2)  # (out_channels, BLOCK_SIZE_H, BLOCK_SIZE_W)
        out_temp = out_temp[:, None, :]  # (out_channels, 1, BLOCK_SIZE_H, BLOCK_SIZE_W)
        acc += out_temp

    # Store result
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=out_mask[:, :, None] & (out_offsets_c < out_channels)
    )


def triton_conv_transpose2d(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    groups: int = 1,
):
    # Input validation
    assert x.dim() == 4, "Expected 4D tensor"
    assert w.dim() == 4, "Expected 4D kernel tensor"
    assert x.is_cuda and w.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == w.dtype, "Input and weight must have same dtype"
    assert groups == 1, "Only single group supported for now"

    # Handle dtype: use bfloat16 for faster Tensor Core usage on A100
    if x.dtype == torch.bfloat16:
        input_dtype = tl.bfloat16
    elif x.dtype == torch.float16:
        input_dtype = tl.float16
    else:
        input_dtype = tl.float32

    # Shape and strides
    batch_size, in_channels, height_in, width_in = x.shape
    out_channels, _, kernel_size, _ = w.shape

    # Output shape
    height_out = (height_in - 1) * stride + kernel_size + output_padding
    width_out = (width_in - 1) * stride + kernel_size + output_padding

    # Output tensor
    out = torch.empty(batch_size, out_channels, height_out, width_out, device=x.device, dtype=x.dtype)

    # Strides
    x_stride0, x_stride1, x_stride2, x_stride3 = x.stride()
    w_stride0, w_stride1, w_stride2, w_stride3 = w.stride()
    out_stride0, out_stride1, out_stride2, out_stride3 = out.stride()

    # Determine block sizes
    # Use 128, 128 for H/W; 32 for C to fit shared memory and warp limits
    BLOCK_SIZE_H = 128
    BLOCK_SIZE_W = 128
    BLOCK_SIZE_C = 32

    # Grid dimensions
    num_blocks_h = triton.cdiv(height_in, BLOCK_SIZE_H)
    num_blocks_w = triton.cdiv(width_in, BLOCK_SIZE_W)
    num_blocks_c = triton.cdiv(out_channels, BLOCK_SIZE_C)
    grid = (batch_size, num_blocks_h, num_blocks_w, num_blocks_c)

    # Launch kernel
    conv_transpose2d_kernel[grid](
        x_ptr=x, w_ptr=w, out_ptr=out,
        x_stride0=x_stride0, x_stride1=x_stride1, x_stride2=x_stride2, x_stride3=x_stride3,
        w_stride0=w_stride0, w_stride1=w_stride1, w_stride2=w_stride2, w_stride3=w_stride3,
        out_stride0=out_stride0, out_stride1=out_stride1, out_stride2=out_stride2, out_stride3=out_stride3,
        batch_size=batch_size, in_channels=in_channels, out_channels=out_channels,
        height_in=height_in, width_in=width_in,
        kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
        BLOCK_SIZE_H=BLOCK_SIZE_H, BLOCK_SIZE_W=BLOCK_SIZE_W, BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=4,
        num_stages=2
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super().__init__()
        # Initialize weight with fan-in/fan-out normalization
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.bfloat16, device="cuda")
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.bfloat16, device="cuda"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert input to bfloat16 for optimal performance on A100
        x = x.to(torch.bfloat16)
        self.weight = self.weight.to(torch.bfloat16)
        out = triton_conv_transpose2d(
            x, self.weight,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups
        )
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
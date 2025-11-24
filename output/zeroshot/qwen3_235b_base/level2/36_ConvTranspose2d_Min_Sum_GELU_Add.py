import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def min_sum_gelu_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_stride, channel_stride, height_stride, width_stride,
    bias_batch_stride, bias_channel_stride, bias_height_stride, bias_width_stride,
    B, H, W,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offset_w = tl.arange(0, BLOCK_W)
    mask_w = offset_w < W

    # Pointers for spatial block in input
    x_block_ptr = (
        x_ptr + pid_b * batch_stride +
        pid_h * BLOCK_H * height_stride +
        offset_w * width_stride
    )

    # Load bias value (scalar per batch element, broadcasted)
    bias = tl.load(bias_ptr + pid_b * bias_batch_stride) if bias_batch_stride == 0 else tl.load(bias_ptr)

    # Initialize reduction for min and sum
    min_val = tl.full([BLOCK_W], value=float('inf'), dtype=tl.float32)
    sum_val = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Iterate over channels
    for c in range(0, channel_stride):
        x_ptrs = x_block_ptr + c * channel_stride
        x = tl.load(x_ptrs, mask=mask_w, other=float('inf'))

        # Update min: min_val = minimum(min_val, x)
        min_val = tl.minimum(min_val, x)

        # Accumulate sum
        sum_val += x

    # Apply GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_gelu = sum_val
    x_cubed = x_gelu * x_gelu * x_gelu
    inner = 0.044715 * x_cubed + x_gelu
    tanh_inner = tl.tanh(0.79788456 * inner)  # 0.79788456 = sqrt(2/pi)
    gelu_out = 0.5 * x_gelu * (1.0 + tanh_inner)

    # Add bias
    gelu_out += bias

    # Store output: output shape is [B, 1, H, W] -> we write to [pid_b, 0, pid_h*BLOCK_H:..., :]
    out_block_ptr = (
        out_ptr +
        pid_b * B * H * W +
        pid_h * BLOCK_H * W +
        offset_w
    )
    for i in range(BLOCK_H):
        mask_h = (pid_h * BLOCK_H + i) < H
        mask = mask_w & mask_h
        tl.store(out_block_ptr + i * W, gelu_out, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of Model using fused Triton kernel for min, sum, gelu, and bias add.
    The ConvTranspose2d is kept as-is since it's highly optimized in PyTorch,
    but the subsequent operations are fused into a single kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)  # Shape: [B, C, H, W]

        B, C, H, W = x.shape

        # Allocate output tensor: [B, 1, H, W] -> [B, H, W]
        out = torch.empty((B, H, W), dtype=x.dtype, device=x.device)

        # Launch Triton kernel
        # We process H in blocks
        BLOCK_H = 16
        BLOCK_W = 64

        grid = (B, triton.cdiv(H, BLOCK_H))

        min_sum_gelu_bias_kernel[grid](
            x_ptr=x,
            bias_ptr=self.bias,
            out_ptr=out,
            batch_stride=x.stride(0),
            channel_stride=x.stride(1),
            height_stride=x.stride(2),
            width_stride=x.stride(3),
            bias_batch_stride=self.bias.stride(0) if self.bias.stride(0) == 0 else 1,
            bias_channel_stride=self.bias.stride(1) if len(self.bias.shape) > 1 else 1,
            bias_height_stride=self.bias.stride(2) if len(self.bias.shape) > 2 else 1,
            bias_width_stride=self.bias.stride(3) if len(self.bias.shape) > 3 else 1,
            B=B, H=H, W=W,
            BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        )

        # Reshape output to [B, 1, H, W] to match original
        return out.unsqueeze(1)
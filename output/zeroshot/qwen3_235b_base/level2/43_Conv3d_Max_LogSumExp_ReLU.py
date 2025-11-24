import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def logsumexp_relu_kernel(
    input_ptr, output_ptr, batch_stride, channel_stride, depth_stride, height_stride, width_stride,
    batch_size, channels, depth, height, width,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # 2D block across (depth, height, width) and reduce over channels
    pid_b = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    # Offset for current batch
    input_off_b = pid_b * batch_stride
    output_off_b = pid_b * depth_stride

    # Flattened spatial indices
    dhw_offset = pid_dhw * BLOCK_D * BLOCK_H * BLOCK_W
    d_offsets = (dhw_offset + tl.arange(0, BLOCK_D * BLOCK_H * BLOCK_W)) // (BLOCK_H * BLOCK_W)
    hw_offset = (dhw_offset + tl.arange(0, BLOCK_D * BLOCK_H * BLOCK_W)) % (BLOCK_H * BLOCK_W)
    h_offsets = hw_offset // BLOCK_W
    w_offsets = hw_offset % BLOCK_W

    # Bounds checking for spatial dims
    d_mask = d_offsets < depth
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    valid_mask = d_mask & h_mask & w_mask

    # Initialize max and sum for logsumexp
    max_val = tl.full([BLOCK_D * BLOCK_H * BLOCK_W], value=float("-inf"), dtype=tl.float32)
    sum_val = tl.zeros([BLOCK_D * BLOCK_H * BLOCK_W], dtype=tl.float32)

    # Channel loop for logsumexp reduction
    for c in range(0, channels, BLOCK_C):
        c_offsets = c + tl.arange(0, BLOCK_C)
        channel_mask = c_offsets < channels
        mask = channel_mask[None, :] & valid_mask[:, None]

        # Load block of input: [BLOCK_D*H*W, BLOCK_C]
        offsets = input_off_b + c_offsets[None, :] * channel_stride + \
                  d_offsets[:, None] * depth_stride + h_offsets[:, None] * height_stride + \
                  w_offsets[:, None] * width_stride
        vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))

        # Update max
        vals_fp32 = vals.to(tl.float32)
        block_max = tl.max(vals_fp32, axis=1)
        max_val = tl.maximum(max_val, block_max)

    # Broadcast max and compute exp(x - max)
    max_val = max_val[:, None]
    sum_val = tl.zeros([BLOCK_D * BLOCK_H * BLOCK_W], dtype=tl.float32)
    for c in range(0, channels, BLOCK_C):
        c_offsets = c + tl.arange(0, BLOCK_C)
        channel_mask = c_offsets < channels
        mask = channel_mask[None, :] & valid_mask[:, None]

        offsets = input_off_b + c_offsets[None, :] * channel_stride + \
                  d_offsets[:, None] * depth_stride + h_offsets[:, None] * height_stride + \
                  w_offsets[:, None] * width_stride
        vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)

        exp_vals = tl.exp(vals - max_val)
        sum_val += tl.sum(exp_vals, axis=1)

    # Final logsumexp + ReLU: log(sum) + max, then relu
    result = tl.log(sum_val) + max_val
    result_relu = tl.where(result > 0, result, 0.0)

    # Store output: output shape is (batch, 1, depth, height, width)
    out_offsets = output_off_b + d_offsets * depth_stride + h_offsets * height_stride + w_offsets * width_stride
    tl.store(output_ptr + out_offsets, result_relu, mask=valid_mask)


def triton_logsumexp_relu(x):
    batch_size, channels, depth, height, width = x.shape
    out = torch.empty((batch_size, 1, depth, height, width), dtype=torch.float32, device=x.device)

    # Use tiling for spatial dimensions
    BLOCK_C = 16
    BLOCK_D = 4
    BLOCK_H = 8
    BLOCK_W = 16

    # Number of blocks
    num_dhw = (depth * height * width + BLOCK_D * BLOCK_H * BLOCK_W - 1) // (BLOCK_D * BLOCK_H * BLOCK_W)
    grid = (batch_size, num_dhw)

    logsumexp_relu_kernel[grid](
        x, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        batch_size, channels, depth, height, width,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for fused logsumexp + ReLU.
    Conv3d and MaxPool3d are kept as native PyTorch ops due to complexity and efficiency.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = triton_logsumexp_relu(x)
        return x
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_tanh_tanh_kernel(
    x_ptr,                  # pointer to input (after conv), shape (batch, out_channels, height, width)
    output_ptr,             # pointer to output
    batch_stride,           # stride for batch dim
    channel_stride,         # stride for channel dim
    height_stride,          # stride for height dim
    width: tl.constexpr,   # width of feature map
    height: tl.constexpr,  # height of feature map
    out_channels: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # 2D grid: each block handles a subset of batch and spatial dimensions
    batch_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    # Compute offsets for spatial block
    hw_offset = hw_pid * BLOCK_SIZE_HW
    hw_offsets = hw_offset + tl.arange(0, BLOCK_SIZE_HW)
    hw_mask = hw_offsets < height * width

    # Batch indices
    batch_offsets = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offsets < batch_stride

    # Broadcast masks
    mask = hw_mask[None, :] & batch_mask[:, None]
    mask = tl.reshape(mask, (BLOCK_SIZE_BATCH * BLOCK_SIZE_HW,))

    # Flatten batch and spatial dims
    flat_offsets = tl.reshape(
        batch_offsets[:, None] * batch_stride + hw_offsets[None, :] * width,
        (BLOCK_SIZE_BATCH * BLOCK_SIZE_HW,)
    )

    # Load all elements across channels for this batch-spatial block
    total_elements = BLOCK_SIZE_BATCH * BLOCK_SIZE_HW * out_channels
    input_offsets = tl.arange(0, total_elements)
    input_mask = input_offsets < (batch_stride * out_channels * height * width)
    
    # Reshape to (BLOCK_SIZE_BATCH, out_channels, BLOCK_SIZE_HW)
    grouped_offsets = tl.reshape(input_offsets, (BLOCK_SIZE_BATCH, out_channels, BLOCK_SIZE_HW))
    grouped_mask = tl.reshape(input_mask, (BLOCK_SIZE_BATCH, out_channels, BLOCK_SIZE_HW))

    # Base offset into x_ptr
    base_offset = flat_offsets[:, None]  # (BLOCK_SIZE_BATCH*BLOCK_SIZE_HW, 1)
    base_offset = tl.reshape(base_offset, (BLOCK_SIZE_BATCH, 1, BLOCK_SIZE_HW))
    
    # Channel offsets
    channel_offsets = tl.arange(0, out_channels) * channel_stride  # (out_channels,)
    channel_offsets = channel_offsets[None, :, None]  # (1, out_channels, 1)

    # Final offsets
    offsets = base_offset + channel_offsets
    data = tl.load(x_ptr + offsets, mask=grouped_mask, other=-float('inf'))

    # Compute min across channel: (BLOCK_SIZE_BATCH, BLOCK_SIZE_HW)
    min_vals = tl.min(data, axis=1)

    # First tanh
    tanh1 = tl.where(mask, tl.tanh(min_vals), 0.0)
    # Second tanh
    tanh2 = tl.where(mask, tl.tanh(tanh1), 0.0)

    # Store result
    tl.store(output_ptr + flat_offsets, tanh2, mask=tl.reshape(mask, flat_offsets.shape))


# Fused Conv + Min + Tanh + Tanh via custom Triton kernel wrapper
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Apply convolution first (use PyTorch's optimized Conv2d)
        x = self.conv(x)

        # Get shapes
        batch, ch, h, w = x.shape
        assert ch == self.conv.out_channels

        # Output tensor
        out = torch.empty((batch, 1, h, w), dtype=x.dtype, device=x.device)

        # Strides
        batch_stride = ch * h * w
        channel_stride = h * w
        height_stride = w

        # Define block sizes
        BLOCK_SIZE_BATCH = triton.next_power_of_2(batch)
        while BLOCK_SIZE_BATCH > 32:  # limit block size to avoid too many threads
            BLOCK_SIZE_BATCH //= 2
        BLOCK_SIZE_BATCH = max(1, BLOCK_SIZE_BATCH)

        BLOCK_SIZE_HW = triton.next_power_of_2(h * w)
        while BLOCK_SIZE_HW > 1024:
            BLOCK_SIZE_HW //= 2
        BLOCK_SIZE_HW = min(BLOCK_SIZE_HW, 1024)

        # Grid
        grid = (
            triton.cdiv(batch, BLOCK_SIZE_BATCH),
            triton.cdiv(h * w, BLOCK_SIZE_HW)
        )

        # Launch kernel
        min_tanh_tanh_kernel[grid](
            x,
            out,
            batch_stride,
            channel_stride,
            height_stride,
            w,
            h,
            ch,
            BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
            BLOCK_SIZE_HW=BLOCK_SIZE_HW,
        )

        return out
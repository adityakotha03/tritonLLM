import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def sum_channel_kernel(
    x_ptr,          # pointer to input tensor (B, C, D, H, W)
    output_ptr,     # pointer to output tensor (B, 1, D, H, W)
    B, C, D, H, W,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_outb, stride_outd, stride_outh, stride_outw,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # Calculate offsets
    offset_b = pid_b
    offset_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)[:, None]
    offset_hw = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)[None, :]

    # Mask for D and HW dimensions
    mask_d = offset_d < D
    mask_hw = offset_hw < (H * W)
    mask_dhw = mask_d and mask_hw

    # Expand offsets to cover full D x HW block
    offset_d = tl.broadcast_to(offset_d, (BLOCK_SIZE_D, BLOCK_SIZE_HW))
    offset_hw = tl.broadcast_to(offset_hw, (BLOCK_SIZE_D, BLOCK_SIZE_HW))

    # Convert HW flat index to H, W
    h_idx = (offset_hw // W) % H
    w_idx = offset_hw % W
    mask_hw = (h_idx < H) & (w_idx < W)
    mask = mask_dhw & mask_hw

    # Base offset for this (b, d, h, w) position in input
    x_block_base = x_ptr + offset_b * stride_xb + offset_d * stride_xd + h_idx * stride_xh + w_idx * stride_xw

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_HW), dtype=tl.float32)

    # Sum over channels in blocks
    for c in range(0, C, BLOCK_SIZE_C):
        c_offsets = c + tl.arange(0, BLOCK_SIZE_C)
        mask_c = c_offsets < C
        mask_c = tl.broadcast_to(mask_c[None, :], (BLOCK_SIZE_D, BLOCK_SIZE_HW, BLOCK_SIZE_C))
        mask_c = tl.load(mask_c)
        x_ptrs = x_block_base + c_offsets[None, None, :] * stride_xc
        x = tl.load(x_ptrs, mask=mask_c, other=0.0)
        x_sum = tl.sum(x, axis=2)  # reduce over C
        acc += x_sum

    # Write output
    output_offset = output_ptr + offset_b * stride_outb + offset_d * stride_outd + h_idx * stride_outh + w_idx * stride_outw
    tl.store(output_offset, acc, mask=mask)


def triton_sum_channel(x):
    B, C, D, H, W = x.shape
    output = torch.empty((B, 1, D, H, W), dtype=torch.float32, device=x.device)

    # Flatten H and W for tiling
    HW = H * W
    # Choose block sizes
    BLOCK_SIZE_C = 16
    BLOCK_SIZE_D = 8
    BLOCK_SIZE_HW = 32

    # Grid dimensions
    grid = (B, triton.cdiv(D, BLOCK_SIZE_D), triton.cdiv(HW, BLOCK_SIZE_HW))

    sum_channel_kernel[grid](
        x_ptr=x.contiguous(),
        output_ptr=output,
        B=B, C=C, D=D, H=H, W=W,
        stride_xb=x.stride(0), stride_xc=x.stride(1), stride_xd=x.stride(2),
        stride_xh=x.stride(3), stride_xw=x.stride(4),
        stride_outb=output.stride(0), stride_outd=output.stride(2),
        stride_outh=output.stride(3), stride_outw=output.stride(4),
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return output


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for channel-wise sum to reduce memory bandwidth usage.
    The transposed convolution and max pooling layers are kept as native PyTorch ops
    since they are already highly optimized and use cuDNN, but the final torch.sum(dim=1)
    is replaced with a fused Triton kernel that reduces memory traffic.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = triton_sum_channel(x)
        return x
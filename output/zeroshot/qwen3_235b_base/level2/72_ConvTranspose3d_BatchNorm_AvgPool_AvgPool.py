import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_3d_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    depth,
    height,
    width,
    pool_size: tl.constexpr,
    output_depth,
    output_height,
    output_width,
    input_stride_d,
    input_stride_h,
    input_stride_w,
    output_stride_d,
    output_stride_h,
    output_stride_w,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Compute starting positions for this block
    d_start = pid_d * BLOCK_SIZE_D
    hw_start = pid_hw * BLOCK_SIZE_HW

    # Define offsets for the block
    d_offsets = d_start + tl.arange(0, BLOCK_SIZE_D)
    h_offsets_base = hw_start // output_width
    w_offsets_base = hw_start % output_width
    h_offsets = h_offsets_base + tl.arange(0, BLOCK_SIZE_HW // output_width + 1)
    w_offsets = w_offsets_base + tl.arange(0, BLOCK_SIZE_HW % output_width + 1)

    # Clamp to valid ranges
    d_mask = d_offsets < output_depth
    h_mask = h_offsets < output_height
    w_mask = w_offsets < output_width

    # Broadcast masks appropriately
    d_mask = tl.reshape(d_mask, (BLOCK_SIZE_D, 1, 1))
    h_mask = tl.reshape(h_mask, (1, BLOCK_SIZE_HW // output_width + 1, 1))
    w_mask = tl.reshape(w_mask, (1, 1, BLOCK_SIZE_HW % output_width + 1))

    # Iterate over channels
    for c in range(n_channels):
        # Iterate over pooled output locations in this block
        for di in range(BLOCK_SIZE_D):
            for hi in range(BLOCK_SIZE_HW // output_width + 1):
                for wi in range(BLOCK_SIZE_HW % output_width + 1):
                    do = d_start + di
                    ho = h_offsets_base + hi
                    wo = w_offsets_base + wi
                    if do >= output_depth or ho >= output_height or wo >= output_width:
                        continue

                    # Map to input region
                    d_start_in = do * pool_size
                    h_start_in = ho * pool_size
                    w_start_in = wo * pool_size

                    acc = 0.0
                    count = 0
                    for kd in range(pool_size):
                        for kh in range(pool_size):
                            for kw in range(pool_size):
                                d_in = d_start_in + kd
                                h_in = h_start_in + kh
                                w_in = w_start_in + kw
                                if d_in < depth and h_in < height and w_in < width:
                                    input_offset = c * input_stride_d + d_in * input_stride_h + h_in * input_stride_w + w_in
                                    val = tl.load(input_ptr + input_offset)
                                    acc += val
                                    count += 1

                    # Store average
                    output_offset = c * output_stride_d + do * output_stride_h + ho * output_stride_w + wo
                    tl.store(output_ptr + output_offset, acc / count)


def triton_avg_pool_3d(x, kernel_size=2):
    *batch_shape, d, h, w = x.shape
    flat_batch = torch.prod(torch.tensor(batch_shape)).item()
    x_reshaped = x.view(flat_batch, -1, d, h, w)
    n, c, d, h, w = x_reshaped.shape

    out_d, out_h, out_w = d // kernel_size, h // kernel_size, w // kernel_size
    out = torch.empty(n, c, out_d, out_h, out_w, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (
            triton.cdiv(out_d, meta["BLOCK_SIZE_D"]),
            triton.cdiv(out_h * out_w, meta["BLOCK_SIZE_HW"]),
            n * c,
        )

    # Heuristics for block sizes
    BLOCK_SIZE_D = 4
    BLOCK_SIZE_HW = 16

    avg_pool_3d_kernel[grid](
        x_reshaped,
        out,
        c,
        d,
        h,
        w,
        kernel_size,
        out_d,
        out_h,
        out_w,
        d * h * w,
        h * w,
        w,
        out_d * out_h * out_w,
        out_h * out_w,
        out_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    return out.view(*batch_shape, c, out_d, out_h, out_w)


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernel for fused average pooling operations.
    ConvTranspose3d and BatchNorm3d are kept as-is since they are already optimized in PyTorch,
    but we fuse the two AvgPool3d operations into a single 4x4x4 pooling via Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        # Two 2x2x2 average pools = one 4x4x4 average pool
        x = triton_avg_pool_3d(x, kernel_size=4)
        return x
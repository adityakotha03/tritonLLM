import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def avg_pool_3d_kernel(
    x_ptr,
    y_ptr,
    B,
    C,
    D,
    H,
    W,
    Kd,
    Kh,
    Kw,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    od,
    oh,
    ow,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Program IDs correspond to output spatial locations
    pid_od = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    # Handle output dimensions
    if pid_od >= od or pid_oh >= oh or pid_ow >= ow:
        return

    # Calculate input start coordinates (top-left of kernel)
    d_start = pid_od * stride_d - pad_d
    h_start = pid_oh * stride_h - pad_h
    w_start = pid_ow * stride_w - pad_w

    # Initialize sum and count
    sum_val = tl.zeros([BLOCK_D, BLOCK_H, BLOCK_W], dtype=tl.float32)
    count = 0

    # Iterate over kernel
    for kd in range(Kd):
        for kh in range(Kh):
            for kw in range(Kw):
                d_offset = d_start + kd
                h_offset = h_start + kh
                w_offset = w_start + kw

                # Check bounds
                d_mask = (d_offset >= 0) & (d_offset < D)
                h_mask = (h_offset >= 0) & (h_offset < H)
                w_mask = (w_offset >= 0) & (w_offset < W)

                # Load data if in bounds
                mask = d_mask & h_mask & w_mask
                if tl.sum(mask) != 0:
                    # Compute offsets in input tensor
                    offset = (
                        tl.arange(0, BLOCK_D)[:, None, None] * C * D * H * W +
                        tl.arange(0, BLOCK_H)[None, :, None] * C * H * W +
                        tl.arange(0, BLOCK_W)[None, None, :] * C * W +
                        d_offset * C * H * W +
                        h_offset * C * W +
                        w_offset * C
                    )
                    data = tl.load(x_ptr + offset, mask=mask[None, :, :], other=0.0)
                    sum_val += tl.where(mask[None, :, :], data, 0.0)
                    count += 1

    # Compute average
    avg_val = sum_val / count

    # Store output
    output_offset = (
        pid_od * oh * ow * C +
        pid_oh * ow * C +
        pid_ow * C +
        tl.arange(0, C)
    )
    tl.store(y_ptr + output_offset, avg_val)


def triton_avg_pool_3d(
    x: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
):
    if isinstance(kernel_size, int):
        Kd = Kh = Kw = kernel_size
    else:
        Kd, Kh, Kw = kernel_size

    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding

    B, C, D, H, W = x.shape

    # Output dimensions
    od = (D + 2 * pad_d - Kd) // stride_d + 1
    oh = (H + 2 * pad_h - Kh) // stride_h + 1
    ow = (W + 2 * pad_w - Kw) // stride_w + 1

    # Output tensor
    y = torch.empty((B, C, od, oh, ow), dtype=x.dtype, device=x.device)

    # Flatten batch and channel dimensions for kernel processing
    x_flat = x.view(B * C, 1, D, H, W)
    y_flat = y.view(B * C, 1, od, oh, ow)

    # Block sizes
    BLOCK_D = min(4, od)
    BLOCK_H = min(8, oh)
    BLOCK_W = min(16, ow)

    # Grid
    grid = (triton.cdiv(od, BLOCK_D), triton.cdiv(oh, BLOCK_H), triton.cdiv(ow, BLOCK_W))

    # Launch kernel for each sample and channel
    for b in range(B * C):
        avg_pool_3d_kernel[grid](
            x_flat[b].data_ptr(),
            y_flat[b].data_ptr(),
            1, 1, D, H, W,
            Kd, Kh, Kw,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            od, oh, ow,
            BLOCK_D=BLOCK_D,
            BLOCK_H=BLOCK_H,
            BLOCK_W=BLOCK_W,
        )

    return y


class ModelNew(nn.Module):
    """
    Optimized version of 3D Average Pooling using Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool_3d(x, self.kernel_size, self.stride, self.padding)
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool2d_kernel(
    x_ptr,          # pointer to input tensor
    y_ptr,          # pointer to output tensor
    n_batch,        # batch size
    n_channels,     # number of channels
    in_height,      # input height
    in_width,       # input width
    out_height,     # output height
    out_width,      # output width
    kernel_size_h,  # kernel height
    kernel_size_w,  # kernel width
    stride_h,       # stride height
    stride_w,       # stride width
    padding_h,      # padding height
    padding_w,      # padding width
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # compute starting offsets for this block
    c_start = pid_c * BLOCK_SIZE_C
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # offsets within blocks
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)

    # masks
    c_mask = c_offsets < n_channels
    h_mask = h_offsets < out_height
    w_mask = w_offsets < out_width

    # broadcast masks for 3D
    mask = c_mask[:, None, None] & h_mask[None, :, None] & w_mask[None, None, :]

    # input pixel coordinates in output space
    out_h = h_offsets
    out_w = w_offsets

    # corresponding input top-left corner
    in_h_base = out_h * stride_h - padding_h
    in_w_base = out_w * stride_w - padding_w

    # initialize accumulator and counter
    sum_val = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    # iterate over kernel window
    for kh in range(kernel_size_h):
        for kw in range(kernel_size_w):
            in_h = in_h_base + kh
            in_w = in_w_base + kw

            # bounds check for input
            in_h_valid = (in_h >= 0) & (in_h < in_height)
            in_w_valid = (in_w >= 0) & (in_w < in_width)

            # combine validity
            valid = in_h_valid[None, :, None] & in_w_valid[None, None, :]

            # gather input indices
            in_idx = (
                pid_b * n_channels * in_height * in_width +
                c_offsets[:, None, None] * in_height * in_width +
                in_h[None, :, None] * in_width +
                in_w[None, None, :]
            )

            # load data (zero out-of-bounds)
            x = tl.load(x_ptr + in_idx, mask=valid & mask, other=0.0)

            # accumulate
            sum_val += x.to(tl.float32)

    # compute average
    area = kernel_size_h * kernel_size_w
    output = sum_val / area

    # store output
    out_idx = (
        pid_b * n_channels * out_height * out_width +
        c_offsets[:, None, None] * out_height * out_width +
        out_h[None, :, None] * out_width +
        out_w[None, None, :]
    )
    tl.store(y_ptr + out_idx, output, mask=mask)


class ModelNew(nn.Module):
    """
    Optimized version of 2D Average Pooling using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "Input tensor must be on CUDA."

        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # allocate output
        y = torch.empty((batch_size, channels, out_height, out_width), dtype=x.dtype, device=x.device)

        # block sizes
        BLOCK_SIZE_C = triton.next_power_of_2(channels) if channels <= 64 else 64
        BLOCK_SIZE_H = min(16, triton.cdiv(out_height, 4))
        BLOCK_SIZE_W = min(16, triton.cdiv(out_width, 4))

        # ensure block sizes are at least 1
        BLOCK_SIZE_H = max(BLOCK_SIZE_H, 1)
        BLOCK_SIZE_W = max(BLOCK_SIZE_W, 1)

        # grid
        grid = (
            batch_size,
            triton.cdiv(channels, BLOCK_SIZE_C),
            triton.cdiv(out_height, BLOCK_SIZE_H),
            triton.cdiv(out_width, BLOCK_SIZE_W),
        )

        # launch kernel
        avg_pool2d_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            n_batch=batch_size,
            n_channels=channels,
            in_height=in_height,
            in_width=in_width,
            out_height=out_height,
            out_width=out_width,
            kernel_size_h=self.kernel_size,
            kernel_size_w=self.kernel_size,
            stride_h=self.stride,
            stride_w=self.stride,
            padding_h=self.padding,
            padding_w=self.padding,
            BLOCK_SIZE_C=BLOCK_SIZE_C,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )

        return y
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,                # input tensor
    out_ptr,              # output tensor
    batch: tl.constexpr,
    channels: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    out_H: tl.constexpr,
    out_W: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # total number of output elements
    n_out = batch * channels * out_H * out_W

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    idxs = offset + tl.arange(0, BLOCK_SIZE)

    # mask for valid elements within the grid
    mask = idxs < n_out

    # compute coordinates of each output element
    b = (idxs // (channels * out_H * out_W)) % batch
    c = (idxs // (out_H * out_W)) % channels
    oh = (idxs // out_W) % out_H
    ow = idxs % out_W

    # base offset for each batch/channels slice in the input
    base = (b * channels + c) * H * W

    in_h_start = oh * stride - padding
    in_w_start = ow * stride - padding

    # initialise the maximum to negative infinity
    max_val = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)

    # iterate over the pooling window
    for i in range(kernel_size):
        h_offset = in_h_start + i * dilation
        h_cond = (h_offset >= 0) & (h_offset < H)
        for j in range(kernel_size):
            w_offset = in_w_start + j * dilation
            w_cond = (w_offset >= 0) & (w_offset < W)

            # mask for valid input positions
            mask_in = h_cond & w_cond

            # linear index into the flattened input tensor
            idx_in = base + h_offset * W + w_offset
            idx_in = tl.cast(idx_in, tl.int32)

            val = tl.load(x_ptr + idx_in, mask=mask_in, other=-float("inf"))
            max_val = tl.maximum(max_val, val)

    # store results
    out_offsets = offset + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_offsets, max_val, mask=mask)


def triton_max_pool2d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """
    Custom Triton implementation of 2D max pooling.
    """
    x = x.contiguous()
    batch, channels, H, W = x.shape

    out_H = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    out_W = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    out = torch.empty((batch, channels, out_H, out_W), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 128  # tunable

    grid = lambda meta: (
        (batch * channels * out_H * out_W + meta["BLOCK_SIZE"] - 1)
        // meta["BLOCK_SIZE"],
    )

    maxpool2d_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch=batch,
        channels=channels,
        H=H,
        W=W,
        out_H=out_H,
        out_W=out_W,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimised model that performs Max Pooling 2D using a custom Triton kernel.
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
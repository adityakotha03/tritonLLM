import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 32, "BLOCK_W": 32, "BLOCK_C": 8}, num_warps=4),
        triton.Config({"BLOCK_H": 64, "BLOCK_W": 32, "BLOCK_C": 4}, num_warps=4),
    ],
    key=["H", "W", "C"],
)
@triton.jit
def _min_sum_gelu_add_kernel(
    x_ptr,          # shape (N, C, H, W)
    bias_ptr,       # shape (1, 1, 1)
    out_ptr,        # shape (N, 1, 1, W)
    N, C, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program ids
    program_id_n = tl.program_id(0)
    program_id_h = tl.program_id(1)
    program_id_w = tl.program_id(2)

    # calculate block start indices
    h_start = program_id_h * BLOCK_H
    w_start = program_id_w * BLOCK_W

    # load bias (scalar broadcast)
    bias = tl.load(bias_ptr)

    # iterate over channel tiles
    c_start = 0
    while c_start < C:
        # load a tile of the input for this channel block
        offsets_c = tl.arange(0, BLOCK_C) + c_start
        offsets_h = tl.arange(0, BLOCK_H) + h_start
        offsets_w = tl.arange(0, BLOCK_W) + w_start

        mask_c = offsets_c < C
        mask_h = offsets_h < H
        mask_w = offsets_w < W
        mask = mask_c[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

        # stride: N, C, H, W
        stride_N = C * H * W
        stride_C = H * W
        stride_H = W
        stride_W = 1

        # compute flat index for each element in the tile
        idx = (
            program_id_n * stride_N
            + offsets_c[:, None, None] * stride_C
            + offsets_h[None, :, None] * stride_H
            + offsets_w[None, None, :]
        )
        # load values
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)

        # compute per-channel minima
        vals_min = tl.min(vals, axis=0)

        # sum over height dimension (axis=0 after min)
        sum_h = tl.sum(vals_min, axis=0)

        # accumulate across channel tiles
        if c_start == 0:
            acc = sum_h
        else:
            acc += sum_h

        c_start += BLOCK_C

    # GELU activation (approximation)
    sqrt2 = tl.constexpr(1.4142135623730951)
    inv_sqrt2 = tl.constexpr(0.7071067811865475)
    t = acc * inv_sqrt2
    erf = tl.erf(t)
    gelu = 0.5 * acc * (1.0 + erf)

    # add bias
    out = gelu + bias

    # store result: output shape (N, 1, 1, W)
    out_offset = program_id_n * W + w_start
    out_idx = out_offset
    out_mask = w_start + tl.arange(0, BLOCK_W) < W
    tl.store(out_ptr + out_idx, out, mask=out_mask)


def min_sum_gelu_add(x: torch.Tensor, bias: torch.Tensor):
    """
    x: shape (N, C, H, W)
    bias: shape (1, 1, 1) broadcastable
    returns: (N, 1, 1, W)
    """
    N, C, H, W = x.shape
    out = torch.empty((N, 1, 1, W), dtype=x.dtype, device=x.device)

    grid = lambda meta: (
        N,
        (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"],
        (W + meta["BLOCK_W"] - 1) // meta["BLOCK_W"],
    )

    _min_sum_gelu_add_kernel[grid](
        x, bias, out, N, C, H, W,
        BLOCK_H=32, BLOCK_W=32, BLOCK_C=8,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized model using Triton for min, sum, GELU and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape, device="cuda"))

    def forward(self, x):
        x = self.conv_transpose(x)
        # x shape: (N, C, H, W)
        x = min_sum_gelu_add(x, self.bias)
        return x
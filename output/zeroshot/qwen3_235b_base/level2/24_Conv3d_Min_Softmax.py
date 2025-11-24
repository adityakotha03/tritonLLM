import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def min_kernel(
    x_ptr,               # pointer to input tensor
    output_ptr,          # pointer to output tensor (min over dim D)
    D: tl.constexpr,     # depth dimension size
    H: tl.constexpr,     # height
    W: tl.constexpr,     # width
    C: tl.constexpr,     # channels (out_channels)
    BD: tl.constexpr,    # block size for D
    BC: tl.constexpr,    # block size for C
    BH: tl.constexpr,    # block size for H
    BW: tl.constexpr,    # block size for W
):
    # 2D block across (C, H, W), each block reduces over D
    pid_chw = tl.program_id(0)
    num_chw = C * H * W
    # Compute C, H, W indices from 1D pid
    c = (pid_chw // (H * W)) % C
    h = (pid_chw // W) % H
    w = pid_chw % W

    # Base offset for this (c, h, w) across D
    offset_chw = c * D * H * W + h * W + w
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(D, C, H, W),
        strides=(C * H * W, H * W, W, 1),
        offsets=(0, c, h, w),
        block_shape=(BD, 1, 1, 1),
        order=(0, 3, 2, 1)
    )
    # Load full D slice for this (c, h, w)
    mask = tl.arange(0, BD) < D
    x = tl.load(x_block_ptr, boundary_check=(0,), padding_option="zero")
    # Reduce along D
    min_val = tl.min(x, axis=0)
    # Store result
    tl.store(output_ptr + c * H * W + h * W + w, min_val)


@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    n_rows, row_width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < row_width
    x_row = tl.load(x_ptr + row_id * row_width + col_offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x_row, axis=0)
    x_exp = tl.exp(x_row - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_output = x_exp / x_sum
    tl.store(output_ptr + row_id * row_width + col_offsets, softmax_output, mask=mask)


def triton_min(x: torch.Tensor, dim: int):
    # Assume dim=2 corresponds to D in (B, C, D, H, W)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # -> (B, D, C, H, W)
    B, D, C, H, W = x.shape
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    # Each block handles one (C, H, W) element, reducing over D
    grid = (B * C * H * W,)
    # Use small block sizes, but Triton will coalesce
    min_kernel[grid](
        x, out,
        D=D, H=H, W=W, C=C,
        BD=triton.next_power_of_2(D),
        BC=1, BH=1, BW=1,
    )
    return out


def triton_softmax(x: torch.Tensor, dim: int):
    x = x.contiguous()
    B, C, H, W = x.shape
    x_reshaped = x.view(-1, C)
    n_rows, row_width = x_reshaped.shape
    out = torch.empty_like(x_reshaped)
    BLOCK_SIZE = triton.next_power_of_2(C)
    grid = (n_rows,)
    softmax_kernel[grid](x_reshaped, out, n_rows, row_width, BLOCK_SIZE=BLOCK_SIZE)
    return out.view(B, C, H, W)


class ModelNew(nn.Module):
    """
    Optimized version of Model using Triton kernels for min and softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = self.conv(x)
        x = triton_min(x, dim=self.dim)
        x = triton_softmax(x, dim=1)
        return x
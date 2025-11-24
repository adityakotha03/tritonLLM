import torch
import torch.nn as nn
import triton
import triton.language as tl

# -------------------------------------------------------------
# Triton kernel for a 2D convolution (no bias, int8 friendly)
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_W": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE_W": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_W": 512}, num_warps=8),
    ],
    key=["N", "C", "H", "W", "KH", "KW", "stride_h", "stride_w",
         "pad_h", "pad_w", "dil_h", "dil_w"],
)
@triton.jit
def conv2d_forward_kernel(
    X_ptr,          # [N, C, H, W]
    K_ptr,          # [M, C, KH, KW]
    Y_ptr,          # [N, M, OH, OW]
    N, C, H, W,     # input dimensions
    M, KH, KW,      # output channel, kernel size
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    OH, OW,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Each program handles a contiguous block of output width positions
    for one (batch, output_channel, output_height) slice.
    """
    # program id corresponds to linear index over (N, M, OH, OW)
    prog_id = tl.program_id(0)

    # compute indices
    ow_start = (prog_id % OW) * BLOCK_SIZE_W
    out_w_idx = ow_start + tl.arange(0, BLOCK_SIZE_W)
    ow_mask = out_w_idx < OW

    # compute linear index of (n, m, oh)
    idx = prog_id // OW
    oh = idx % OH
    idx = idx // OH
    n = idx % N
    m = idx // N

    # base pointers
    x_offset = n * C * H * W
    k_offset = m * C * KH * KW
    y_offset = n * M * OH * OW + m * OH * OW + oh * OW

    # iterate over kernel spatial dimensions and input channels
    acc = tl.zeros((BLOCK_SIZE_W,), dtype=tl.float32)

    for kc in range(C):
        for kh in range(KH):
            for kw in range(KW):
                # compute input spatial coordinates
                ih = oh * stride_h + kh * dil_h - pad_h
                iw = out_w_idx * stride_w + kw * dil_w - pad_w

                # load kernel weight
                k_val = tl.load(K_ptr + k_offset + kc * KH * KW + kh * KW + kw)

                # load input values with mask
                inp = tl.load(
                    X_ptr + x_offset + kc * H * W + ih * W + iw,
                    mask=ih >= 0,
                    other=0.0,
                )
                inp = tl.where(iw >= 0, inp, 0.0)

                acc += inp * k_val

    # store results
    tl.store(
        Y_ptr + y_offset + out_w_idx,
        acc,
        mask=ow_mask,
    )


# -------------------------------------------------------------
# Wrapper functions
# -------------------------------------------------------------
def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int,
    padding: tuple[int, int],
    dilation: tuple[int, int],
    out_height: int,
    out_width: int,
) -> torch.Tensor:
    """
    Performs a 2D convolution using the Triton kernel.
    """
    N, C, H, W = x.shape
    M, _, KH, KW = weight.shape
    stride_h, stride_w = stride, stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # output tensor
    y = torch.empty((N, M, out_height, out_width), dtype=x.dtype, device=x.device)

    # grid: one program per (n, m, oh, ow) slice
    num_progs = N * M * out_height * out_width
    grid = lambda meta: (num_progs,)

    conv2d_forward_kernel[grid](
        x,
        weight,
        y,
        N, C, H, W,
        M, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        out_height, out_width,
        BLOCK_SIZE_W=256,
    )
    return y


# -------------------------------------------------------------
# Optimized Model using Triton convolution
# -------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel_size, dtype=torch.float32)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, h, w = x.shape
        KH, KW = self.kernel_size
        stride = self.stride
        pad_h, pad_w = self.padding
        dil_h, dil_w = self.dilation

        # compute output spatial dims
        oh = (h + 2 * pad_h - dil_h * (KH - 1) - 1) // stride + 1
        ow = (w + 2 * pad_w - dil_w * (KW - 1) - 1) // stride + 1

        out = triton_conv2d(
            x,
            self.weight,
            stride,
            (pad_h, pad_w),
            (dil_h, dil_w),
            oh,
            ow,
        )

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
    ],
    key=["B", "OC", "OH", "OW"],
)
@triton.jit
def conv2d_kernel(
    inp_ptr,          # pointer to input tensor
    weight_ptr,       # pointer to weight tensor
    bias_ptr,         # pointer to bias tensor (can be None)
    out_ptr,          # pointer to output tensor
    B: tl.constexpr,   # batch size
    IC: tl.constexpr,  # input channels
    IH: tl.constexpr,  # input height
    IW: tl.constexpr,  # input width
    OC: tl.constexpr,  # output channels
    OH: tl.constexpr,  # output height
    OW: tl.constexpr,  # output width
    KH: tl.constexpr,  # kernel height
    KW: tl.constexpr,  # kernel width
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of output elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Convert linear offset to (b, oc, out_row, out_col)
    b_offset = offsets // (OC * OH * OW)
    oc_offset = (offsets % (OC * OH * OW)) // (OH * OW)
    out_row = (offsets % (OH * OW)) // OW
    out_col = offsets % OW

    mask = (b_offset < B) & (oc_offset < OC) & (out_row < OH) & (out_col < OW)

    # Compute the value for each output element
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Iterate over input channels and kernel spatial dimensions
    for ic in range(IC):
        for kh in range(KH):
            in_row = out_row * stride_h + kh * dilation_h - pad_h
            row_mask = (in_row >= 0) & (in_row < IH)

            for kw in range(KW):
                in_col = out_col * stride_w + kw * dilation_w - pad_w
                col_mask = (in_col >= 0) & (in_col < IW)

                valid = row_mask & col_mask

                # Compute flattened input index
                inp_offset = (
                    b_offset * IC * IH * IW
                    + ic * IH * IW
                    + in_row * IW
                    + in_col
                )
                inp = tl.load(inp_ptr + inp_offset, mask=valid, other=0.0)

                weight_offset = (
                    oc_offset * IC * KH * KW
                    + ic * KH * KW
                    + kh * KW
                    + kw
                )
                w = tl.load(weight_ptr + weight_offset, mask=tl.full([BLOCK_SIZE], True), other=0.0)

                acc += inp * w

    # Add bias if provided
    if bias_ptr is not None:
        bias_offset = oc_offset
        bias = tl.load(bias_ptr + bias_offset, mask=tl.full([BLOCK_SIZE], True), other=0.0)
        acc += bias

    out_offset = (
        b_offset * OC * OH * OW
        + oc_offset * OH * OW
        + out_row * OW
        + out_col
    )
    tl.store(out_ptr + out_offset, acc, mask=mask)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int = 1,
):
    """
    Triton implementation of a 2D convolution.
    Only groups == 1 is supported.
    """
    assert groups == 1, "Only groups=1 is supported in this Triton kernel."
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    # Compute output dimensions
    OH = (IH + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    OW = (IW + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

    out = torch.empty((B, OC, OH, OW), dtype=x.dtype, device=x.device)

    grid = lambda meta: ((OH * OW * B * OC + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    conv2d_kernel[grid](
        x,
        weight,
        bias,
        out,
        B,
        IC,
        IH,
        IW,
        OC,
        OH,
        OW,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        BLOCK_SIZE=256,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton‑accelerated 2D convolution with an asymmetric kernel.
    Supports stride, padding, dilation, bias.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, *kernel_size, device="cuda")
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device="cuda"))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
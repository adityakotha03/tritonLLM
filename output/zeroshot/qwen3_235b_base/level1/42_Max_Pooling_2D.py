import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool2d_kernel(
    x_ptr,          # pointer to input tensor
    y_ptr,          # pointer to output tensor
    N: tl.constexpr, # batch size
    C: tl.constexpr, # number of channels
    H: tl.constexpr, # input height
    W: tl.constexpr, # input width
    OH: tl.constexpr, # output height
    OW: tl.constexpr, # output width
    KH: tl.constexpr, # kernel height
    KW: tl.constexpr, # kernel width
    SH: tl.constexpr, # stride height
    SW: tl.constexpr, # stride width
    PH: tl.constexpr, # padding height
    PW: tl.constexpr, # padding width
    DH: tl.constexpr, # dilation height
    DW: tl.constexpr, # dilation width
    stride_xn: tl.constexpr, # stride for batch
    stride_xc: tl.constexpr, # stride for channel
    stride_xh: tl.constexpr, # stride for input height
    stride_xw: tl.constexpr, # stride for input width
    stride_yn: tl.constexpr, # stride for output batch
    stride_yc: tl.constexpr, # stride for output channel
    stride_yh: tl.constexpr, # stride for output height
    stride_yw: tl.constexpr, # stride for output width
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_oh = tl.program_id(2)
    pid_ow = tl.program_id(3)

    # offsets for output patch
    oh_start = pid_oh * BLOCK_SIZE_H
    ow_start = pid_ow * BLOCK_SIZE_W

    # compute output block size
    oh_end = tl.minimum(oh_start + BLOCK_SIZE_H, OH)
    ow_end = tl.minimum(ow_start + BLOCK_SIZE_W, OW)

    # initialize offsets for output
    offsets_oh = oh_start + tl.arange(0, BLOCK_SIZE_H)
    offsets_ow = ow_start + tl.arange(0, BLOCK_SIZE_W)

    # mask for valid output pixels
    mask_oh = offsets_oh < OH
    mask_ow = offsets_ow < OW
    mask_out = mask_oh[:, None] & mask_ow[None, :]

    # calculate input pixel locations
    ih_base = offsets_oh * SH - PH
    iw_base = offsets_ow * SW - PW

    # initialize max values
    max_val = tl.full([BLOCK_SIZE_H, BLOCK_SIZE_W], value=float("-inf"), dtype=tl.float32)

    # iterate over kernel window
    for kh in range(KH):
        for kw in range(KW):
            ih = ih_base + kh * DH
            iw = iw_base + kw * DW

            # mask for valid input pixels
            mask_h = (ih >= 0) & (ih < H)
            mask_w = (iw >= 0) & (iw < W)
            mask_valid = mask_h[:, None] & mask_w[None, :] & mask_out

            # gather input indices
            input_offsets = (
                pid_n * stride_xn +
                pid_c * stride_xc +
                ih[:, None] * stride_xh +
                iw[None, :] * stride_xw
            )
            x = tl.load(x_ptr + input_offsets, mask=mask_valid, other=float("-inf"))

            # update max
            max_val = tl.maximum(max_val, x)

    # store output
    output_offsets = (
        pid_n * stride_yn +
        pid_c * stride_yc +
        offsets_oh[:, None] * stride_yh +
        offsets_ow[None, :] * stride_yw
    )
    tl.store(y_ptr + output_offsets, max_val, mask=mask_out)


class ModelNew(nn.Module):
    """
    Optimized version of MaxPool2d using a custom Triton kernel.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype
        x = x.contiguous()

        N, C, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride
        PH, PW = self.padding, self.padding
        DH, DW = self.dilation, self.dilation

        # compute output spatial dimensions
        OH = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1
        OW = (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1

        # output tensor
        y = torch.empty((N, C, OH, OW), dtype=dtype, device=device)

        # handle edge case: empty output
        if OH == 0 or OW == 0:
            return y

        # compute strides
        stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
        stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

        # set block sizes
        BLOCK_SIZE_H = min(32, triton.next_power_of_2(OH))
        BLOCK_SIZE_W = min(32, triton.next_power_of_2(OW))

        # grid configuration: one block per (N, C, output tile)
        grid = (N, C, triton.cdiv(OH, BLOCK_SIZE_H), triton.cdiv(OW, BLOCK_SIZE_W))

        # launch kernel
        maxpool2d_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            N=N, C=C, H=H, W=W,
            OH=OH, OW=OW,
            KH=KH, KW=KW,
            SH=SH, SW=SW,
            PH=PH, PW=PW,
            DH=DH, DW=DW,
            stride_xn=stride_xn, stride_xc=stride_xc, stride_xh=stride_xh, stride_xw=stride_xw,
            stride_yn=stride_yn, stride_yc=stride_yc, stride_yh=stride_yh, stride_yw=stride_yw,
            BLOCK_SIZE_H=BLOCK_SIZE_H,
            BLOCK_SIZE_W=BLOCK_SIZE_W,
        )

        return y
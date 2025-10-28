import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OC": 32, "BLOCK_P": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OC": 64, "BLOCK_P": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OC": 64, "BLOCK_P": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OC": 32, "BLOCK_P": 128}, num_warps=8, num_stages=2),
    ],
    key=["OC", "OH", "OW"],
)
@triton.jit
def conv2d_nchw_fwd_kernel(
    x_ptr,           # *f32 [N, C, H, W]
    w_ptr,           # *f32 [IC, KH, KW, OC] (packed)
    b_ptr,           # *f32 [OC] or nullptr if no bias
    out_ptr,         # *f32 [N, OC, OH, OW]
    N: tl.constexpr,
    IC: tl.constexpr,  # input channels (constexpr for unrolling)
    H, W,
    OC,              # output channels
    KH: tl.constexpr,
    KW: tl.constexpr,
    OH, OW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    P,               # total positions = OH * OW
    BLOCK_OC: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    oc_mask = oc_offsets < OC
    p_mask = p_offsets < P

    # Derive (oh, ow) for each position
    OW_i = OW
    oh = tl.where(p_mask, p_offsets // OW_i, 0)
    ow = tl.where(p_mask, p_offsets % OW_i, 0)

    ih_base = oh * STRIDE_H - PAD_H
    iw_base = ow * STRIDE_W - PAD_W

    # Accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_P), dtype=tl.float32)

    # Loop over input channels and kernel spatial dims (unrolled by Triton due to constexpr)
    for ic in range(IC):
        for kh in range(KH):
            ih = ih_base + kh
            in_h_mask = (ih >= 0) & (ih < H)
            for kw in range(KW):
                iw = iw_base + kw
                in_w_mask = (iw >= 0) & (iw < W)
                in_bounds = p_mask & in_h_mask & in_w_mask

                # Compute input pointer offsets for the P-tile; broadcasting over P
                # index = ((n*C + ic)*H + ih) * W + iw
                nci = (pid_n * IC + ic)
                in_index = (nci * H + ih) * W + iw
                x_vals = tl.load(x_ptr + in_index, mask=in_bounds, other=0.0)
                x_vals = x_vals.to(tl.float32)

                # Compute weight pointer offsets for the OC tile; broadcasting over OC
                # w_ptr layout: [IC, KH, KW, OC] contiguous
                w_index = (((ic * KH + kh) * KW + kw) * OC) + oc_offsets
                w_vals = tl.load(w_ptr + w_index, mask=oc_mask, other=0.0)
                w_vals = w_vals.to(tl.float32)

                # Outer product accumulate
                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias if provided
    if tl.constexpr(b_ptr is not None):
        b_vals = tl.load(b_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
        acc += b_vals[:, None]

    # Store results to output
    # out index = ((n*OC + oc)*OH + oh) * OW + ow
    out_index = ((pid_n * OC + oc_offsets[:, None]) * OH + oh[None, :]) * OW + ow[None, :]
    out_mask = oc_mask[:, None] & p_mask[None, :]
    tl.store(out_ptr + out_index, acc, mask=out_mask)


def conv2d_triton_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                       stride=(1, 1), padding=(0, 0)):
    """
    Triton-based Conv2D forward for a single group, NCHW layout, dilation=1.
    weight expected shape: [OC, IC, KH, KW]
    x expected shape: [N, IC, H, W]
    """
    if not x.is_cuda:
        # CPU fallback
        return F.conv2d(x, weight, bias, stride=stride, padding=padding)

    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32) or True, "Unsupported dtype"
    device = x.device
    dtype = x.dtype

    N, IC, H, W = x.shape
    OC, IC_w, KH, KW = weight.shape
    assert IC == IC_w, "Input channels mismatch"

    stride_h, stride_w = stride
    pad_h, pad_w = padding

    # Compute output dims
    OH = (H + 2 * pad_h - (KH)) // stride_h + 1
    OW = (W + 2 * pad_w - (KW)) // stride_w + 1
    P = OH * OW

    # Ensure contiguous
    x_contig = x.contiguous()
    # Repack weights for better memory access: [IC, KH, KW, OC]
    w_packed = weight.permute(1, 2, 3, 0).contiguous()
    # Cast to fp32 in kernel for accumulation; but we can keep inputs as original dtype and convert inside kernel.
    out = torch.empty((N, OC, OH, OW), device=device, dtype=torch.float32)

    # Bias handling: pass pointer or None
    if bias is not None:
        b = bias.contiguous()
        b_ptr = b
    else:
        b_ptr = None

    # Launch kernel
    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta["BLOCK_OC"]),
        triton.cdiv(P, meta["BLOCK_P"]),
    )
    conv2d_nchw_fwd_kernel[grid](
        x_contig,
        w_packed,
        b_ptr,
        out,
        N,
        IC,  # constexpr
        H, W,
        OC,
        KH,  # constexpr
        KW,  # constexpr
        OH, OW,
        stride_h, stride_w,
        pad_h, pad_w,
        P,
    )
    # Cast back to input dtype
    return out.to(dtype)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        # Keep original Conv2d module for parameter management and initialization,
        # but we'll replace its forward with our Triton kernel.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)

    def forward(self, x):
        # Use our Triton conv2d implementation
        stride = self.conv1.stride
        padding = self.conv1.padding
        return conv2d_triton_nchw(x, self.conv1.weight, self.conv1.bias, stride=stride, padding=padding)
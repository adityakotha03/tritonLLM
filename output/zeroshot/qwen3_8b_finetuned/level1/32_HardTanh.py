import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardtanh_kernel(
    in_ptr0,  # Pointer to input tensor
    out_ptr0,  # Pointer to output tensor
    xnumel,  # Total number of elements
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = -1.0
    tmp2 = tmp0 < tmp1
    tmp3 = 1.0
    tmp4 = tmp0 > tmp3
    tmp5 = tl.where(tmp2, tmp1, tmp0)
    tmp6 = tl.where(tmp4, tmp3, tmp5)
    tl.store(out_ptr0 + x0, tmp6, xmask)


def triton_poi_fused_hardtanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = -1.0
    tmp2 = tmp0 < tmp1
    tmp3 = 1.0
    tmp4 = tmp0 > tmp3
    tmp5 = tl.where(tmp2, tmp1, tmp0)
    tmp6 = tl.where(tmp4, tmp3, tmp5)
    tl.store(in_out_ptr0 + x0, tmp6, xmask)


class ModelNew(nn.Module):
    """
    Simple model that performs a HardTanh activation using a custom Triton kernel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0):
        arg0_1 = input_0
        arg0_2 = input_0
        output = triton_poi_fused_hardtanh_0(arg0_1, arg0_2, 1610612160, XBLOCK=16384)
        return output
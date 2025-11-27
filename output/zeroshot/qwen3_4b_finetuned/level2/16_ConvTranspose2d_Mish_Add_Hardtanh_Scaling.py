import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_hardtanh_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_18(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_20(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_21(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_22(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_24(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_25(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_27(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_28(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_29(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_30(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_31(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_32(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_33(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_34(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_35(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 > tmp1
    tmp3 = -1.0
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 | tmp4
    tmp6 = 0.0
    tmp7 = tl.where(tmp5, tmp6, tmp0)
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_36(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnum
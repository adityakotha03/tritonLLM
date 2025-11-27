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
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_1(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_2(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_3(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_4(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_5(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_6(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_7(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 96
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_8(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 96
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_9(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 160
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_10(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 160
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_11(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_12(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 320
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32 % 1280
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_14(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_15(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_16(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_17(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_18(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_19(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_20(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_21(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_22(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_23(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_24(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_25(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_26(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_27(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_28(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_29(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_30(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1280 % 1280
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp2 < 0
    tmp5 = tl.where(tmp4, tmp2, tmp3)
    tmp6 = tmp2 > 0
    tmp7 = tl.where(tmp6, tmp2, 0)
    tmp8 = tl.where(tmp4, tmp7, tmp5)
    tl.store(in_out_ptr0 + x2, tmp8, xmask)


@
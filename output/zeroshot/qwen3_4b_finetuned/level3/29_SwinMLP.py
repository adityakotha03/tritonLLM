import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = xindex // 14 % 14
    x2 = xindex // 196
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (24 * x0 + 896 * x2), xmask)
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 15
    x1 = xindex // 15
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tl.store(out_ptr0 + (x0 + 15 * x1), tmp0, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_gelu_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.erf(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_out_ptr0 + x2, None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, None)


@triton.jit
def triton_poi_fused_clone_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 28
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)


@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 15
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, xmask)


@triton.jit
def triton_poi_fused_add_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, None)


@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, None)


@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)


@triton.jit
def triton_poi_fused_gelu_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.erf(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)


@triton.jit
def triton_poi_fused_convolution_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)


@triton.jit
def triton_poi_fused_gelu_17(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.erf(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, None)


@triton.jit
def triton_poi_fused_clone_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_20(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)


@triton.jit
def triton_poi_fused_gelu_21(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.erf(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_clone_23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x2, None)
    tmp2 = tl.load(in_ptr2 + x0, None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + x0, None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + x0, None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + x0, None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + x0, None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + x0, None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + x0, None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + x0, None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr12 + x0, None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr13 + x0, None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + x0, None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 + tmp15
    tmp10 = tmp8 - tmp9
    tmp11 = 3.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 - tmp12
    tmp14 = 0.5
    tmp16 = tmp13 * tmp14
    tmp18 = tmp13 + tmp14
    tmp19 = tmp16 * tmp18
    tmp21 = tmp19 * tmp18
    tmp22 = 0.25
    tmp24 = tmp21 * tmp22
    tmp25 = tmp10 * tmp24
    tmp26 = tmp19 + tmp25
    tmp28 = tmp26 * tmp28
    tmp29 = tmp22 * tmp28
    tmp31 = tmp26 + tmp29
    tmp32 = tmp28 + tmp31
    tmp33 = tmp26 - tmp32
    tmp35 = tmp31 + tmp34
    tmp36 = tmp33 - tmp35
    tmp38 = tmp36 * tmp11
    tmp39 = tmp38 * tmp11
    tmp41 = tmp39 + tmp40
    tmp42 = tmp39 + tmp43
    tmp44 = tmp42 - tmp41
    tmp45 = tmp39 + tmp44
    tmp47 = tmp47 * tmp11
    tmp48 = tmp47 + tmp46
    tmp49 = 1.1723928423207855
    tmp50 = tmp48 * tmp49
    tmp51 = 0.873526306430798
    tmp52 = tmp50 * tmp51
    tl.store(out_ptr0 + x2, tmp52, None)


@triton.jit
def triton_poi_fused_convolution_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + x0, xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + x0, xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + x0, xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + x0, xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr12 + x0, xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr13 + x0, xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 + tmp15
    tmp10 = tmp8 - tmp9
    tmp11 = 3.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 - tmp12
    tmp14 = 0.5
    tmp16 = tmp13 * tmp14
    tmp18 = tmp13 + tmp14
    tmp19 = tmp16 * tmp18
    tmp21 = tmp19 * tmp18
    tmp22 = 0.25
    tmp24 = tmp21 * tmp22
    tmp25 = tmp10 * tmp24
    tmp26 = tmp19 + tmp25
    tmp28 = tmp26 * tmp28
    tmp29 = tmp22 * tmp28
    tmp31 = tmp26 + tmp29
    tmp32 = tmp28 + tmp31
    tmp33 = tmp26 - tmp32
    tmp35 = tmp31 + tmp34
    tmp36 = tmp33 - tmp35
    tmp38 = tmp36 * tmp11
    tmp39 = tmp38 * tmp11
    tmp41 = tmp39 + tmp40
    tmp42 = tmp39 + tmp43
    tmp44 = tmp42 - tmp41
    tmp45 = tmp39 + tmp44
    tmp47 = tmp47 * tmp11
    tmp48 = tmp47 + tmp46
    tl.store(out_ptr0 + x2, tmp48, xmask)


@triton.jit
def triton_poi_fused_clone_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + x0, xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + x0, xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + x0, xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + x0, xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr12 + x0, xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr13 + x0, xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 + tmp15
    tmp10 = tmp8 - tmp9
    tmp11 = 3.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 - tmp12
    tmp14 = 0.5
    tmp16 = tmp13 * tmp14
    tmp18 = tmp13 + tmp14
    tmp19 = tmp16 * tmp18
    tmp21 = tmp19 * tmp18
    tmp22 = 0.25
    tmp24 = tmp21 * tmp22
    tmp25 = tmp10 * tmp24
    tmp26 = tmp19 + tmp25
    tmp28 = tmp26 * tmp28
    tmp29 = tmp22 * tmp28
    tmp31 = tmp26 + tmp29
    tmp32 = tmp28 + tmp31
    tmp33 = tmp26 - tmp32
    tmp35 = tmp31 + tmp34
    tmp36 = tmp33 - tmp35
    tmp38 = tmp36 * tmp11
    tmp39 = tmp38 * tmp11
    tmp41 = tmp39 + tmp40
    tmp42 = tmp39 + tmp43
    tmp44 = tmp42 - tmp41
    tmp45 = tmp39 + tmp44
    tmp47 = tmp47 * tmp11
    tmp48 = tmp47 + tmp46
    tl.store(out_ptr0 + x2, tmp48, xmask)


@triton.jit
def triton_poi_fused_convolution_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + x0, xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + x0, xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + x0, xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr10 + x0, xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr11 + x0, xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr12 + x0, xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr13 + x0, xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr14 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 + tmp15
    tmp10 = tmp8 - tmp9
    tmp11 = 3.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 - tmp12
    tmp14 = 0.5
    tmp16 = tmp13 * tmp14
    tmp18 = tmp13 + tmp14
    tmp19 = tmp16 * tmp18
    tmp21 = tmp19 * tmp18
    tmp22 = 0.25
    tmp24 = tmp21 * tmp22
    tmp25 = tmp10 * tmp24
    tmp26 = tmp19 + tmp25
    tmp28 = tmp26 * tmp28
    tmp29 = tmp22 * tmp28
    tmp31 = tmp26 + tmp29
    tmp32 = tmp28 + tmp31
    tmp33 = tmp26 - tmp32
    tmp35 = tmp31 + tmp34
    tmp36 = tmp33 - tmp35
    tmp38 = tmp36 * tmp11
    tmp39 = tmp38 * tmp11
    tmp41 = tmp39 + tmp40
    tmp42 = tmp39 + tmp43
    tmp44 = tmp42 - tmp41
    tmp45 = tmp39 + tmp44
    tmp47 = tmp47 * tmp11
    tmp48 = tmp47 + tmp46
    tl.store(out_ptr0 + x2, tmp48, xmask)


@triton.jit
def triton_poi_fused_convolution_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
   
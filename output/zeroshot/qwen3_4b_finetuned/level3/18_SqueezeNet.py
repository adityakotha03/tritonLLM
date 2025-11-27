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
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64 * x1 + 64 * x0 + 16384 * x2), tmp4 & xmask,
        other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full(tmp7.shape, 1.0, tmp7.dtype)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp13 = tl.load(in_ptr1 + (64 * x1 + 64 * (-64 + x0) + 16384 * x2),
        tmp10 & xmask, other=0.0)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 1.0, tmp15.dtype)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.where(tmp4, tmp9, tmp17)
    tl.store(out_ptr0 + x3, tmp18, xmask)


@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp11 = tl.load(in_ptr1 + x1, tmp8 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 1.0, tmp13.dtype)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp17 = tmp0 < 128
    tmp18 = tl.full([1], 192, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr2 + x1, tmp20 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 1.0, tmp23.dtype)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(tmp20, tmp25, tmp16)
    tmp27 = tmp0 >= 192
    tmp28 = tl.full([1], 256, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr3 + x1, tmp30 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 1.0, tmp33.dtype)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.where(tmp30, tmp35, tmp26)
    tmp37 = tmp0 >= 256
    tl.full([1], 384, tl.int64)
    tmp40 = tl.load(in_ptr4 + x1, tmp37 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 1.0, tmp42.dtype)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.where(tmp30, tmp36, tmp44)
    tmp46 = tmp0 >= 384
    tl.full([1], 512, tl.int64)
    tmp49 = tl.load(in_ptr5 + x1, tmp46 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tl.full(tmp51.shape, 1.0, tmp51.dtype)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.where(tmp30, tmp45, tmp53)
    tmp55 = tmp0 >= 512
    tl.full([1], 512, tl.int64)
    tmp58 = tl.load(in_ptr6 + x1, tmp55 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp55, tmp58, tmp59)
    tmp61 = tl.full(tmp60.shape, 1.0, tmp60.dtype)
    tmp62 = tmp60 + tmp61
    tmp63 = tl.where(tmp30, tmp54, tmp62)
    tmp64 = tmp0 >= 768
    tl.full([1], 768, tl.int64)
    tmp67 = tl.load(in_ptr7 + x1, tmp64 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp64, tmp67, tmp68)
    tmp70 = tl.full(tmp69.shape, 1.0, tmp69.dtype)
    tmp71 = tmp69 + tmp70
    tmp72 = tl.where(tmp30, tmp63, tmp71)
    tmp73 = tmp0 >= 1024
    tl.full([1], 1024, tl.int64)
    tmp76 = tl.load(in_ptr8 + x1, tmp73 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp73, tmp76, tmp77)
    tmp79 = tl.full(tmp78.shape, 1.0, tmp78.dtype)
    tmp80 = tmp78 + tmp79
    tmp81 = tl.where(tmp30, tmp72, tmp80)
    tmp82 = tmp0 >= 1280
    tl.full([1], 1280, tl.int64)
    tmp85 = tl.load(in_ptr9 + x1, tmp82 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp82, tmp85, tmp86)
    tmp88 = tl.full(tmp87.shape, 1.0, tmp87.dtype)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.where(tmp30, tmp81, tmp89)
    tmp91 = tmp0 >= 1536
    tl.full([1], 1536, tl.int64)
    tmp94 = tl.load(in_ptr10 + x1, tmp91 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp91, tmp94, tmp95)
    tmp97 = tl.full(tmp96.shape, 1.0, tmp96.dtype)
    tmp98 = tmp96 + tmp97
    tmp99 = tl.where(tmp30, tmp90, tmp98)
    tmp100 = tmp0 >= 1792
    tl.full([1], 1792, tl.int64)
    tmp103 = tl.load(in_ptr11 + x1, tmp100 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp104 = tl.full(tmp103.shape, 0.0, tmp103.dtype)
    tmp105 = tl.where(tmp100, tmp103, tmp104)
    tmp106 = tl.full(tmp105.shape, 1.0, tmp105.dtype)
    tmp107 = tmp105 + tmp106
    tmp108 = tl.where(tmp30, tmp99, tmp107)
    tmp109 = tmp0 >= 2048
    tl.full([1], 2048, tl.int64)
    tmp112 = tl.load(in_ptr12 + x1, tmp109 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp113 = tl.full(tmp112.shape, 0.0, tmp112.dtype)
    tmp114 = tl.where(tmp109, tmp112, tmp113)
    tmp115 = tl.full(tmp114.shape, 1.0, tmp114.dtype)
    tmp116 = tmp114 + tmp115
    tmp117 = tl.where(tmp30, tmp108, tmp116)
    tmp118 = tmp0 >= 2304
    tl.full([1], 2304, tl.int64)
    tmp121 = tl.load(in_ptr13 + x1, tmp118 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp118, tmp121, tmp122)
    tmp124 = tl.full(tmp123.shape, 1.0, tmp123.dtype)
    tmp125 = tmp123 + tmp124
    tmp126 = tl.where(tmp30, tmp117, tmp125)
    tmp127 = tmp0 >= 2560
    tl.full([1], 2560, tl.int64)
    tmp130 = tl.load(in_ptr14 + x1, tmp127 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp131 = tl.full(tmp130.shape, 0.0, tmp130.dtype)
    tmp132 = tl.where(tmp127, tmp130, tmp131)
    tmp133 = tl.full(tmp132.shape, 1.0, tmp132.dtype)
    tmp134 = tmp132 + tmp133
    tmp135 = tl.where(tmp30, tmp126, tmp134)
    tmp136 = tmp0 >= 2816
    tl.full([1], 2816, tl.int64)
    tmp139 = tl.load(in_ptr15 + x1, tmp136 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tl.full(tmp141.shape, 1.0, tmp141.dtype)
    tmp143 = tmp141 + tmp142
    tmp144 = tl.where(tmp30, tmp135, tmp143)
    tmp145 = tmp0 >= 3024
    tl.full([1], 3024, tl.int64)
    tmp148 = tl.load(in_ptr16 + x1, tmp145 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp149 = tl.full(tmp148.shape, 0.0, tmp148.dtype)
    tmp150 = tl.where(tmp145, tmp148, tmp149)
    tmp151 = tl.full(tmp150.shape, 1.0, tmp150.dtype)
    tmp152 = tmp150 + tmp151
    tmp153 = tl.where(tmp30, tmp144, tmp152)
    tmp154 = tmp0 >= 3232
    tl.full([1], 3232, tl.int64)
    tmp157 = tl.load(in_ptr17 + x1, tmp154 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp158 = tl.full(tmp157.shape, 0.0, tmp157.dtype)
    tmp159 = tl.where(tmp154, tmp157, tmp158)
    tmp160 = tl.full(tmp159.shape, 1.0, tmp159.dtype)
    tmp161 = tmp159 + tmp160
    tmp162 = tl.where(tmp30, tmp153, tmp161)
    tl.store(out_ptr0 + x2, tmp162, xmask)


@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp11 = tl.load(in_ptr1 + x1, tmp8 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 1.0, tmp13.dtype)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp17 = tmp0 < 128
    tmp18 = tl.full([1], 192, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr2 + x1, tmp20 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 1.0, tmp23.dtype)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(tmp20, tmp25, tmp16)
    tmp27 = tmp0 >= 192
    tmp28 = tl.full([1], 256, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr3 + x1, tmp30 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 1.0, tmp33.dtype)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.where(tmp30, tmp35, tmp26)
    tmp37 = tmp0 >= 256
    tl.full([1], 384, tl.int64)
    tmp40 = tl.load(in_ptr4 + x1, tmp37 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 1.0, tmp42.dtype)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.where(tmp30, tmp36, tmp44)
    tmp46 = tmp0 >= 384
    tl.full([1], 512, tl.int64)
    tmp49 = tl.load(in_ptr5 + x1, tmp46 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tl.full(tmp51.shape, 1.0, tmp51.dtype)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.where(tmp30, tmp45, tmp53)
    tmp55 = tmp0 >= 512
    tl.full([1], 512, tl.int64)
    tmp58 = tl.load(in_ptr6 + x1, tmp55 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp55, tmp58, tmp59)
    tmp61 = tl.full(tmp60.shape, 1.0, tmp60.dtype)
    tmp62 = tmp60 + tmp61
    tmp63 = tl.where(tmp30, tmp54, tmp62)
    tmp64 = tmp0 >= 768
    tl.full([1], 768, tl.int64)
    tmp67 = tl.load(in_ptr7 + x1, tmp64 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp64, tmp67, tmp68)
    tmp70 = tl.full(tmp69.shape, 1.0, tmp69.dtype)
    tmp71 = tmp69 + tmp70
    tmp72 = tl.where(tmp30, tmp63, tmp71)
    tmp73 = tmp0 >= 1024
    tl.full([1], 1024, tl.int64)
    tmp76 = tl.load(in_ptr8 + x1, tmp73 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp73, tmp76, tmp77)
    tmp79 = tl.full(tmp78.shape, 1.0, tmp78.dtype)
    tmp80 = tmp78 + tmp79
    tmp81 = tl.where(tmp30, tmp72, tmp80)
    tmp82 = tmp0 >= 1280
    tl.full([1], 1280, tl.int64)
    tmp85 = tl.load(in_ptr9 + x1, tmp82 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp82, tmp85, tmp86)
    tmp88 = tl.full(tmp87.shape, 1.0, tmp87.dtype)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.where(tmp30, tmp81, tmp89)
    tmp91 = tmp0 >= 1536
    tl.full([1], 1536, tl.int64)
    tmp94 = tl.load(in_ptr10 + x1, tmp91 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp91, tmp94, tmp95)
    tmp97 = tl.full(tmp96.shape, 1.0, tmp96.dtype)
    tmp98 = tmp96 + tmp97
    tmp99 = tl.where(tmp30, tmp90, tmp98)
    tmp100 = tmp0 >= 1792
    tl.full([1], 1792, tl.int64)
    tmp103 = tl.load(in_ptr11 + x1, tmp100 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp104 = tl.full(tmp103.shape, 0.0, tmp103.dtype)
    tmp105 = tl.where(tmp100, tmp103, tmp104)
    tmp106 = tl.full(tmp105.shape, 1.0, tmp105.dtype)
    tmp107 = tmp105 + tmp106
    tmp108 = tl.where(tmp30, tmp99, tmp107)
    tmp109 = tmp0 >= 2048
    tl.full([1], 2048, tl.int64)
    tmp112 = tl.load(in_ptr12 + x1, tmp109 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp113 = tl.full(tmp112.shape, 0.0, tmp112.dtype)
    tmp114 = tl.where(tmp109, tmp112, tmp113)
    tmp115 = tl.full(tmp114.shape, 1.0, tmp114.dtype)
    tmp116 = tmp114 + tmp115
    tmp117 = tl.where(tmp30, tmp108, tmp116)
    tmp118 = tmp0 >= 2304
    tl.full([1], 2304, tl.int64)
    tmp121 = tl.load(in_ptr13 + x1, tmp118 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp118, tmp121, tmp122)
    tmp124 = tl.full(tmp123.shape, 1.0, tmp123.dtype)
    tmp125 = tmp123 + tmp124
    tmp126 = tl.where(tmp30, tmp117, tmp125)
    tmp127 = tmp0 >= 2560
    tl.full([1], 2560, tl.int64)
    tmp130 = tl.load(in_ptr14 + x1, tmp127 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp131 = tl.full(tmp130.shape, 0.0, tmp130.dtype)
    tmp132 = tl.where(tmp127, tmp130, tmp131)
    tmp133 = tl.full(tmp132.shape, 1.0, tmp132.dtype)
    tmp134 = tmp132 + tmp133
    tmp135 = tl.where(tmp30, tmp126, tmp134)
    tmp136 = tmp0 >= 2816
    tl.full([1], 2816, tl.int64)
    tmp139 = tl.load(in_ptr15 + x1, tmp136 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp136, tmp139, tmp140)
    tmp142 = tl.full(tmp141.shape, 1.0, tmp141.dtype)
    tmp143 = tmp141 + tmp142
    tmp144 = tl.where(tmp30, tmp135, tmp143)
    tmp145 = tmp0 >= 3024
    tl.full([1], 3024, tl.int64)
    tmp148 = tl.load(in_ptr16 + x1, tmp145 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp149 = tl.full(tmp148.shape, 0.0, tmp148.dtype)
    tmp150 = tl.where(tmp145, tmp148, tmp149)
    tmp151 = tl.full(tmp150.shape, 1.0, tmp150.dtype)
    tmp152 = tmp150 + tmp151
    tmp153 = tl.where(tmp30, tmp144, tmp152)
    tmp154 = tmp0 >= 3232
    tl.full([1], 3232, tl.int64)
    tmp157 = tl.load(in_ptr17 + x1, tmp154 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp158 = tl.full(tmp157.shape, 0.0, tmp157.dtype)
    tmp159 = tl.where(tmp154, tmp157, tmp158)
    tmp160 = tl.full(tmp159.shape, 1.0, tmp159.dtype)
    tmp161 = tmp159 + tmp160
    tmp162 = tl.where(tmp30, tmp153, tmp161)
    tl.store(out_ptr0 + x2, tmp162, xmask)


@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x1, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp11 = tl.load(in_ptr1 + x1, tmp8 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.full(tmp13.shape, 1.0, tmp13.dtype)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tmp17 = tmp0 < 128
    tmp18 = tl.full([1], 192, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = tl.load(in_ptr2 + x1, tmp20 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full(tmp23.shape, 1.0, tmp23.dtype)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.where(tmp20, tmp25, tmp16)
    tmp27 = tmp0 >= 192
    tmp28 = tl.full([1], 256, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr3 + x1, tmp30 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp30, tmp31, tmp32)
    tmp34 = tl.full(tmp33.shape, 1.0, tmp33.dtype)
    tmp35 = tmp33 + tmp34
    tmp36 = tl.where(tmp30, tmp35, tmp26)
    tmp37 = tmp0 >= 256
    tl.full([1], 384, tl.int64)
    tmp40 = tl.load(in_ptr4 + x1, tmp37 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 1.0, tmp42.dtype)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.where(tmp30, tmp36, tmp44)
    tmp46 = tmp0 >= 384
    tl.full([1], 512, tl.int64)
    tmp49 = tl.load(in_ptr5 + x1, tmp46 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 =
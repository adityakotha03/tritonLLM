import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_max_pool3d_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16 % 32
    x0 = xindex % 16
    x1 = xindex // 16 % 32
    x3 = xindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (5 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (5 + x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (6 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (6 + x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (7 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (7 + x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (8 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (9 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (9 + x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (10 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (10 + x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (11 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (11 + x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (12 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (13 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (13 + x0), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (14 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr1 + (14 + x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (15 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr1 + (15 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = triton_helpers.maximum(tmp2, tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp14, tmp17)
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(tmp18, tmp22)
    tmp27 = tmp25 + tmp26
    tmp28 = triton_helpers.maximum(tmp23, tmp27)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(tmp28, tmp32)
    tmp37 = tmp35 + tmp36
    tmp38 = triton_helpers.maximum(tmp33, tmp37)
    tmp42 = tmp40 + tmp41
    tmp43 = triton_helpers.maximum(tmp38, tmp42)
    tmp47 = tmp45 + tmp46
    tmp48 = triton_helpers.maximum(tmp43, tmp47)
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(tmp48, tmp52)
    tmp57 = tmp55 + tmp56
    tmp58 = triton_helpers.maximum(tmp53, tmp57)
    tmp62 = tmp60 + tmp61
    tmp63 = triton_helpers.maximum(tmp58, tmp62)
    tmp67 = tmp65 + tmp66
    tmp68 = triton_helpers.maximum(tmp63, tmp67)
    tmp72 = tmp70 + tmp71
    tmp73 = triton_helpers.maximum(tmp68, tmp72)
    tl.store(out_ptr0 + x4, tmp73, xmask)
    tl.store(out_ptr1 + (x0 + 16 * x2 + 256 * x1 + 4096 * x3), tmp73, xmask)


@triton.jit
def triton_poi_fused_convolution_max_pool3d_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16 % 32
    x0 = xindex % 16
    x1 = xindex // 16 % 32
    x3 = xindex // 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + x0 + 16 * x2 + 256 * x1 + 4096 * x3), xmask,
        eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (4 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (5 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (5 + x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (6 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (6 + x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (7 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (7 + x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (8 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (9 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (9 + x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (10 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (10 + x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (11 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (11 + x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (12 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (13 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (13 + x0), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (14 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr1 + (14 + x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (15 + x0 + 16 * x2 + 256 * x1 + 4096 * x3),
        xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr1 + (15 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = triton_helpers.maximum(tmp2, tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp14, tmp17)
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(tmp18, tmp22)
    tmp27 = tmp25 + tmp26
    tmp28 = triton_helpers.maximum(tmp23, tmp27)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(tmp28, tmp32)
    tmp37 = tmp35 + tmp36
    tmp38 = triton_helpers.maximum(tmp33, tmp37)
    tmp42 = tmp40 + tmp41
    tmp43 = triton_helpers.maximum(tmp38, tmp42)
    tmp47 = tmp45 + tmp46
    tmp48 = triton_helpers.maximum(tmp43, tmp47)
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(tmp48, tmp52)
    tmp57 = tmp55 + tmp56
    tmp58 = triton_helpers.maximum(tmp53, tmp57)
    tmp62 = tmp60 + tmp61
    tmp63 = triton_helpers.maximum(tmp58, tmp62)
    tmp67 = tmp65 + tmp66
    tmp68 = triton_helpers.maximum(tmp63, tmp67)
    tmp72 = tmp70 + tmp71
    tmp73 = triton_helpers.maximum(tmp68, tmp72)
    tl.store(out_ptr0 + x4, tmp73, xmask)
    tl.store(out_ptr1 + (x0 + 16 * x2 + 256 * x1 + 4096 * x3), tmp73, xmask)


@triton.jit
def triton_poi_fused_convolution_max_pool3d_2(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 32
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x2 + 256 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (4 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (5 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (6 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp13 = tl.load(in_ptr0 + (7 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (8 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (9 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp19 = tl.load(in_ptr0 + (10 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp21 = tl.load(in_ptr0 + (11 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp23 = tl.load(in_ptr0 + (12 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp25 = tl.load(in_ptr0 + (13 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp27 = tl.load(in_ptr0 + (14 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp29 = tl.load(in_ptr0 + (15 + x0 + 16 * x2 + 256 * x1), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp24 = triton_helpers.maximum(tmp22, tmp23)
    tmp26 = triton_helpers.maximum(tmp24, tmp25)
    tmp28 = triton_helpers.maximum(tmp26, tmp27)
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tl.store(out_ptr0 + x3, tmp30, xmask)


@triton.jit
def triton_poi_fused_sub_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_sigmoid_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp1 * tmp0
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 136496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0), xmask)
    tmp2 = tl.load(in_ptr0 + (32 + x0), xmask)
    tmp3 = tl.load(in_ptr0 + (48 + x0), xmask)
    tmp4 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp5 = tl.load(in_ptr0 + (80 + x0), xmask)
    tmp6 = tl.load(in_ptr0 + (96 + x0), xmask)
    tmp7 = tl.load(in_ptr0 + (112 + x0), xmask)
    tmp8 = tl.load(in_ptr0 + (128 + x0), xmask)
    tmp9 = tl.load(in_ptr0 + (144 + x0), xmask)
    tmp10 = tl.load(in_ptr0 + (160 + x0), xmask)
    tmp11 = tl.load(in_ptr0 + (176 + x0), xmask)
    tmp12 = tl.load(in_ptr0 + (192 + x0), xmask)
    tmp13 = tl.load(in_ptr0 + (208 + x0), xmask)
    tmp14 = tl.load(in_ptr0 + (224 + x0), xmask)
    tmp15 = tl.load(in_ptr0 + (240 + x0), xmask)
    tmp16 = tl.load(in_ptr0 + (256 + x0), xmask)
    tmp17 = tl.load(in_ptr0 + (272 + x0), xmask)
    tmp18 = tl.load(in_ptr0 + (288 + x0), xmask)
    tmp19 = tl.load(in_ptr0 + (304 + x0), xmask)
    tmp20 = tl.load(in_ptr0 + (320 + x0), xmask)
    tmp21 = tl.load(in_ptr0 + (336 + x0), xmask)
    tmp22 = tl.load(in_ptr0 + (352 + x0), xmask)
    tmp23 = tl.load(in_ptr0 + (368 + x0), xmask)
    tmp24 = tl.load(in_ptr0 + (384 + x0), xmask)
    tmp25 = tl.load(in_ptr0 + (400 + x0), xmask)
    tmp26 = tl.load(in_ptr0 + (416 + x0), xmask)
    tmp27 = tl.load(in_ptr0 + (432 + x0), xmask)
    tmp28 = tl.load(in_ptr0 + (448 + x0), xmask)
    tmp29 = tl.load(in_ptr0 + (464 + x0), xmask)
    tmp30 = tl.load(in_ptr0 + (480 + x0), xmask)
    tmp31 = tl.load(in_ptr0 + (496 + x0), xmask)
    tmp32 = tl.load(in_ptr0 + (512 + x0), xmask)
    tmp33 = tl.load(in_ptr0 + (528 + x0), xmask)
    tmp34 = tl.load(in_ptr0 + (544 + x0), xmask)
    tmp35 = tl.load(in_ptr0 + (560 + x0), xmask)
    tmp36 = tl.load(in_ptr0 + (576 + x0), xmask)
    tmp37 = tl.load(in_ptr0 + (592 + x0), xmask)
    tmp38 = tl.load(in_ptr0 + (608 + x0), xmask)
    tmp39 = tl.load(in_ptr0 + (624 + x0), xmask)
    tmp40 = tl.load(in_ptr0 + (640 + x0), xmask)
    tmp41 = tl.load(in_ptr0 + (656 + x0), xmask)
    tmp42 = tl.load(in_ptr0 + (672 + x0), xmask)
    tmp43 = tl.load(in_ptr0 + (688 + x0), xmask)
    tmp44 = tl.load(in_ptr0 + (704 + x0), xmask)
    tmp45 = tl.load(in_ptr0 + (720 + x0), xmask)
    tmp46 = tl.load(in_ptr0 + (736 + x0), xmask)
    tmp47 = tl.load(in_ptr0 + (752 + x0), xmask)
    tmp48 = tl.load(in_ptr0 + (768 + x0), xmask)
    tmp49 = tl.load(in_ptr0 + (784 + x0), xmask)
    tmp50 = tl.load(in_ptr0 + (800 + x0), xmask)
    tmp51 = tl.load(in_ptr0 + (816 + x0), xmask)
    tmp52 = tl.load(in_ptr0 + (832 + x0), xmask)
    tmp53 = tl.load(in_ptr0 + (848 + x0), xmask)
    tmp54 = tl.load(in_ptr0 + (864 + x0), xmask)
    tmp55 = tl.load(in_ptr0 + (880 + x0), xmask)
    tmp56 = tl.load(in_ptr0 + (896 + x0), xmask)
    tmp57 = tl.load(in_ptr0 + (912 + x0), xmask)
    tmp58 = tl.load(in_ptr0 + (928 + x0), xmask)
    tmp59 = tl.load(in_ptr0 + (944 + x0), xmask)
    tmp60 = tl.load(in_ptr0 + (960 + x0), xmask)
    tmp61 = tl.load(in_ptr0 + (976 + x0), xmask)
    tmp62 = tl.load(in_ptr0 + (992 + x0), xmask)
    tmp63 = tl.load(in_ptr0 + (1008 + x0), xmask)
    tmp64 = tl.load(in_ptr0 + (1024 + x0), xmask)
    tmp65 = tl.load(in_ptr0 + (1040 + x0), xmask)
    tmp66 = tl.load(in_ptr0 + (1056 + x0), xmask)
    tmp67 = tl.load(in_ptr0 + (1072 + x0), xmask)
    tmp68 = tl.load(in_ptr0 + (1088 + x0), xmask)
    tmp69 = tl.load(in_ptr0 + (1104 + x0), xmask)
    tmp70 = tl.load(in_ptr0 + (1120 + x0), xmask)
    tmp71 = tl.load(in_ptr0 + (1136 + x0), xmask)
    tmp72 = tl.load(in_ptr0 + (1152 + x0), xmask)
    tmp73 = tl.load(in_ptr0 + (1168 + x0), xmask)
    tmp74 = tl.load(in_ptr0 + (1184 + x0), xmask)
    tmp75 = tl.load(in_ptr0 + (1200 + x0), xmask)
    tmp76 = tl.load(in_ptr0 + (1216 + x0), xmask)
    tmp77 = tl.load(in_ptr0 + (1232 + x0), xmask)
    tmp78 = tl.load(in_ptr0 + (1248 + x0), xmask)
    tmp79 = tl.load(in_ptr0 + (1264 + x0), xmask)
    tmp80 = tl.load(in_ptr0 + (1280 + x0), xmask)
    tmp81 = tl.load(in_ptr0 + (1296 + x0), xmask)
    tmp82 = tl.load(in_ptr0 + (1312 + x0), xmask)
    tmp83 = tl.load(in_ptr0 + (1328 + x0), xmask)
    tmp84 = tl.load(in_ptr0 + (1344 + x0), xmask)
    tmp85 = tl.load(in_ptr0 + (1360 + x0), xmask)
    tmp86 = tl.load(in_ptr0 + (1376 + x0), xmask)
    tmp87 = tl.load(in_ptr0 + (1392 + x0), xmask)
    tmp88 = tl.load(in_ptr0 + (1408 + x0), xmask)
    tmp89 = tl.load(in_ptr0 + (1424 + x0), xmask)
    tmp90 = tl.load(in_ptr0 + (1440 + x0), xmask)
    tmp91 = tl.load(in_ptr0 + (1456 + x0), xmask)
    tmp92 = tl.load(in_ptr0 + (1472 + x0), xmask)
    tmp93 = tl.load(in_ptr0 + (1488 + x0), xmask)
    tmp94 = tl.load(in_ptr0 + (1504 + x0), xmask)
    tmp95 = tl.load(in_ptr0 + (1520 + x0), xmask)
    tmp96 = tl.load(in_ptr0 + (1536 + x0), xmask)
    tmp97 = tl.load(in_ptr0 + (1552 + x0), xmask)
    tmp98 = tl.load(in_ptr0 + (1568 + x0), xmask)
    tmp99 = tl.load(in_ptr0 + (1584 + x0), xmask)
    tmp100 = tl.load(in_ptr0 + (1600 + x0), xmask)
    tmp101 = tl.load(in_ptr0 + (1616 + x0), xmask)
    tmp102 = tl.load(in_ptr0 + (1632 + x0), xmask)
    tmp103 = tl.load(in_ptr0 + (1648 + x0), xmask)
    tmp104 = tl.load(in_ptr0 + (1664 + x0), xmask)
    tmp105 = tl.load(in_ptr0 + (1680 + x0), xmask)
    tmp106 = tl.load(in_ptr0 + (1696 + x0), xmask)
    tmp107 = tl.load(in_ptr0 + (1712 + x0), xmask)
    tmp108 = tl.load(in_ptr0 + (1728 + x0), xmask)
    tmp109 = tl.load(in_ptr0 + (1744 + x0), xmask)
    tmp110 = tl.load(in_ptr0 + (1760 + x0), xmask)
    tmp111 = tl.load(in_ptr0 + (1776 + x0), xmask)
    tmp112 = tl.load(in_ptr0 + (1792 + x0), xmask)
    tmp113 = tl.load(in_ptr0 + (1808 + x0), xmask)
    tmp114 = tl.load(in_ptr0 + (1824 + x0), xmask)
    tmp115 = tl.load(in_ptr0 + (1840 + x0), xmask)
    tmp116 = tl.load(in_ptr0 + (1856 + x0), xmask)
    tmp117 = tl.load(in_ptr0 + (1872 + x0), xmask)
    tmp118 = tl.load(in_ptr0 + (1888 + x0), xmask)
    tmp119 = tl.load(in_ptr0 + (1904 + x0), xmask)
    tmp120 = tl.load(in_ptr0 + (1920 + x0), xmask)
    tmp121 = tl.load(in_ptr0 + (1936 + x0), xmask)
    tmp122 = tl.load(in_ptr0 + (1952 + x0), xmask)
    tmp123 = tl.load(in_ptr0 + (1968 + x0), xmask)
    tmp124 = tl.load(in_ptr0 + (1984 + x0), xmask)
    tmp125 = tl.load(in_ptr0 + (2000 + x0), xmask)
    tmp126 = tl.load(in_ptr0 + (2016 + x0), xmask)
    tmp127 = tl.load(in_ptr0 + (2032 + x0), xmask)
    tmp128 = tl.load(in_ptr0 + (
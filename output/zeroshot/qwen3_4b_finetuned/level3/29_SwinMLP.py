import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 196896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 196 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 96 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (64 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (65 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (66 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (67 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (68 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (69 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (70 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (71 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (72 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (73 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (74 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (75 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (76 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (77 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (78 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (79 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (80 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (81 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (82 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (83 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (84 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (85 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (86 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (87 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (88 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (89 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (90 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (91 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (92 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (93 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (94 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (95 + 96 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp1
    tmp4 = tmp3 + tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tmp7 / 96.0
    tmp9 = 96.0
    tmp10 = tmp9 - tmp8
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.sqrt(tmp12)
    tmp14 = 1e-05
    tmp15 = tmp6 + tmp14
    tmp16 = tmp15 / tmp13
    tmp17 = tmp13.to(tl.float32)
    tmp18 = tmp16 / tmp17
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 256
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4 * x2 + 196 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 49 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 392 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 196 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_native_layer_norm_4(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + 192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + 192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + 192 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 / tmp23
    tl.store(out_ptr0 + x0, tmp26, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)


@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 128
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2 * x2 + 392 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 196 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_native_layer_norm_6(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (2 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 4 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 / tmp23
    tl.store(out_ptr0 + x2, tmp26, xmask)
    tl.store(out_ptr1 + x2, tmp23, xmask)


@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 256
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 4
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4 * x2 + 196 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 49 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_native_layer_norm_8(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 8 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 8 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (6 + x0 + 8 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 / tmp23
    tl.store(out_ptr0 + x2, tmp26, xmask)
    tl.store(out_ptr1 + x2, tmp23, xmask)


@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK:
    tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 8
    y1 = yindex // 8
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 8 * x2 + 392 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 49 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_native_layer_norm_10(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (8 + x0 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + x0 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (12 + x0 + 16 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 / tmp23
    tl.store(out_ptr0 + x2, tmp26, xmask)
    tl.store(out_ptr1 + x2, tmp23, xmask)


@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 16
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16 * x2 + 784 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 49 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_native_layer_norm_12(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (24 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.sqrt(tmp22)
    tmp24 = 1e-05
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 / tmp23
    tl.store(out_ptr0 + x2, tmp26, xmask)
    tl.store(out_ptr1 + x2, tmp23, xmask)


@triton.jit
def triton_poi_fused
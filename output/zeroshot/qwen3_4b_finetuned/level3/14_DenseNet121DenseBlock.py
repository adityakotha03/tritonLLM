import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_0(in_ptr0, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp48 = tmp47 + tmp46
    tmp50 = tmp49 + tmp48
    tmp52 = tmp51 + tmp50
    tmp54 = tmp53 + tmp52
    tmp56 = tmp55 + tmp54
    tmp58 = tmp57 + tmp56
    tmp60 = tmp59 + tmp58
    tmp62 = tmp61 + tmp60
    tmp63 = 32.0
    tmp64 = tmp62 / tmp63
    tmp65 = tmp0 - tmp64
    tmp66 = tmp65 * tmp65
    tmp67 = tmp1 - tmp64
    tmp68 = tmp67 * tmp67
    tmp69 = tmp66 + tmp68
    tmp70 = tmp3 - tmp64
    tmp71 = tmp70 * tmp70
    tmp72 = tmp69 + tmp71
    tmp73 = tmp5 - tmp64
    tmp74 = tmp73 * tmp73
    tmp75 = tmp72 + tmp74
    tmp76 = tmp7 + tmp64
    tmp77 = tmp76 * tmp76
    tmp78 = tmp75 + tmp77
    tmp79 = tmp9 + tmp64
    tmp80 = tmp79 * tmp79
    tmp81 = tmp78 + tmp80
    tmp82 = tmp11 + tmp64
    tmp83 = tmp82 * tmp82
    tmp84 = tmp81 + tmp83
    tmp85 = tmp13 + tmp64
    tmp86 = tmp85 * tmp85
    tmp87 = tmp84 + tmp86
    tmp88 = tmp15 + tmp64
    tmp89 = tmp88 * tmp88
    tmp90 = tmp87 + tmp89
    tmp91 = tmp17 + tmp64
    tmp92 = tmp91 * tmp91
    tmp93 = tmp90 + tmp92
    tmp94 = tmp19 + tmp64
    tmp95 = tmp94 * tmp94
    tmp96 = tmp93 + tmp95
    tmp97 = tmp21 + tmp64
    tmp98 = tmp97 * tmp97
    tmp99 = tmp96 + tmp98
    tmp100 = tmp23 + tmp64
    tmp101 = tmp100 * tmp100
    tmp102 = tmp99 + tmp101
    tmp103 = tmp25 + tmp64
    tmp104 = tmp103 * tmp103
    tmp105 = tmp102 + tmp104
    tmp106 = tmp27 + tmp64
    tmp107 = tmp106 * tmp106
    tmp108 = tmp105 + tmp107
    tmp109 = tmp29 + tmp64
    tmp110 = tmp109 * tmp109
    tmp111 = tmp108 + tmp110
    tmp112 = tmp31 + tmp64
    tmp113 = tmp112 * tmp112
    tmp114 = tmp111 + tmp113
    tmp115 = tmp33 + tmp64
    tmp116 = tmp115 * tmp115
    tmp117 = tmp114 + tmp116
    tmp118 = tmp35 + tmp64
    tmp119 = tmp118 * tmp118
    tmp120 = tmp117 + tmp119
    tmp121 = tmp37 + tmp64
    tmp122 = tmp121 * tmp121
    tmp123 = tmp120 + tmp122
    tmp124 = tmp39 + tmp64
    tmp125 = tmp124 * tmp124
    tmp126 = tmp123 + tmp125
    tmp127 = tmp41 + tmp64
    tmp128 = tmp127 * tmp127
    tmp129 = tmp126 + tmp128
    tmp130 = tmp43 + tmp64
    tmp131 = tmp130 * tmp130
    tmp132 = tmp129 + tmp131
    tmp133 = tmp45 + tmp64
    tmp134 = tmp133 * tmp133
    tmp135 = tmp132 + tmp134
    tmp136 = tmp47 + tmp64
    tmp137 = tmp136 * tmp136
    tmp138 = tmp135 + tmp137
    tmp139 = tmp49 + tmp64
    tmp140 = tmp139 * tmp139
    tmp141 = tmp138 + tmp140
    tmp142 = tmp51 + tmp64
    tmp143 = tmp142 * tmp142
    tmp144 = tmp141 + tmp143
    tmp145 = tmp53 + tmp64
    tmp146 = tmp145 * tmp145
    tmp147 = tmp144 + tmp146
    tmp148 = tmp55 + tmp64
    tmp149 = tmp148 * tmp148
    tmp150 = tmp147 + tmp149
    tmp151 = tmp57 + tmp64
    tmp152 = tmp151 * tmp151
    tmp153 = tmp150 + tmp152
    tmp154 = tmp59 + tmp64
    tmp155 = tmp154 * tmp154
    tmp156 = tmp153 + tmp155
    tmp157 = tmp61 + tmp64
    tmp158 = tmp157 * tmp157
    tmp159 = tmp156 + tmp158
    tmp160 = 30.0
    tmp161 = tmp159 / tmp160
    tmp162 = tmp65 / tmp161
    tmp163 = 1e-05
    tmp164 = tmp162 + tmp163
    tmp165 = libdevice.rsqrt(tmp164)
    tmp166 = tmp0 * tmp165
    tmp167 = tmp1 * tmp165
    tmp168 = tmp166 + tmp167
    tmp169 = tmp3 * tmp165
    tmp170 = tmp168 + tmp169
    tmp171 = tmp5 * tmp165
    tmp172 = tmp170 + tmp171
    tmp173 = tmp7 * tmp165
    tmp174 = tmp172 + tmp173
    tmp175 = tmp9 * tmp165
    tmp176 = tmp174 + tmp175
    tmp177 = tmp11 * tmp165
    tmp178 = tmp176 + tmp177
    tmp179 = tmp13 * tmp165
    tmp180 = tmp178 + tmp179
    tmp181 = tmp15 * tmp165
    tmp182 = tmp180 + tmp181
    tmp183 = tmp17 * tmp165
    tmp184 = tmp182 + tmp183
    tmp185 = tmp19 * tmp165
    tmp186 = tmp184 + tmp185
    tmp187 = tmp21 * tmp165
    tmp188 = tmp186 + tmp187
    tmp189 = tmp23 * tmp165
    tmp190 = tmp188 + tmp189
    tmp191 = tmp25 * tmp165
    tmp192 = tmp190 + tmp191
    tmp193 = tmp27 * tmp165
    tmp194 = tmp192 + tmp193
    tmp195 = tmp29 * tmp165
    tmp196 = tmp194 + tmp195
    tmp197 = tmp31 * tmp165
    tmp198 = tmp196 + tmp197
    tmp199 = tmp165 * tmp198
    tmp200 = tl.full([1], 0, tl.int32)
    tmp201 = triton_helpers.maximum(tmp200, tmp199)
    tl.store(out_ptr0 + x0, tmp201, xmask)
    tl.store(out_ptr1 + x0, tmp165, xmask)
    tl.store(out_ptr2 + x0, tmp161, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_1(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 - tmp3
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0) / tmp5
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15 * tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tl.store(out_ptr2 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_2(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 - tmp3
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0) / tmp5
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15 * tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tl.store(out_ptr2 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_3(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 128 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 96 * x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + 128 * x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + 128 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 - tmp3
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0) / tmp5
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15 * tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tl.store(out_ptr2 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_4(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 160 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 128 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 128 * x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + 160 * x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + 160 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 - tmp3
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0) / tmp5
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15 * tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tl.store(out_ptr2 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_5(in_ptr0, in_ptr1,
    in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 192 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 160 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + 160 * x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + 192 * x0, xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + 192 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 - tmp3
    tmp5 = 32.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp0 - tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.sum(tmp9, 0) / tmp5
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp6 * tmp14
    tmp16 = tmp15 * tmp14
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + x0, tmp18, xmask)
    tl.store(out_ptr1 + x0, tmp14, xmask)
    tl.store(out_ptr2 + x0, tmp13, xmask)


@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11,
    in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18,
    in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25,
    in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32,
    in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39,
    in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46,
    in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53,
    in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60,
    in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67,
    in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74,
    in_ptr75, in_ptr76, in_ptr77, in_ptr78, in_ptr79, in_ptr80, in_ptr81,
    in_ptr82, in_ptr83, in_ptr84, in_ptr85, in_ptr86, in_ptr87, in_ptr88,
    in_ptr89, in_ptr90, in_ptr91, in_ptr92, in_ptr93, in_ptr94, in_ptr95,
    in_ptr96, in_ptr97, in_ptr98, in_ptr99, in_ptr100, in_ptr101, in_ptr102,
    in_ptr103, in_ptr104, in_ptr105, in_ptr106, in_ptr107, in_ptr108,
    in_ptr109, in_ptr110, in_ptr111, in_ptr112, in_ptr113, in_ptr114,
    in_ptr115, in_ptr116, in_ptr117, in_ptr118, in_ptr119, in_ptr120,
    in_ptr121, in_ptr122, in_ptr123, in_ptr124, in_ptr125, in_ptr126,
    in_ptr127, in_ptr128, in_ptr129, in_ptr130, in_ptr131, in_ptr132,
    in_ptr133, in_ptr134, in_ptr135, in_ptr136, in_ptr137, in_ptr138,
    in_ptr139, in_ptr140, in_ptr141, in_ptr142, in_ptr143, in_ptr144,
    in_ptr145, in_ptr146, in_ptr147, in_ptr148, in_ptr149, in_ptr150,
    in_ptr151, in_ptr152, in_ptr153, in_ptr154, in_ptr155, in_ptr156,
    in_ptr157, in_ptr158, in_ptr159, in_ptr160, in_ptr161, in_ptr162,
    in_ptr163, in_ptr164, in_ptr165, in_ptr166, in_ptr167, in_ptr168,
    in_ptr169, in_ptr170, in_ptr171, in_ptr172, in_ptr173, in_ptr174,
    in_ptr175, in_ptr176, in_ptr177, in_ptr178, in_ptr179, in_ptr180,
    in_ptr181, in_ptr182, in_ptr183, in_ptr184, in_ptr185, in_ptr186,
    in_ptr187, in_ptr188, in_ptr189, in_ptr190, in_ptr191, in_ptr192,
    in_ptr193, in_ptr194, in_ptr195, in_ptr196, in_ptr197, in_ptr198,
    in_ptr199, in_ptr200, in_ptr201, in_ptr202, in_ptr203, in_ptr204,
    in_ptr205, in_ptr206, in_ptr207, in_ptr208, in_ptr209, in_ptr210,
    in_ptr211, in_ptr212, in_ptr213, in_ptr214, in_ptr215, in_ptr216,
    in_ptr217, in_ptr218, in_ptr219, in_ptr220, in_ptr221, in_ptr222,
    in_ptr223, in_ptr224, in_ptr225, in_ptr226, in_ptr227, in_ptr228,
    in_ptr229, in_ptr230, in_ptr231, in_ptr232, in_ptr233, in_ptr234,
    in_ptr235, in_ptr236, in_ptr237, in_ptr238, in_ptr239, in_ptr240,
    in_ptr241, in_ptr242, in_ptr243, in_ptr244, in_ptr245, in_ptr246,
    in_ptr247, in_ptr248, in_ptr249, in_ptr250, in_ptr251, in_ptr252,
    in_ptr253, in_ptr254, in_ptr255, in_ptr256, in_ptr257, in_ptr258,
    in_ptr259, in_ptr260, in_ptr261, in_ptr262, in_ptr263, in_ptr264,
    in_ptr265, in_ptr266, in_ptr267, in_ptr268, in_ptr269, in_ptr270,
    in_ptr271, in_ptr272, in_ptr273, in_ptr274, in_ptr275, in_ptr276,
    in_ptr277, in_ptr278, in_ptr279, in_ptr280, in_ptr281, in_ptr282,
    in_ptr283, in_ptr284, in_ptr285, in_ptr286, in_ptr287, in_ptr288,
    in_ptr289, in_ptr290, in_ptr291, in_ptr292, in_ptr293, in_ptr294,
    in_ptr295, in_ptr296, in_ptr297, in_ptr298, in_ptr299, in_ptr300,
    in_ptr301, in_ptr302, in_ptr303, in_ptr304, in_ptr305, in_ptr306,
    in_ptr307, in_ptr308, in_ptr309, in_ptr310, in_ptr311, in_ptr312,
    in_ptr313, in_ptr314, in_ptr315, in_ptr316, in_ptr317, in_ptr318,
    in_ptr319, in_ptr320, in_ptr321,
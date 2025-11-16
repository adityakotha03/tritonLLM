import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = -1 + x0
    tmp1 = tl.full([1], 128, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (-128 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp4 = 0 <= x0
    tmp5 = tl.full([1], 127, tl.int64)
    tmp6 = x0 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tl.load(in_ptr0 + (-127 + x1 + 128 * x0), tmp7 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp9 = tl.where(tmp2, tmp3, tmp8)
    tmp10 = tl.load(in_ptr0 + (128 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp11 = tl.where(tmp4 & tmp6, tmp8, tmp10)
    tmp12 = tl.where(tmp2, tmp3, tmp11)
    tmp13 = tl.load(in_ptr0 + (129 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp14 = tl.where(tmp4 & tmp6, tmp11, tmp13)
    tmp15 = tl.where(tmp2, tmp3, tmp14)
    tmp16 = tl.load(in_ptr0 + (192 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp4 & tmp6, tmp14, tmp16)
    tmp18 = tl.where(tmp2, tmp3, tmp17)
    tmp19 = tl.load(in_ptr0 + (193 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp4 & tmp6, tmp17, tmp19)
    tmp21 = tl.where(tmp2, tmp3, tmp20)
    tmp22 = tl.load(in_ptr0 + (384 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp4 & tmp6, tmp20, tmp22)
    tmp24 = tl.where(tmp2, tmp3, tmp23)
    tmp25 = tl.load(in_ptr0 + (385 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp26 = tl.where(tmp4 & tmp6, tmp23, tmp25)
    tmp27 = tl.where(tmp2, tmp3, tmp26)
    tmp28 = tl.load(in_ptr0 + (640 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp29 = tl.where(tmp4 & tmp6, tmp26, tmp28)
    tmp30 = tl.where(tmp2, tmp3, tmp29)
    tmp31 = tl.load(in_ptr0 + (641 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp32 = tl.where(tmp4 & tmp6, tmp29, tmp31)
    tmp33 = tl.where(tmp2, tmp3, tmp32)
    tmp34 = tl.load(in_ptr0 + (960 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp4 & tmp6, tmp32, tmp34)
    tmp36 = tl.where(tmp2, tmp3, tmp35)
    tmp37 = tl.load(in_ptr0 + (961 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp38 = tl.where(tmp4 & tmp6, tmp35, tmp37)
    tmp39 = tl.where(tmp2, tmp3, tmp38)
    tmp40 = tl.load(in_ptr0 + (1536 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp41 = tl.where(tmp4 & tmp6, tmp38, tmp40)
    tmp42 = tl.where(tmp2, tmp3, tmp41)
    tmp43 = tl.load(in_ptr0 + (1537 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp44 = tl.where(tmp4 & tmp6, tmp41, tmp43)
    tmp45 = tl.where(tmp2, tmp3, tmp44)
    tmp46 = tl.load(in_ptr0 + (2304 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp47 = tl.where(tmp4 & tmp6, tmp44, tmp46)
    tmp48 = tl.where(tmp2, tmp3, tmp47)
    tmp49 = tl.load(in_ptr0 + (2305 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp4 & tmp6, tmp47, tmp49)
    tmp51 = tl.where(tmp2, tmp3, tmp50)
    tmp52 = tl.load(in_ptr0 + (3584 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp53 = tl.where(tmp4 & tmp6, tmp50, tmp52)
    tmp54 = tl.where(tmp2, tmp3, tmp53)
    tmp55 = tl.load(in_ptr0 + (3585 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp56 = tl.where(tmp4 & tmp6, tmp53, tmp55)
    tmp57 = tl.where(tmp2, tmp3, tmp56)
    tmp58 = tl.load(in_ptr0 + (5120 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp59 = tl.where(tmp4 & tmp6, tmp56, tmp58)
    tmp60 = tl.where(tmp2, tmp3, tmp59)
    tmp61 = tl.load(in_ptr0 + (5121 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp62 = tl.where(tmp4 & tmp6, tmp59, tmp61)
    tmp63 = tl.where(tmp2, tmp3, tmp62)
    tmp64 = tl.load(in_ptr0 + (7168 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp65 = tl.where(tmp4 & tmp6, tmp62, tmp64)
    tmp66 = tl.where(tmp2, tmp3, tmp65)
    tmp67 = tl.load(in_ptr0 + (7169 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp68 = tl.where(tmp4 & tmp6, tmp65, tmp67)
    tmp69 = tl.where(tmp2, tmp3, tmp68)
    tmp70 = tl.load(in_ptr0 + (10240 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp71 = tl.where(tmp4 & tmp6, tmp68, tmp70)
    tmp72 = tl.where(tmp2, tmp3, tmp71)
    tmp73 = tl.load(in_ptr0 + (10241 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp74 = tl.where(tmp4 & tmp6, tmp71, tmp73)
    tmp75 = tl.where(tmp2, tmp3, tmp74)
    tmp76 = tl.load(in_ptr0 + (14336 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp77 = tl.where(tmp4 & tmp6, tmp74, tmp76)
    tmp78 = tl.where(tmp2, tmp3, tmp77)
    tmp79 = tl.load(in_ptr0 + (14337 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp80 = tl.where(tmp4 & tmp6, tmp77, tmp79)
    tmp81 = tl.where(tmp2, tmp3, tmp80)
    tmp82 = tl.load(in_ptr0 + (20480 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp83 = tl.where(tmp4 & tmp6, tmp80, tmp82)
    tmp84 = tl.where(tmp2, tmp3, tmp83)
    tmp85 = tl.load(in_ptr0 + (20481 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp86 = tl.where(tmp4 & tmp6, tmp83, tmp85)
    tmp87 = tl.where(tmp2, tmp3, tmp86)
    tmp88 = tl.load(in_ptr0 + (28672 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp89 = tl.where(tmp4 & tmp6, tmp86, tmp88)
    tmp90 = tl.where(tmp2, tmp3, tmp89)
    tmp91 = tl.load(in_ptr0 + (28673 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp92 = tl.where(tmp4 & tmp6, tmp89, tmp91)
    tmp93 = tl.where(tmp2, tmp3, tmp92)
    tmp94 = tl.load(in_ptr0 + (40960 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp95 = tl.where(tmp4 & tmp6, tmp92, tmp94)
    tmp96 = tl.where(tmp2, tmp3, tmp95)
    tmp97 = tl.load(in_ptr0 + (40961 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp98 = tl.where(tmp4 & tmp6, tmp95, tmp97)
    tmp99 = tl.where(tmp2, tmp3, tmp98)
    tmp100 = tl.load(in_ptr0 + (57344 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp101 = tl.where(tmp4 & tmp6, tmp98, tmp100)
    tmp102 = tl.where(tmp2, tmp3, tmp101)
    tmp103 = tl.load(in_ptr0 + (57345 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp104 = tl.where(tmp4 & tmp6, tmp101, tmp103)
    tmp105 = tl.where(tmp2, tmp3, tmp104)
    tmp106 = tl.load(in_ptr0 + (81920 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp107 = tl.where(tmp4 & tmp6, tmp104, tmp106)
    tmp108 = tl.where(tmp2, tmp3, tmp107)
    tmp109 = tl.load(in_ptr0 + (81921 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp110 = tl.where(tmp4 & tmp6, tmp107, tmp109)
    tmp111 = tl.where(tmp2, tmp3, tmp110)
    tmp112 = tl.load(in_ptr0 + (114688 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp113 = tl.where(tmp4 & tmp6, tmp110, tmp112)
    tmp114 = tl.where(tmp2, tmp3, tmp113)
    tmp115 = tl.load(in_ptr0 + (114689 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp116 = tl.where(tmp4 & tmp6, tmp113, tmp115)
    tmp117 = tl.where(tmp2, tmp3, tmp116)
    tmp118 = tl.load(in_ptr0 + (163840 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp119 = tl.where(tmp4 & tmp6, tmp116, tmp118)
    tmp120 = tl.where(tmp2, tmp3, tmp119)
    tmp121 = tl.load(in_ptr0 + (163841 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp122 = tl.where(tmp4 & tmp6, tmp119, tmp121)
    tmp123 = tl.where(tmp2, tmp3, tmp122)
    tmp124 = tl.load(in_ptr0 + (229376 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp125 = tl.where(tmp4 & tmp6, tmp122, tmp124)
    tmp126 = tl.where(tmp2, tmp3, tmp125)
    tmp127 = tl.load(in_ptr0 + (229377 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp128 = tl.where(tmp4 & tmp6, tmp125, tmp127)
    tmp129 = tl.where(tmp2, tmp3, tmp128)
    tmp130 = tl.load(in_ptr0 + (314576 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp131 = tl.where(tmp4 & tmp6, tmp128, tmp130)
    tmp132 = tl.where(tmp2, tmp3, tmp131)
    tmp133 = tl.load(in_ptr0 + (314577 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp134 = tl.where(tmp4 & tmp6, tmp131, tmp133)
    tmp135 = tl.where(tmp2, tmp3, tmp134)
    tmp136 = tl.load(in_ptr0 + (439040 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp137 = tl.where(tmp4 & tmp6, tmp134, tmp136)
    tmp138 = tl.where(tmp2, tmp3, tmp137)
    tmp139 = tl.load(in_ptr0 + (439041 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp140 = tl.where(tmp4 & tmp6, tmp137, tmp139)
    tmp141 = tl.where(tmp2, tmp3, tmp140)
    tmp142 = tl.load(in_ptr0 + (629184 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp143 = tl.where(tmp4 & tmp6, tmp140, tmp142)
    tmp144 = tl.where(tmp2, tmp3, tmp143)
    tmp145 = tl.load(in_ptr0 + (629185 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp146 = tl.where(tmp4 & tmp6, tmp143, tmp145)
    tmp147 = tl.where(tmp2, tmp3, tmp146)
    tmp148 = tl.load(in_ptr0 + (943776 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp149 = tl.where(tmp4 & tmp6, tmp146, tmp148)
    tmp150 = tl.where(tmp2, tmp3, tmp149)
    tmp151 = tl.load(in_ptr0 + (943777 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp152 = tl.where(tmp4 & tmp6, tmp149, tmp151)
    tmp153 = tl.where(tmp2, tmp3, tmp152)
    tmp154 = tl.load(in_ptr0 + (1468032 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp155 = tl.where(tmp4 & tmp6, tmp152, tmp154)
    tmp156 = tl.where(tmp2, tmp3, tmp155)
    tmp157 = tl.load(in_ptr0 + (1468033 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp158 = tl.where(tmp4 & tmp6, tmp155, tmp157)
    tmp159 = tl.where(tmp2, tmp3, tmp158)
    tmp160 = tl.load(in_ptr0 + (2147488 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp161 = tl.where(tmp4 & tmp6, tmp158, tmp160)
    tmp162 = tl.where(tmp2, tmp3, tmp161)
    tmp163 = tl.load(in_ptr0 + (2147489 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp164 = tl.where(tmp4 & tmp6, tmp161, tmp163)
    tmp165 = tl.where(tmp2, tmp3, tmp164)
    tmp166 = tl.load(in_ptr0 + (2936064 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp167 = tl.where(tmp4 & tmp6, tmp164, tmp166)
    tmp168 = tl.where(tmp2, tmp3, tmp167)
    tmp169 = tl.load(in_ptr0 + (2936065 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp170 = tl.where(tmp4 & tmp6, tmp167, tmp169)
    tmp171 = tl.where(tmp2, tmp3, tmp170)
    tmp172 = tl.load(in_ptr0 + (4012080 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp173 = tl.where(tmp4 & tmp6, tmp170, tmp172)
    tmp174 = tl.where(tmp2, tmp3, tmp173)
    tmp175 = tl.load(in_ptr0 + (4012081 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp176 = tl.where(tmp4 & tmp6, tmp173, tmp175)
    tmp177 = tl.where(tmp2, tmp3, tmp176)
    tmp178 = tl.load(in_ptr0 + (5618496 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp179 = tl.where(tmp4 & tmp6, tmp176, tmp178)
    tmp180 = tl.where(tmp2, tmp3, tmp179)
    tmp181 = tl.load(in_ptr0 + (5618497 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp182 = tl.where(tmp4 & tmp6, tmp179, tmp181)
    tmp183 = tl.where(tmp2, tmp3, tmp182)
    tmp184 = tl.load(in_ptr0 + (7786752 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp185 = tl.where(tmp4 & tmp6, tmp182, tmp184)
    tmp186 = tl.where(tmp2, tmp3, tmp185)
    tmp187 = tl.load(in_ptr0 + (7786753 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp188 = tl.where(tmp4 & tmp6, tmp185, tmp187)
    tmp189 = tl.where(tmp2, tmp3, tmp188)
    tmp190 = tl.load(in_ptr0 + (10847680 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp191 = tl.where(tmp4 & tmp6, tmp188, tmp190)
    tmp192 = tl.where(tmp2, tmp3, tmp191)
    tmp193 = tl.load(in_ptr0 + (10847681 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp194 = tl.where(tmp4 & tmp6, tmp191, tmp193)
    tmp195 = tl.where(tmp2, tmp3, tmp194)
    tmp196 = tl.load(in_ptr0 + (15245440 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp197 = tl.where(tmp4 & tmp6, tmp194, tmp196)
    tmp198 = tl.where(tmp2, tmp3, tmp197)
    tmp199 = tl.load(in_ptr0 + (15245441 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp200 = tl.where(tmp4 & tmp6, tmp197, tmp199)
    tmp201 = tl.where(tmp2, tmp3, tmp200)
    tmp202 = tl.load(in_ptr0 + (21113664 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp203 = tl.where(tmp4 & tmp6, tmp200, tmp202)
    tmp204 = tl.where(tmp2, tmp3, tmp203)
    tmp205 = tl.load(in_ptr0 + (21113665 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp206 = tl.where(tmp4 & tmp6, tmp203, tmp205)
    tmp207 = tl.where(tmp2, tmp3, tmp206)
    tmp208 = tl.load(in_ptr0 + (28715136 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp209 = tl.where(tmp4 & tmp6, tmp206, tmp208)
    tmp210 = tl.where(tmp2, tmp3, tmp209)
    tmp211 = tl.load(in_ptr0 + (28715137 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp212 = tl.where(tmp4 & tmp6, tmp209, tmp211)
    tmp213 = tl.where(tmp2, tmp3, tmp212)
    tmp214 = tl.load(in_ptr0 + (39673856 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp215 = tl.where(tmp4 & tmp6, tmp212, tmp214)
    tmp216 = tl.where(tmp2, tmp3, tmp215)
    tmp217 = tl.load(in_ptr0 + (39673857 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp218 = tl.where(tmp4 & tmp6, tmp215, tmp217)
    tmp219 = tl.where(tmp2, tmp3, tmp218)
    tmp220 = tl.load(in_ptr0 + (53820800 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp221 = tl.where(tmp4 & tmp6, tmp218, tmp220)
    tmp222 = tl.where(tmp2, tmp3, tmp221)
    tmp223 = tl.load(in_ptr0 + (53820801 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp224 = tl.where(tmp4 & tmp6, tmp221, tmp223)
    tmp225 = tl.where(tmp2, tmp3, tmp224)
    tmp226 = tl.load(in_ptr0 + (72498560 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp227 = tl.where(tmp4 & tmp6, tmp224, tmp226)
    tmp228 = tl.where(tmp2, tmp3, tmp227)
    tmp229 = tl.load(in_ptr0 + (72498561 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp230 = tl.where(tmp4 & tmp6, tmp227, tmp229)
    tmp231 = tl.where(tmp2, tmp3, tmp230)
    tmp232 = tl.load(in_ptr0 + (95712384 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp233 = tl.where(tmp4 & tmp6, tmp230, tmp232)
    tmp234 = tl.where(tmp2, tmp3, tmp233)
    tmp235 = tl.load(in_ptr0 + (95712385 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp236 = tl.where(tmp4 & tmp6, tmp233, tmp235)
    tmp237 = tl.where(tmp2, tmp3, tmp236)
    tmp238 = tl.load(in_ptr0 + (132776320 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp239 = tl.where(tmp4 & tmp6, tmp236, tmp238)
    tmp240 = tl.where(tmp2, tmp3, tmp239)
    tmp241 = tl.load(in_ptr0 + (132776321 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp242 = tl.where(tmp4 & tmp6, tmp239, tmp241)
    tmp243 = tl.where(tmp2, tmp3, tmp242)
    tmp244 = tl.load(in_ptr0 + (182942528 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp245 = tl.where(tmp4 & tmp6, tmp242, tmp244)
    tmp246 = tl.where(tmp2, tmp3, tmp245)
    tmp247 = tl.load(in_ptr0 + (182942529 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp248 = tl.where(tmp4 & tmp6, tmp245, tmp247)
    tmp249 = tl.where(tmp2, tmp3, tmp248)
    tmp250 = tl.load(in_ptr0 + (260820160 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp251 = tl.where(tmp4 & tmp6, tmp248, tmp250)
    tmp252 = tl.where(tmp2, tmp3, tmp251)
    tmp253 = tl.load(in_ptr0 + (260820161 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp254 = tl.where(tmp4 & tmp6, tmp251, tmp253)
    tmp255 = tl.where(tmp2, tmp3, tmp254)
    tmp256 = tl.load(in_ptr0 + (347679680 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp257 = tl.where(tmp4 & tmp6, tmp254, tmp256)
    tmp258 = tl.where(tmp2, tmp3, tmp257)
    tmp259 = tl.load(in_ptr0 + (347679681 + x1 + 128 * x0), tmp2 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp260 = tl.where(tmp4 & tmp6, tmp257, tmp259)
    tmp261 = tl.where(tmp2, tmp3, tmp260)
    tmp262 = tl.load(in_ptr0 + (4
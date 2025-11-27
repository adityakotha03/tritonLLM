import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_pow_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.044715
    tmp4 = tmp0 * tmp3
    tmp5 = tmp0 * tmp0
    tmp6 = tmp5 * tmp0
    tmp7 = tmp4 * tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tl.store(out_ptr0 + x0, tmp14, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 4096 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4096 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8192 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12288 + x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 >= tmp9
    tmp13 = tl.load(in_ptr0 + (4096 * tmp11 + (-4096 + x2)), tmp12 &
        xmask, other=float('-inf'))
    tmp14 = tl.load(in_ptr0 + (4096 * tmp9 + (-4096 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tl.full([1], 2, tl.int32)
    tmp17 = tmp16 >= tmp9
    tmp18 = tl.load(in_ptr0 + (4096 * tmp16 + (-8192 + x2)), tmp17 &
        xmask, other=float('-inf'))
    tmp19 = tl.load(in_ptr0 + (4096 * tmp9 + (-8192 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = tl.full([1], 3, tl.int32)
    tmp22 = tmp21 >= tmp9
    tmp23 = tl.load(in_ptr0 + (4096 * tmp21 + (-12288 + x2)), tmp22 &
        xmask, other=float('-inf'))
    tmp24 = tl.load(in_ptr0 + (4096 * tmp9 + (-12288 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp26 = tl.full([1], 4, tl.int32)
    tmp27 = tmp26 >= tmp9
    tmp28 = tl.load(in_ptr0 + (4096 * tmp26 + (-16384 + x2)), tmp27 &
        xmask, other=float('-inf'))
    tmp29 = tl.load(in_ptr0 + (4096 * tmp9 + (-16384 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp30 = triton_helpers.maximum(tmp28, tmp29)
    tmp31 = tl.full([1], 5, tl.int32)
    tmp32 = tmp31 >= tmp9
    tmp33 = tl.load(in_ptr0 + (4096 * tmp31 + (-20480 + x2)), tmp32 &
        xmask, other=float('-inf'))
    tmp34 = tl.load(in_ptr0 + (4096 * tmp9 + (-20480 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = tl.full([1], 6, tl.int32)
    tmp37 = tmp36 >= tmp9
    tmp38 = tl.load(in_ptr0 + (4096 * tmp36 + (-24576 + x2)), tmp37 &
        xmask, other=float('-inf'))
    tmp39 = tl.load(in_ptr0 + (4096 * tmp9 + (-24576 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp40 = triton_helpers.maximum(tmp38, tmp39)
    tmp41 = tl.full([1], 7, tl.int32)
    tmp42 = tmp41 >= tmp9
    tmp43 = tl.load(in_ptr0 + (4096 * tmp41 + (-28672 + x2)), tmp42 &
        xmask, other=float('-inf'))
    tmp44 = tl.load(in_ptr0 + (4096 * tmp9 + (-28672 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp45 = triton_helpers.maximum(tmp43, tmp44)
    tmp46 = tl.full([1], 8, tl.int32)
    tmp47 = tmp46 >= tmp9
    tmp48 = tl.load(in_ptr0 + (4096 * tmp46 + (-32768 + x2)), tmp47 &
        xmask, other=float('-inf'))
    tmp49 = tl.load(in_ptr0 + (4096 * tmp9 + (-32768 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp50 = triton_helpers.maximum(tmp48, tmp49)
    tmp51 = tl.full([1], 9, tl.int32)
    tmp52 = tmp51 >= tmp9
    tmp53 = tl.load(in_ptr0 + (4096 * tmp51 + (-36864 + x2)), tmp52 &
        xmask, other=float('-inf'))
    tmp54 = tl.load(in_ptr0 + (4096 * tmp9 + (-36864 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp55 = triton_helpers.maximum(tmp53, tmp54)
    tmp56 = tl.full([1], 10, tl.int32)
    tmp57 = tmp56 >= tmp9
    tmp58 = tl.load(in_ptr0 + (4096 * tmp56 + (-40960 + x2)), tmp57 &
        xmask, other=float('-inf'))
    tmp59 = tl.load(in_ptr0 + (4096 * tmp9 + (-40960 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp60 = triton_helpers.maximum(tmp58, tmp59)
    tmp61 = tl.full([1], 11, tl.int32)
    tmp62 = tmp61 >= tmp9
    tmp63 = tl.load(in_ptr0 + (4096 * tmp61 + (-45056 + x2)), tmp62 &
        xmask, other=float('-inf'))
    tmp64 = tl.load(in_ptr0 + (4096 * tmp9 + (-45056 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp65 = triton_helpers.maximum(tmp63, tmp64)
    tmp66 = tl.full([1], 12, tl.int32)
    tmp67 = tmp66 >= tmp9
    tmp68 = tl.load(in_ptr0 + (4096 * tmp66 + (-49152 + x2)), tmp67 &
        xmask, other=float('-inf'))
    tmp69 = tl.load(in_ptr0 + (4096 * tmp9 + (-49152 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp70 = triton_helpers.maximum(tmp68, tmp69)
    tmp71 = tl.full([1], 13, tl.int32)
    tmp72 = tmp71 >= tmp9
    tmp73 = tl.load(in_ptr0 + (4096 * tmp71 + (-53248 + x2)), tmp72 &
        xmask, other=float('-inf'))
    tmp74 = tl.load(in_ptr0 + (4096 * tmp9 + (-53248 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp75 = triton_helpers.maximum(tmp73, tmp74)
    tmp76 = tl.full([1], 14, tl.int32)
    tmp77 = tmp76 >= tmp9
    tmp78 = tl.load(in_ptr0 + (4096 * tmp76 + (-57344 + x2)), tmp77 &
        xmask, other=float('-inf'))
    tmp79 = tl.load(in_ptr0 + (4096 * tmp9 + (-57344 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp80 = triton_helpers.maximum(tmp78, tmp79)
    tmp81 = tl.full([1], 15, tl.int32)
    tmp82 = tmp81 >= tmp9
    tmp83 = tl.load(in_ptr0 + (4096 * tmp81 + (-61440 + x2)), tmp82 &
        xmask, other=float('-inf'))
    tmp84 = tl.load(in_ptr0 + (4096 * tmp9 + (-61440 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp85 = triton_helpers.maximum(tmp83, tmp84)
    tmp86 = tl.full([1], 16, tl.int32)
    tmp87 = tmp86 >= tmp9
    tmp88 = tl.load(in_ptr0 + (4096 * tmp86 + (-65536 + x2)), tmp87 &
        xmask, other=float('-inf'))
    tmp89 = tl.load(in_ptr0 + (4096 * tmp9 + (-65536 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp90 = triton_helpers.maximum(tmp88, tmp89)
    tmp91 = tl.full([1], 17, tl.int32)
    tmp92 = tmp91 >= tmp9
    tmp93 = tl.load(in_ptr0 + (4096 * tmp91 + (-69632 + x2)), tmp92 &
        xmask, other=float('-inf'))
    tmp94 = tl.load(in_ptr0 + (4096 * tmp9 + (-69632 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp95 = triton_helpers.maximum(tmp93, tmp94)
    tmp96 = tl.full([1], 18, tl.int32)
    tmp97 = tmp96 >= tmp9
    tmp98 = tl.load(in_ptr0 + (4096 * tmp96 + (-73728 + x2)), tmp97 &
        xmask, other=float('-inf'))
    tmp99 = tl.load(in_ptr0 + (4096 * tmp9 + (-73728 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = tl.full([1], 19, tl.int32)
    tmp102 = tmp101 >= tmp9
    tmp103 = tl.load(in_ptr0 + (4096 * tmp101 + (-77824 + x2)), tmp102 &
        xmask, other=float('-inf'))
    tmp104 = tl.load(in_ptr0 + (4096 * tmp9 + (-77824 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp105 = triton_helpers.maximum(tmp103, tmp104)
    tmp106 = tl.full([1], 20, tl.int32)
    tmp107 = tmp106 >= tmp9
    tmp108 = tl.load(in_ptr0 + (4096 * tmp106 + (-81920 + x2)), tmp107 &
        xmask, other=float('-inf'))
    tmp109 = tl.load(in_ptr0 + (4096 * tmp9 + (-81920 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp110 = triton_helpers.maximum(tmp108, tmp109)
    tmp111 = tl.full([1], 21, tl.int32)
    tmp112 = tmp111 >= tmp9
    tmp113 = tl.load(in_ptr0 + (4096 * tmp111 + (-86016 + x2)), tmp112 &
        xmask, other=float('-inf'))
    tmp114 = tl.load(in_ptr0 + (4096 * tmp9 + (-86016 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp115 = triton_helpers.maximum(tmp113, tmp114)
    tmp116 = tl.full([1], 22, tl.int32)
    tmp117 = tmp116 >= tmp9
    tmp118 = tl.load(in_ptr0 + (4096 * tmp116 + (-90112 + x2)), tmp117 &
        xmask, other=float('-inf'))
    tmp119 = tl.load(in_ptr0 + (4096 * tmp9 + (-90112 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp120 = triton_helpers.maximum(tmp118, tmp119)
    tmp121 = tl.full([1], 23, tl.int32)
    tmp122 = tmp121 >= tmp9
    tmp123 = tl.load(in_ptr0 + (4096 * tmp121 + (-94208 + x2)), tmp122 &
        xmask, other=float('-inf'))
    tmp124 = tl.load(in_ptr0 + (4096 * tmp9 + (-94208 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp125 = triton_helpers.maximum(tmp123, tmp124)
    tmp126 = tl.full([1], 24, tl.int32)
    tmp127 = tmp126 >= tmp9
    tmp128 = tl.load(in_ptr0 + (4096 * tmp126 + (-98304 + x2)), tmp127 &
        xmask, other=float('-inf'))
    tmp129 = tl.load(in_ptr0 + (4096 * tmp9 + (-98304 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp130 = triton_helpers.maximum(tmp128, tmp129)
    tmp131 = tl.full([1], 25, tl.int32)
    tmp132 = tmp131 >= tmp9
    tmp133 = tl.load(in_ptr0 + (4096 * tmp131 + (-102400 + x2)), tmp132 &
        xmask, other=float('-inf'))
    tmp134 = tl.load(in_ptr0 + (4096 * tmp9 + (-102400 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp135 = triton_helpers.maximum(tmp133, tmp134)
    tmp136 = tl.full([1], 26, tl.int32)
    tmp137 = tmp136 >= tmp9
    tmp138 = tl.load(in_ptr0 + (4096 * tmp136 + (-106496 + x2)), tmp137 &
        xmask, other=float('-inf'))
    tmp139 = tl.load(in_ptr0 + (4096 * tmp9 + (-106496 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp140 = triton_helpers.maximum(tmp138, tmp139)
    tmp141 = tl.full([1], 27, tl.int32)
    tmp142 = tmp141 >= tmp9
    tmp143 = tl.load(in_ptr0 + (4096 * tmp141 + (-110592 + x2)), tmp142 &
        xmask, other=float('-inf'))
    tmp144 = tl.load(in_ptr0 + (4096 * tmp9 + (-110592 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp145 = triton_helpers.maximum(tmp143, tmp144)
    tmp146 = tl.full([1], 28, tl.int32)
    tmp147 = tmp146 >= tmp9
    tmp148 = tl.load(in_ptr0 + (4096 * tmp146 + (-114688 + x2)), tmp147 &
        xmask, other=float('-inf'))
    tmp149 = tl.load(in_ptr0 + (4096 * tmp9 + (-114688 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp150 = triton_helpers.maximum(tmp148, tmp149)
    tmp151 = tl.full([1], 29, tl.int32)
    tmp152 = tmp151 >= tmp9
    tmp153 = tl.load(in_ptr0 + (4096 * tmp151 + (-118784 + x2)), tmp152 &
        xmask, other=float('-inf'))
    tmp154 = tl.load(in_ptr0 + (4096 * tmp9 + (-118784 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp155 = triton_helpers.maximum(tmp153, tmp154)
    tmp156 = tl.full([1], 30, tl.int32)
    tmp157 = tmp156 >= tmp9
    tmp158 = tl.load(in_ptr0 + (4096 * tmp156 + (-122880 + x2)), tmp157 &
        xmask, other=float('-inf'))
    tmp159 = tl.load(in_ptr0 + (4096 * tmp9 + (-122880 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp160 = triton_helpers.maximum(tmp158, tmp159)
    tmp161 = tl.full([1], 31, tl.int32)
    tmp162 = tmp161 >= tmp9
    tmp163 = tl.load(in_ptr0 + (4096 * tmp161 + (-126976 + x2)), tmp162 &
        xmask, other=float('-inf'))
    tmp164 = tl.load(in_ptr0 + (4096 * tmp9 + (-126976 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp165 = triton_helpers.maximum(tmp163, tmp164)
    tmp166 = tl.full([1], 32, tl.int32)
    tmp167 = tmp166 >= tmp9
    tmp168 = tl.load(in_ptr0 + (4096 * tmp166 + (-131072 + x2)), tmp167 &
        xmask, other=float('-inf'))
    tmp169 = tl.load(in_ptr0 + (4096 * tmp9 + (-131072 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp170 = triton_helpers.maximum(tmp168, tmp169)
    tmp171 = tl.full([1], 33, tl.int32)
    tmp172 = tmp171 >= tmp9
    tmp173 = tl.load(in_ptr0 + (4096 * tmp171 + (-135168 + x2)), tmp172 &
        xmask, other=float('-inf'))
    tmp174 = tl.load(in_ptr0 + (4096 * tmp9 + (-135168 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp175 = triton_helpers.maximum(tmp173, tmp174)
    tmp176 = tl.full([1], 34, tl.int32)
    tmp177 = tmp176 >= tmp9
    tmp178 = tl.load(in_ptr0 + (4096 * tmp176 + (-139264 + x2)), tmp177 &
        xmask, other=float('-inf'))
    tmp179 = tl.load(in_ptr0 + (4096 * tmp9 + (-139264 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp180 = triton_helpers.maximum(tmp178, tmp179)
    tmp181 = tl.full([1], 35, tl.int32)
    tmp182 = tmp181 >= tmp9
    tmp183 = tl.load(in_ptr0 + (4096 * tmp181 + (-143360 + x2)), tmp182 &
        xmask, other=float('-inf'))
    tmp184 = tl.load(in_ptr0 + (4096 * tmp9 + (-143360 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp185 = triton_helpers.maximum(tmp183, tmp184)
    tmp186 = tl.full([1], 36, tl.int32)
    tmp187 = tmp186 >= tmp9
    tmp188 = tl.load(in_ptr0 + (4096 * tmp186 + (-147456 + x2)), tmp187 &
        xmask, other=float('-inf'))
    tmp189 = tl.load(in_ptr0 + (4096 * tmp9 + (-147456 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp190 = triton_helpers.maximum(tmp188, tmp189)
    tmp191 = tl.full([1], 37, tl.int32)
    tmp192 = tmp191 >= tmp9
    tmp193 = tl.load(in_ptr0 + (4096 * tmp191 + (-151552 + x2)), tmp192 &
        xmask, other=float('-inf'))
    tmp194 = tl.load(in_ptr0 + (4096 * tmp9 + (-151552 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp195 = triton_helpers.maximum(tmp193, tmp194)
    tmp196 = tl.full([1], 38, tl.int32)
    tmp197 = tmp196 >= tmp9
    tmp198 = tl.load(in_ptr0 + (4096 * tmp196 + (-155648 + x2)), tmp197 &
        xmask, other=float('-inf'))
    tmp199 = tl.load(in_ptr0 + (4096 * tmp9 + (-155648 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp200 = triton_helpers.maximum(tmp198, tmp199)
    tmp201 = tl.full([1], 39, tl.int32)
    tmp202 = tmp201 >= tmp9
    tmp203 = tl.load(in_ptr0 + (4096 * tmp201 + (-159744 + x2)), tmp202 &
        xmask, other=float('-inf'))
    tmp204 = tl.load(in_ptr0 + (4096 * tmp9 + (-159744 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp205 = triton_helpers.maximum(tmp203, tmp204)
    tmp206 = tl.full([1], 40, tl.int32)
    tmp207 = tmp206 >= tmp9
    tmp208 = tl.load(in_ptr0 + (4096 * tmp206 + (-163840 + x2)), tmp207 &
        xmask, other=float('-inf'))
    tmp209 = tl.load(in_ptr0 + (4096 * tmp9 + (-163840 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp210 = triton_helpers.maximum(tmp208, tmp209)
    tmp211 = tl.full([1], 41, tl.int32)
    tmp212 = tmp211 >= tmp9
    tmp213 = tl.load(in_ptr0 + (4096 * tmp211 + (-167936 + x2)), tmp212 &
        xmask, other=float('-inf'))
    tmp214 = tl.load(in_ptr0 + (4096 * tmp9 + (-167936 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp215 = triton_helpers.maximum(tmp213, tmp214)
    tmp216 = tl.full([1], 42, tl.int32)
    tmp217 = tmp216 >= tmp9
    tmp218 = tl.load(in_ptr0 + (4096 * tmp216 + (-172032 + x2)), tmp217 &
        xmask, other=float('-inf'))
    tmp219 = tl.load(in_ptr0 + (4096 * tmp9 + (-172032 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp220 = triton_helpers.maximum(tmp218, tmp219)
    tmp221 = tl.full([1], 43, tl.int32)
    tmp222 = tmp221 >= tmp9
    tmp223 = tl.load(in_ptr0 + (4096 * tmp221 + (-176128 + x2)), tmp222 &
        xmask, other=float('-inf'))
    tmp224 = tl.load(in_ptr0 + (4096 * tmp9 + (-176128 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp225 = triton_helpers.maximum(tmp223, tmp224)
    tmp226 = tl.full([1], 44, tl.int32)
    tmp227 = tmp226 >= tmp9
    tmp228 = tl.load(in_ptr0 + (4096 * tmp226 + (-180224 + x2)), tmp227 &
        xmask, other=float('-inf'))
    tmp229 = tl.load(in_ptr0 + (4096 * tmp9 + (-180224 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp230 = triton_helpers.maximum(tmp228, tmp229)
    tmp231 = tl.full([1], 45, tl.int32)
    tmp232 = tmp231 >= tmp9
    tmp233 = tl.load(in_ptr0 + (4096 * tmp231 + (-184320 + x2)), tmp232 &
        xmask, other=float('-inf'))
    tmp234 = tl.load(in_ptr0 + (4096 * tmp9 + (-184320 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp235 = triton_helpers.maximum(tmp233, tmp234)
    tmp236 = tl.full([1], 46, tl.int32)
    tmp237 = tmp236 >= tmp9
    tmp238 = tl.load(in_ptr0 + (4096 * tmp236 + (-188416 + x2)), tmp237 &
        xmask, other=float('-inf'))
    tmp239 = tl.load(in_ptr0 + (4096 * tmp9 + (-188416 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp240 = triton_helpers.maximum(tmp238, tmp239)
    tmp241 = tl.full([1], 47, tl.int32)
    tmp242 = tmp241 >= tmp9
    tmp243 = tl.load(in_ptr0 + (4096 * tmp241 + (-192512 + x2)), tmp242 &
        xmask, other=float('-inf'))
    tmp244 = tl.load(in_ptr0 + (4096 * tmp9 + (-192512 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp245 = triton_helpers.maximum(tmp243, tmp244)
    tmp246 = tl.full([1], 48, tl.int32)
    tmp247 = tmp246 >= tmp9
    tmp248 = tl.load(in_ptr0 + (4096 * tmp246 + (-196608 + x2)), tmp247 &
        xmask, other=float('-inf'))
    tmp249 = tl.load(in_ptr0 + (4096 * tmp9 + (-196608 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp250 = triton_helpers.maximum(tmp248, tmp249)
    tmp251 = tl.full([1], 49, tl.int32)
    tmp252 = tmp251 >= tmp9
    tmp253 = tl.load(in_ptr0 + (4096 * tmp251 + (-200704 + x2)), tmp252 &
        xmask, other=float('-inf'))
    tmp254 = tl.load(in_ptr0 + (4096 * tmp9 + (-200704 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp255 = triton_helpers.maximum(tmp253, tmp254)
    tmp256 = tl.full([1], 50, tl.int32)
    tmp257 = tmp256 >= tmp9
    tmp258 = tl.load(in_ptr0 + (4096 * tmp256 + (-204800 + x2)), tmp257 &
        xmask, other=float('-inf'))
    tmp259 = tl.load(in_ptr0 + (4096 * tmp9 + (-204800 + x2)), tmp9 >= tmp9 &
        xmask, other=float('-inf'))
    tmp260 = triton_helpers.maximum(tmp258, tmp259)
    tmp261 = tl.full([1], 51, tl.int32)
    tmp262 = tmp261 >= tmp9
    tmp263 = tl.load(in_ptr0 + (4096 * tmp261 + (-20
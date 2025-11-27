import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16 * x0 + 128 * x1), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 + tmp0
    tmp3 = tl.full([1], 128, tl.int32)
    tmp4 = tmp1 < tmp3
    tmp5 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 128), tmp4 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp6 = tmp1 + tmp5
    tmp7 = tl.full([1], 256, tl.int32)
    tmp8 = tmp1 < tmp7
    tmp9 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 256), tmp8 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tmp1 + tmp9
    tmp11 = tl.full([1], 384, tl.int32)
    tmp12 = tmp1 < tmp11
    tmp13 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 384), tmp12 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp14 = tmp1 + tmp13
    tmp15 = tl.full([1], 512, tl.int32)
    tmp16 = tmp1 < tmp15
    tmp17 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 512), tmp16 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp18 = tmp1 + tmp17
    tmp19 = tl.full([1], 640, tl.int32)
    tmp20 = tmp1 < tmp19
    tmp21 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 640), tmp20 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp22 = tmp1 + tmp21
    tmp23 = tl.full([1], 768, tl.int32)
    tmp24 = tmp1 < tmp23
    tmp25 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 768), tmp24 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp26 = tmp1 + tmp25
    tmp27 = tl.full([1], 896, tl.int32)
    tmp28 = tmp1 < tmp27
    tmp29 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 896), tmp28 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp30 = tmp1 + tmp29
    tmp31 = tl.full([1], 1024, tl.int32)
    tmp32 = tmp1 < tmp31
    tmp33 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1024), tmp32 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp34 = tmp1 + tmp33
    tmp35 = tl.full([1], 1152, tl.int32)
    tmp36 = tmp1 < tmp35
    tmp37 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1152), tmp36 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp38 = tmp1 + tmp37
    tmp39 = tl.full([1], 1280, tl.int32)
    tmp40 = tmp1 < tmp39
    tmp41 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1280), tmp40 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp42 = tmp1 + tmp41
    tmp43 = tl.full([1], 1408, tl.int32)
    tmp44 = tmp1 < tmp43
    tmp45 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1408), tmp44 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp46 = tmp1 + tmp45
    tmp47 = tl.full([1], 1536, tl.int32)
    tmp48 = tmp1 < tmp47
    tmp49 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1536), tmp48 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp50 = tmp1 + tmp49
    tmp51 = tl.full([1], 1664, tl.int32)
    tmp52 = tmp1 < tmp51
    tmp53 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1664), tmp52 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp54 = tmp1 + tmp53
    tmp55 = tl.full([1], 1792, tl.int32)
    tmp56 = tmp1 < tmp55
    tmp57 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1792), tmp56 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp58 = tmp1 + tmp57
    tmp59 = tl.full([1], 1920, tl.int32)
    tmp60 = tmp1 < tmp59
    tmp61 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 1920), tmp60 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp62 = tmp1 + tmp61
    tmp63 = tl.full([1], 2048, tl.int32)
    tmp64 = tmp1 < tmp63
    tmp65 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2048), tmp64 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp66 = tmp1 + tmp65
    tmp67 = tl.full([1], 2176, tl.int32)
    tmp68 = tmp1 < tmp67
    tmp69 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2176), tmp68 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp70 = tmp1 + tmp69
    tmp71 = tl.full([1], 2304, tl.int32)
    tmp72 = tmp1 < tmp71
    tmp73 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2304), tmp72 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp74 = tmp1 + tmp73
    tmp75 = tl.full([1], 2432, tl.int32)
    tmp76 = tmp1 < tmp75
    tmp77 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2432), tmp76 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp78 = tmp1 + tmp77
    tmp79 = tl.full([1], 2560, tl.int32)
    tmp80 = tmp1 < tmp79
    tmp81 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2560), tmp80 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp82 = tmp1 + tmp81
    tmp83 = tl.full([1], 2688, tl.int32)
    tmp84 = tmp1 < tmp83
    tmp85 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2688), tmp84 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp86 = tmp1 + tmp85
    tmp87 = tl.full([1], 2816, tl.int32)
    tmp88 = tmp1 < tmp87
    tmp89 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2816), tmp88 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp90 = tmp1 + tmp89
    tmp91 = tl.full([1], 2944, tl.int32)
    tmp92 = tmp1 < tmp91
    tmp93 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 2944), tmp92 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp94 = tmp1 + tmp93
    tmp95 = tl.full([1], 3072, tl.int32)
    tmp96 = tmp1 < tmp95
    tmp97 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3072), tmp96 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp98 = tmp1 + tmp97
    tmp99 = tl.full([1], 3200, tl.int32)
    tmp100 = tmp1 < tmp99
    tmp101 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3200), tmp100 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp102 = tmp1 + tmp101
    tmp103 = tl.full([1], 3328, tl.int32)
    tmp104 = tmp1 < tmp103
    tmp105 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3328), tmp104 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp106 = tmp1 + tmp105
    tmp107 = tl.full([1], 3456, tl.int32)
    tmp108 = tmp1 < tmp107
    tmp109 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3456), tmp108 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp110 = tmp1 + tmp109
    tmp111 = tl.full([1], 3584, tl.int32)
    tmp112 = tmp1 < tmp111
    tmp113 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3584), tmp112 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp114 = tmp1 + tmp113
    tmp115 = tl.full([1], 3712, tl.int32)
    tmp116 = tmp1 < tmp115
    tmp117 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3712), tmp116 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp118 = tmp1 + tmp117
    tmp119 = tl.full([1], 3840, tl.int32)
    tmp120 = tmp1 < tmp119
    tmp121 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3840), tmp120 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp122 = tmp1 + tmp121
    tmp123 = tl.full([1], 3968, tl.int32)
    tmp124 = tmp1 < tmp123
    tmp125 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 3968), tmp124 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp126 = tmp1 + tmp125
    tmp127 = tl.full([1], 4096, tl.int32)
    tmp128 = tmp1 < tmp127
    tmp129 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4096), tmp128 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp130 = tmp1 + tmp129
    tmp131 = tl.full([1], 4224, tl.int32)
    tmp132 = tmp1 < tmp131
    tmp133 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4224), tmp132 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp134 = tmp1 + tmp133
    tmp135 = tl.full([1], 4352, tl.int32)
    tmp136 = tmp1 < tmp135
    tmp137 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4352), tmp136 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp138 = tmp1 + tmp137
    tmp139 = tl.full([1], 4480, tl.int32)
    tmp140 = tmp1 < tmp139
    tmp141 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4480), tmp140 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp142 = tmp1 + tmp141
    tmp143 = tl.full([1], 4608, tl.int32)
    tmp144 = tmp1 < tmp143
    tmp145 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4608), tmp144 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp146 = tmp1 + tmp145
    tmp147 = tl.full([1], 4736, tl.int32)
    tmp148 = tmp1 < tmp147
    tmp149 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4736), tmp148 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp150 = tmp1 + tmp149
    tmp151 = tl.full([1], 4864, tl.int32)
    tmp152 = tmp1 < tmp151
    tmp153 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 4864), tmp152 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp154 = tmp1 + tmp153
    tmp155 = tl.full([1], 5000, tl.int32)
    tmp156 = tmp1 < tmp155
    tmp157 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5000), tmp156 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp158 = tmp1 + tmp157
    tmp159 = tl.full([1], 5128, tl.int32)
    tmp160 = tmp1 < tmp159
    tmp161 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5128), tmp160 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp162 = tmp1 + tmp161
    tmp163 = tl.full([1], 5256, tl.int32)
    tmp164 = tmp1 < tmp163
    tmp165 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5256), tmp164 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp166 = tmp1 + tmp165
    tmp167 = tl.full([1], 5384, tl.int32)
    tmp168 = tmp1 < tmp167
    tmp169 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5384), tmp168 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp170 = tmp1 + tmp169
    tmp171 = tl.full([1], 5512, tl.int32)
    tmp172 = tmp1 < tmp171
    tmp173 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5512), tmp172 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp174 = tmp1 + tmp173
    tmp175 = tl.full([1], 5640, tl.int32)
    tmp176 = tmp1 < tmp175
    tmp177 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5640), tmp176 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp178 = tmp1 + tmp177
    tmp179 = tl.full([1], 5768, tl.int32)
    tmp180 = tmp1 < tmp179
    tmp181 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5768), tmp180 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp182 = tmp1 + tmp181
    tmp183 = tl.full([1], 5896, tl.int32)
    tmp184 = tmp1 < tmp183
    tmp185 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 5896), tmp184 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp186 = tmp1 + tmp185
    tmp187 = tl.full([1], 6024, tl.int32)
    tmp188 = tmp1 < tmp187
    tmp189 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6024), tmp188 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp190 = tmp1 + tmp189
    tmp191 = tl.full([1], 6152, tl.int32)
    tmp192 = tmp1 < tmp191
    tmp193 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6152), tmp192 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp194 = tmp1 + tmp193
    tmp195 = tl.full([1], 6280, tl.int32)
    tmp196 = tmp1 < tmp195
    tmp197 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6280), tmp196 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp198 = tmp1 + tmp197
    tmp199 = tl.full([1], 6408, tl.int32)
    tmp200 = tmp1 < tmp199
    tmp201 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6408), tmp200 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp202 = tmp1 + tmp201
    tmp203 = tl.full([1], 6536, tl.int32)
    tmp204 = tmp1 < tmp203
    tmp205 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6536), tmp204 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp206 = tmp1 + tmp205
    tmp207 = tl.full([1], 6664, tl.int32)
    tmp208 = tmp1 < tmp207
    tmp209 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6664), tmp208 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp210 = tmp1 + tmp209
    tmp211 = tl.full([1], 6792, tl.int32)
    tmp212 = tmp1 < tmp211
    tmp213 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6792), tmp212 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp214 = tmp1 + tmp213
    tmp215 = tl.full([1], 6920, tl.int32)
    tmp216 = tmp1 < tmp215
    tmp217 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 6920), tmp216 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp218 = tmp1 + tmp217
    tmp219 = tl.full([1], 7048, tl.int32)
    tmp220 = tmp1 < tmp219
    tmp221 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7048), tmp220 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp222 = tmp1 + tmp221
    tmp223 = tl.full([1], 7176, tl.int32)
    tmp224 = tmp1 < tmp223
    tmp225 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7176), tmp224 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp226 = tmp1 + tmp225
    tmp227 = tl.full([1], 7304, tl.int32)
    tmp228 = tmp1 < tmp227
    tmp229 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7304), tmp228 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp230 = tmp1 + tmp229
    tmp231 = tl.full([1], 7432, tl.int32)
    tmp232 = tmp1 < tmp231
    tmp233 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7432), tmp232 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp234 = tmp1 + tmp233
    tmp235 = tl.full([1], 7560, tl.int32)
    tmp236 = tmp1 < tmp235
    tmp237 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7560), tmp236 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp238 = tmp1 + tmp237
    tmp239 = tl.full([1], 7688, tl.int32)
    tmp240 = tmp1 < tmp239
    tmp241 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7688), tmp240 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp242 = tmp1 + tmp241
    tmp243 = tl.full([1], 7816, tl.int32)
    tmp244 = tmp1 < tmp243
    tmp245 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7816), tmp244 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp246 = tmp1 + tmp245
    tmp247 = tl.full([1], 7944, tl.int32)
    tmp248 = tmp1 < tmp247
    tmp249 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 7944), tmp248 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp250 = tmp1 + tmp249
    tmp251 = tl.full([1], 8072, tl.int32)
    tmp252 = tmp1 < tmp251
    tmp253 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8072), tmp252 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp254 = tmp1 + tmp253
    tmp255 = tl.full([1], 8200, tl.int32)
    tmp256 = tmp1 < tmp255
    tmp257 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8200), tmp256 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp258 = tmp1 + tmp257
    tmp259 = tl.full([1], 8328, tl.int32)
    tmp260 = tmp1 < tmp259
    tmp261 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8328), tmp260 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp262 = tmp1 + tmp261
    tmp263 = tl.full([1], 8456, tl.int32)
    tmp264 = tmp1 < tmp263
    tmp265 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8456), tmp264 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp266 = tmp1 + tmp265
    tmp267 = tl.full([1], 8584, tl.int32)
    tmp268 = tmp1 < tmp267
    tmp269 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8584), tmp268 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp270 = tmp1 + tmp269
    tmp271 = tl.full([1], 8712, tl.int32)
    tmp272 = tmp1 < tmp271
    tmp273 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8712), tmp272 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp274 = tmp1 + tmp273
    tmp275 = tl.full([1], 8840, tl.int32)
    tmp276 = tmp1 < tmp275
    tmp277 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8840), tmp276 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp278 = tmp1 + tmp277
    tmp279 = tl.full([1], 8968, tl.int32)
    tmp280 = tmp1 < tmp279
    tmp281 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 8968), tmp280 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp282 = tmp1 + tmp281
    tmp283 = tl.full([1], 9096, tl.int32)
    tmp284 = tmp1 < tmp283
    tmp285 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9096), tmp284 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp286 = tmp1 + tmp285
    tmp287 = tl.full([1], 9224, tl.int32)
    tmp288 = tmp1 < tmp287
    tmp289 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9224), tmp288 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp290 = tmp1 + tmp289
    tmp291 = tl.full([1], 9352, tl.int32)
    tmp292 = tmp1 < tmp291
    tmp293 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9352), tmp292 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp294 = tmp1 + tmp293
    tmp295 = tl.full([1], 9480, tl.int32)
    tmp296 = tmp1 < tmp295
    tmp297 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9480), tmp296 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp298 = tmp1 + tmp297
    tmp299 = tl.full([1], 9608, tl.int32)
    tmp300 = tmp1 < tmp299
    tmp301 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9608), tmp300 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp302 = tmp1 + tmp301
    tmp303 = tl.full([1], 9736, tl.int32)
    tmp304 = tmp1 < tmp303
    tmp305 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9736), tmp304 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp306 = tmp1 + tmp305
    tmp307 = tl.full([1], 9864, tl.int32)
    tmp308 = tmp1 < tmp307
    tmp309 = tl.load(in_ptr0 + (16 * x0 + 128 * x1 + 9864), tmp308 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp310 = tmp1 + tmp30
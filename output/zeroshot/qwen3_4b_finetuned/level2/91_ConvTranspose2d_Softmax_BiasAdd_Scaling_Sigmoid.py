import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_convolution_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 16 % 16
    x0 = xindex % 16
    x2 = xindex // 256
    x4 = xindex
    tmp0 = -1 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = -1 + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp10 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp16 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 - tmp16
    tmp19 = tl.load(in_ptr1 + x1, tmp16 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp7 & tmp8
    tmp22 = tmp5 & tmp21
    tmp23 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp22 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tmp23 - tmp22
    tmp25 = tl.load(in_ptr1 + x0, tmp22 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp26 - tmp16
    tmp28 = tmp15 & tmp8
    tmp29 = tmp5 & tmp28
    tmp30 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp29 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 - tmp29
    tmp32 = tl.load(in_ptr1 + (-64 + x0 + x1), tmp29 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 - tmp26
    tmp35 = tmp21 & tmp8
    tmp36 = tmp5 & tmp35
    tmp37 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp36 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 - tmp36
    tmp39 = tl.load(in_ptr1 + (-65 + x0 + x1), tmp36 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40 - tmp34
    tmp42 = tmp28 & tmp8
    tmp43 = tmp5 & tmp42
    tmp44 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp43 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 - tmp43
    tmp46 = tl.load(in_ptr1 + (-65 + x0 + x1), tmp43 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp47 - tmp41
    tmp49 = tmp43 & tmp8
    tmp50 = tmp5 & tmp49
    tmp51 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp50 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 - tmp50
    tmp53 = tl.load(in_ptr1 + (-66 + x0 + x1), tmp50 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp54 = tmp52 + tmp53
    tmp55 = tmp54 - tmp48
    tmp56 = tmp42 & tmp8
    tmp57 = tmp5 & tmp56
    tmp58 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp57 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 - tmp57
    tmp60 = tl.load(in_ptr1 + (-66 + x0 + x1), tmp57 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 + tmp60
    tmp62 = tmp61 - tmp55
    tmp63 = tmp57 & tmp8
    tmp64 = tmp5 & tmp63
    tmp65 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp64 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp65 - tmp64
    tmp67 = tl.load(in_ptr1 + (-67 + x0 + x1), tmp64 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tmp68 - tmp62
    tmp70 = tmp63 & tmp8
    tmp71 = tmp5 & tmp70
    tmp72 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp71 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp72 - tmp71
    tmp74 = tl.load(in_ptr1 + (-67 + x0 + x1), tmp71 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp75 = tmp73 + tmp74
    tmp76 = tmp75 - tmp69
    tmp77 = tmp71 & tmp8
    tmp78 = tmp5 & tmp77
    tmp79 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp78 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp79 - tmp78
    tmp81 = tl.load(in_ptr1 + (-68 + x0 + x1), tmp78 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp82 = tmp80 + tmp81
    tmp83 = tmp82 - tmp76
    tmp84 = tmp77 & tmp8
    tmp85 = tmp5 & tmp84
    tmp86 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp85 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 - tmp85
    tmp88 = tl.load(in_ptr1 + (-68 + x0 + x1), tmp85 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tmp89 - tmp83
    tmp91 = tmp85 & tmp8
    tmp92 = tmp5 & tmp91
    tmp93 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp92 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tmp93 - tmp92
    tmp95 = tl.load(in_ptr1 + (-69 + x0 + x1), tmp92 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 + tmp95
    tmp97 = tmp96 - tmp90
    tmp98 = tmp92 & tmp8
    tmp99 = tmp5 & tmp98
    tmp100 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp99 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp100 - tmp99
    tmp102 = tl.load(in_ptr1 + (-69 + x0 + x1), tmp99 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp103 = tmp101 + tmp102
    tmp104 = tmp103 - tmp97
    tmp105 = tmp99 & tmp8
    tmp106 = tmp5 & tmp105
    tmp107 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp106 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp107 - tmp106
    tmp109 = tl.load(in_ptr1 + (-70 + x0 + x1), tmp106 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp110 = tmp108 + tmp109
    tmp111 = tmp110 - tmp104
    tmp112 = tmp106 & tmp8
    tmp113 = tmp5 & tmp112
    tmp114 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp113 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp115 = tmp114 - tmp113
    tmp116 = tl.load(in_ptr1 + (-70 + x0 + x1), tmp113 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp117 = tmp115 + tmp116
    tmp118 = tmp117 - tmp111
    tmp119 = tmp113 & tmp8
    tmp120 = tmp5 & tmp119
    tmp121 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp120 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp121 - tmp120
    tmp123 = tl.load(in_ptr1 + (-71 + x0 + x1), tmp120 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = tmp124 - tmp118
    tmp126 = tmp120 & tmp8
    tmp127 = tmp5 & tmp126
    tmp128 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp127 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp129 = tmp128 - tmp127
    tmp130 = tl.load(in_ptr1 + (-71 + x0 + x1), tmp127 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp131 = tmp129 + tmp130
    tmp132 = tmp131 - tmp125
    tmp133 = tmp127 & tmp8
    tmp134 = tmp5 & tmp133
    tmp135 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp134 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tmp135 - tmp134
    tmp137 = tl.load(in_ptr1 + (-72 + x0 + x1), tmp134 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp138 = tmp136 + tmp137
    tmp139 = tmp138 - tmp132
    tmp140 = tmp134 & tmp8
    tmp141 = tmp5 & tmp140
    tmp142 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp141 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp143 = tmp142 - tmp141
    tmp144 = tl.load(in_ptr1 + (-72 + x0 + x1), tmp141 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp145 = tmp143 + tmp144
    tmp146 = tmp145 - tmp139
    tmp147 = tmp141 & tmp8
    tmp148 = tmp5 & tmp147
    tmp149 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp148 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp149 - tmp148
    tmp151 = tl.load(in_ptr1 + (-73 + x0 + x1), tmp148 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp152 = tmp150 + tmp151
    tmp153 = tmp152 - tmp146
    tmp154 = tmp148 & tmp8
    tmp155 = tmp5 & tmp154
    tmp156 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp155 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp157 = tmp156 - tmp155
    tmp158 = tl.load(in_ptr1 + (-73 + x0 + x1), tmp155 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp159 = tmp157 + tmp158
    tmp160 = tmp159 - tmp153
    tmp161 = tmp155 & tmp8
    tmp162 = tmp5 & tmp161
    tmp163 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp162 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp164 = tmp163 - tmp162
    tmp165 = tl.load(in_ptr1 + (-74 + x0 + x1), tmp162 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp166 = tmp164 + tmp165
    tmp167 = tmp166 - tmp160
    tmp168 = tmp162 & tmp8
    tmp169 = tmp5 & tmp168
    tmp170 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp169 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp171 = tmp170 - tmp169
    tmp172 = tl.load(in_ptr1 + (-74 + x0 + x1), tmp169 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp173 = tmp171 + tmp172
    tmp174 = tmp173 - tmp167
    tmp175 = tmp169 & tmp8
    tmp176 = tmp5 & tmp175
    tmp177 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp176 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp178 = tmp177 - tmp176
    tmp179 = tl.load(in_ptr1 + (-75 + x0 + x1), tmp176 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp180 = tmp178 + tmp179
    tmp181 = tmp180 - tmp174
    tmp182 = tmp176 & tmp8
    tmp183 = tmp5 & tmp182
    tmp184 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp183 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp185 = tmp184 - tmp183
    tmp186 = tl.load(in_ptr1 + (-75 + x0 + x1), tmp183 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp187 = tmp185 + tmp186
    tmp188 = tmp187 - tmp181
    tmp189 = tmp183 & tmp8
    tmp190 = tmp5 & tmp189
    tmp191 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp190 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tmp191 - tmp190
    tmp193 = tl.load(in_ptr1 + (-76 + x0 + x1), tmp190 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp194 = tmp192 + tmp193
    tmp195 = tmp194 - tmp188
    tmp196 = tmp190 & tmp8
    tmp197 = tmp5 & tmp196
    tmp198 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp197 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp198 - tmp197
    tmp200 = tl.load(in_ptr1 + (-76 + x0 + x1), tmp197 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = tmp201 - tmp195
    tmp203 = tmp197 & tmp8
    tmp204 = tmp5 & tmp203
    tmp205 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp204 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tmp205 - tmp204
    tmp207 = tl.load(in_ptr1 + (-77 + x0 + x1), tmp204 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp208 = tmp206 + tmp207
    tmp209 = tmp208 - tmp202
    tmp210 = tmp204 & tmp8
    tmp211 = tmp5 & tmp210
    tmp212 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp211 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp213 = tmp212 - tmp211
    tmp214 = tl.load(in_ptr1 + (-77 + x0 + x1), tmp211 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp215 = tmp213 + tmp214
    tmp216 = tmp215 - tmp209
    tmp217 = tmp211 & tmp8
    tmp218 = tmp5 & tmp217
    tmp219 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp218 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp220 = tmp219 - tmp218
    tmp221 = tl.load(in_ptr1 + (-78 + x0 + x1), tmp218 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp222 = tmp220 + tmp221
    tmp223 = tmp222 - tmp216
    tmp224 = tmp218 & tmp8
    tmp225 = tmp5 & tmp224
    tmp226 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp225 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp227 = tmp226 - tmp225
    tmp228 = tl.load(in_ptr1 + (-78 + x0 + x1), tmp225 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp229 = tmp227 + tmp228
    tmp230 = tmp229 - tmp223
    tmp231 = tmp225 & tmp8
    tmp232 = tmp5 & tmp231
    tmp233 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp232 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp234 = tmp233 - tmp232
    tmp235 = tl.load(in_ptr1 + (-79 + x0 + x1), tmp232 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp236 = tmp234 + tmp235
    tmp237 = tmp236 - tmp230
    tmp238 = tmp232 & tmp8
    tmp239 = tmp5 & tmp238
    tmp240 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp239 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp241 = tmp240 - tmp239
    tmp242 = tl.load(in_ptr1 + (-79 + x0 + x1), tmp239 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp243 = tmp241 + tmp242
    tmp244 = tmp243 - tmp237
    tmp245 = tmp239 & tmp8
    tmp246 = tmp5 & tmp245
    tmp247 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp246 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp248 = tmp247 - tmp246
    tmp249 = tl.load(in_ptr1 + (-80 + x0 + x1), tmp246 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp250 = tmp248 + tmp249
    tmp251 = tmp250 - tmp244
    tmp252 = tmp246 & tmp8
    tmp253 = tmp5 & tmp252
    tmp254 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp253 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp255 = tmp254 - tmp253
    tmp256 = tl.load(in_ptr1 + (-80 + x0 + x1), tmp253 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp257 = tmp255 + tmp256
    tmp258 = tmp257 - tmp251
    tmp259 = tmp253 & tmp8
    tmp260 = tmp5 & tmp259
    tmp261 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp260 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp262 = tmp261 - tmp260
    tmp263 = tl.load(in_ptr1 + (-81 + x0 + x1), tmp260 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp264 = tmp262 + tmp263
    tmp265 = tmp264 - tmp258
    tmp266 = tmp260 & tmp8
    tmp267 = tmp5 & tmp266
    tmp268 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp267 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp269 = tmp268 - tmp267
    tmp270 = tl.load(in_ptr1 + (-81 + x0 + x1), tmp267 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp271 = tmp269 + tmp270
    tmp272 = tmp271 - tmp265
    tmp273 = tmp267 & tmp8
    tmp274 = tmp5 & tmp273
    tmp275 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp274 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp276 = tmp275 - tmp274
    tmp277 = tl.load(in_ptr1 + (-82 + x0 + x1), tmp274 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp278 = tmp276 + tmp277
    tmp279 = tmp278 - tmp272
    tmp280 = tmp274 & tmp8
    tmp281 = tmp5 & tmp280
    tmp282 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp281 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp283 = tmp282 - tmp281
    tmp284 = tl.load(in_ptr1 + (-82 + x0 + x1), tmp281 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp285 = tmp283 + tmp284
    tmp286 = tmp285 - tmp279
    tmp287 = tmp281 & tmp8
    tmp288 = tmp5 & tmp287
    tmp289 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp288 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp290 = tmp289 - tmp288
    tmp291 = tl.load(in_ptr1 + (-83 + x0 + x1), tmp288 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp292 = tmp290 + tmp291
    tmp293 = tmp292 - tmp286
    tmp294 = tmp288 & tmp8
    tmp295 = tmp5 & tmp294
    tmp296 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp295 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp297 = tmp296 - tmp295
    tmp298 = tl.load(in_ptr1 + (-83 + x0 + x1), tmp295 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp299 = tmp297 + tmp298
    tmp300 = tmp299 - tmp293
    tmp301 = tmp295 & tmp8
    tmp302 = tmp5 & tmp301
    tmp303 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp302 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp304 = tmp303 - tmp302
    tmp305 = tl.load(in_ptr1 + (-84 + x0 + x1), tmp302 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp306 = tmp304 + tmp305
    tmp307 = tmp306 - tmp300
    tmp308 = tmp302 & tmp8
    tmp309 = tmp5 & tmp308
    tmp310 = tl.load(in_ptr0 + (-64 + x0 + 64 * x1 + 4096 * x2), tmp309 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp311 = tmp310 - tmp309
    tmp312 = tl.load(in_ptr1 + (-84 + x0 + x1), tmp309 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp313 = tmp311 + tmp312
    tmp314 = tmp313 - tmp307
    tmp315 = tmp309 & tmp8
    tmp316 = tmp5 & tmp315
    tmp317 = tl.load(in_ptr0 + (-65 + x0 + 64 * x1 + 4096 * x2), tmp316 &
        xmask, eviction_policy='evict_last', other=0.0)
    tmp318 = tmp317 - tmp316
    tmp319 = tl.load(in_ptr1 + (-85 + x0 + x1), tmp316 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp320 = tmp318 + tmp319
    tmp321 = tmp320 - tmp314
    tmp322 = tmp316 & tmp8
    tmp323 = tmp5 &
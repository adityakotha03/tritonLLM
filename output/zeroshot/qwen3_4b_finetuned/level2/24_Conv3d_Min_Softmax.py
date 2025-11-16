import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_min_softmax_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1255584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 384 % 24
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.full([1], 16, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (384 + x3), xmask, eviction_policy='evict_last')
    tmp8 = tmp7 + tmp1
    tmp9 = triton_helpers.maximum(tmp3, tmp8)
    tmp10 = tl.full([1], 16, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tl.load(in_ptr0 + (768 + x3), xmask, eviction_policy='evict_last')
    tmp13 = tmp12 + tmp1
    tmp14 = triton_helpers.maximum(tmp3, tmp13)
    tmp15 = tl.full([1], 16, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp6 & tmp11
    tmp18 = tmp17 & tmp16
    tmp19 = tl.load(in_ptr0 + (1152 + x3), xmask, eviction_policy='evict_last'
        )
    tmp20 = tmp19 + tmp1
    tmp21 = triton_helpers.maximum(tmp3, tmp20)
    tmp22 = tl.full([1], 16, tl.int64)
    tmp23 = tmp21 < tmp22
    tmp24 = tmp18 & tmp23
    tmp25 = tl.load(in_ptr0 + (1536 + x3), xmask, eviction_policy='evict_last')
    tmp26 = tmp25 + tmp1
    tmp27 = triton_helpers.maximum(tmp3, tmp26)
    tmp28 = tl.full([1], 16, tl.int64)
    tmp29 = tmp27 < tmp28
    tmp30 = tmp24 & tmp29
    tmp31 = tl.load(in_ptr0 + (1920 + x3), xmask, eviction_policy='evict_last'
        )
    tmp32 = tmp31 + tmp1
    tmp33 = triton_helpers.maximum(tmp3, tmp32)
    tmp34 = tl.full([1], 16, tl.int64)
    tmp35 = tmp33 < tmp34
    tmp36 = tmp30 & tmp35
    tmp37 = tl.load(in_ptr0 + (2304 + x3), xmask, eviction_policy='evict_last')
    tmp38 = tmp37 + tmp1
    tmp39 = triton_helpers.maximum(tmp3, tmp38)
    tmp40 = tl.full([1], 16, tl.int64)
    tmp41 = tmp39 < tmp40
    tmp42 = tmp36 & tmp41
    tmp43 = tl.load(in_ptr0 + (2688 + x3), xmask, eviction_policy='evict_last')
    tmp44 = tmp43 + tmp1
    tmp45 = triton_helpers.maximum(tmp3, tmp44)
    tmp46 = tl.full([1], 16, tl.int64)
    tmp47 = tmp45 < tmp46
    tmp48 = tmp42 & tmp47
    tmp49 = tl.load(in_ptr0 + (3072 + x3), xmask, eviction_policy='evict_last')
    tmp50 = tmp49 + tmp1
    tmp51 = triton_helpers.maximum(tmp3, tmp50)
    tmp52 = tl.full([1], 16, tl.int64)
    tmp53 = tmp51 < tmp52
    tmp54 = tmp48 & tmp53
    tmp55 = tl.load(in_ptr0 + (3456 + x3), xmask, eviction_policy='evict_last')
    tmp56 = tmp55 + tmp1
    tmp57 = triton_helpers.maximum(tmp3, tmp56)
    tmp58 = tl.full([1], 16, tl.int64)
    tmp59 = tmp57 < tmp58
    tmp60 = tmp54 & tmp59
    tmp61 = tl.load(in_ptr0 + (3840 + x3), xmask, eviction_policy='evict_last')
    tmp62 = tmp61 + tmp1
    tmp63 = triton_helpers.maximum(tmp3, tmp62)
    tmp64 = tl.full([1], 16, tl.int64)
    tmp65 = tmp63 < tmp64
    tmp66 = tmp60 & tmp65
    tmp67 = tl.load(in_ptr0 + (4224 + x3), xmask, eviction_policy='evict_last')
    tmp68 = tmp67 + tmp1
    tmp69 = triton_helpers.maximum(tmp3, tmp68)
    tmp70 = tl.full([1], 16, tl.int64)
    tmp71 = tmp69 < tmp70
    tmp72 = tmp66 & tmp71
    tmp73 = tl.load(in_ptr0 + (4608 + x3), xmask, eviction_policy='evict_last')
    tmp74 = tmp73 + tmp1
    tmp75 = triton_helpers.maximum(tmp3, tmp74)
    tmp76 = tl.full([1], 16, tl.int64)
    tmp77 = tmp75 < tmp76
    tmp78 = tmp72 & tmp77
    tmp79 = tl.load(in_ptr0 + (4992 + x3), xmask, eviction_policy='evict_last')
    tmp80 = tmp79 + tmp1
    tmp81 = triton_helpers.maximum(tmp3, tmp80)
    tmp82 = tl.full([1], 16, tl.int64)
    tmp83 = tmp81 < tmp82
    tmp84 = tmp78 & tmp83
    tmp85 = tl.load(in_ptr0 + (5376 + x3), xmask, eviction_policy='evict_last')
    tmp86 = tmp85 + tmp1
    tmp87 = triton_helpers.maximum(tmp3, tmp86)
    tmp88 = tl.full([1], 16, tl.int64)
    tmp89 = tmp87 < tmp88
    tmp90 = tmp84 & tmp89
    tmp91 = tl.load(in_ptr0 + (5760 + x3), xmask, eviction_policy='evict_last')
    tmp92 = tmp91 + tmp1
    tmp93 = triton_helpers.maximum(tmp3, tmp92)
    tmp94 = tl.full([1], 16, tl.int64)
    tmp95 = tmp93 < tmp94
    tmp96 = tmp90 & tmp95
    tmp97 = tl.load(in_ptr0 + (6144 + x3), xmask, eviction_policy='evict_last')
    tmp98 = tmp97 + tmp1
    tmp99 = triton_helpers.maximum(tmp3, tmp98)
    tmp100 = tl.full([1], 16, tl.int64)
    tmp101 = tmp99 < tmp100
    tmp102 = tmp96 & tmp101
    tmp103 = tl.load(in_ptr0 + (6528 + x3), xmask, eviction_policy='evict_last')
    tmp104 = tmp103 + tmp1
    tmp105 = triton_helpers.maximum(tmp3, tmp104)
    tmp106 = tl.full([1], 16, tl.int64)
    tmp107 = tmp105 < tmp106
    tmp108 = tmp102 & tmp107
    tmp109 = tl.load(in_ptr0 + (6912 + x3), xmask, eviction_policy='evict_last'
        )
    tmp110 = tmp109 + tmp1
    tmp111 = triton_helpers.maximum(tmp3, tmp110)
    tmp112 = tl.full([1], 16, tl.int64)
    tmp113 = tmp111 < tmp112
    tmp114 = tmp108 & tmp113
    tmp115 = tl.load(in_ptr0 + (7296 + x3), xmask, eviction_policy='evict_last'
        )
    tmp116 = tmp115 + tmp1
    tmp117 = triton_helpers.maximum(tmp3, tmp116)
    tmp118 = tl.full([1], 16, tl.int64)
    tmp119 = tmp117 < tmp118
    tmp120 = tmp114 & tmp119
    tmp121 = tl.load(in_ptr0 + (7680 + x3), xmask, eviction_policy='evict_last'
        )
    tmp122 = tmp121 + tmp1
    tmp123 = triton_helpers.maximum(tmp3, tmp122)
    tmp124 = tl.full([1], 16, tl.int64)
    tmp125 = tmp123 < tmp124
    tmp126 = tmp120 & tmp125
    tmp127 = tl.load(in_ptr0 + (8064 + x3), xmask, eviction_policy='evict_last'
        )
    tmp128 = tmp127 + tmp1
    tmp129 = triton_helpers.maximum(tmp3, tmp128)
    tmp130 = tl.full([1], 16, tl.int64)
    tmp131 = tmp129 < tmp130
    tmp132 = tmp126 & tmp131
    tmp133 = tl.load(in_ptr0 + (8448 + x3), xmask, eviction_policy='evict_last'
        )
    tmp134 = tmp133 + tmp1
    tmp135 = triton_helpers.maximum(tmp3, tmp134)
    tmp136 = tl.full([1], 16, tl.int64)
    tmp137 = tmp135 < tmp136
    tmp138 = tmp132 & tmp137
    tmp139 = tl.load(in_ptr0 + (8832 + x3), xmask, eviction_policy='evict_last'
        )
    tmp140 = tmp139 + tmp1
    tmp141 = triton_helpers.maximum(tmp3, tmp140)
    tmp142 = tl.full([1], 16, tl.int64)
    tmp143 = tmp141 < tmp142
    tmp144 = tmp138 & tmp143
    tmp145 = tl.load(in_ptr0 + (9216 + x3), xmask, eviction_policy='evict_last'
        )
    tmp146 = tmp145 + tmp1
    tmp147 = triton_helpers.maximum(tmp3, tmp146)
    tmp148 = tl.full([1], 16, tl.int64)
    tmp149 = tmp147 < tmp148
    tmp150 = tmp144 & tmp149
    tmp151 = tl.load(in_ptr0 + (9600 + x3), xmask, eviction_policy='evict_last'
        )
    tmp152 = tmp151 + tmp1
    tmp153 = triton_helpers.maximum(tmp3, tmp152)
    tmp154 = tl.full([1], 16, tl.int64)
    tmp155 = tmp153 < tmp154
    tmp156 = tmp150 & tmp155
    tmp157 = tl.load(in_ptr0 + (9984 + x3), xmask, eviction_policy='evict_last'
        )
    tmp158 = tmp157 + tmp1
    tmp159 = triton_helpers.maximum(tmp3, tmp158)
    tmp160 = tl.full([1], 16, tl.int64)
    tmp161 = tmp159 < tmp160
    tmp162 = tmp156 & tmp161
    tmp163 = tl.load(in_ptr0 + (10368 + x3), xmask, eviction_policy='evict_last'
        )
    tmp164 = tmp163 + tmp1
    tmp165 = triton_helpers.maximum(tmp3, tmp164)
    tmp166 = tl.full([1], 16, tl.int64)
    tmp167 = tmp165 < tmp166
    tmp168 = tmp162 & tmp167
    tmp169 = tl.load(in_ptr0 + (10752 + x3), xmask, eviction_policy='evict_last'
        )
    tmp170 = tmp169 + tmp1
    tmp171 = triton_helpers.maximum(tmp3, tmp170)
    tmp172 = tl.full([1], 16, tl.int64)
    tmp173 = tmp171 < tmp172
    tmp174 = tmp168 & tmp173
    tmp175 = tl.load(in_ptr0 + (11136 + x3), xmask, eviction_policy='evict_last'
        )
    tmp176 = tmp175 + tmp1
    tmp177 = triton_helpers.maximum(tmp3, tmp176)
    tmp178 = tl.full([1], 16, tl.int64)
    tmp179 = tmp177 < tmp178
    tmp180 = tmp174 & tmp179
    tmp181 = tl.load(in_ptr0 + (11520 + x3), xmask, eviction_policy='evict_last'
        )
    tmp182 = tmp181 + tmp1
    tmp183 = triton_helpers.maximum(tmp3, tmp182)
    tmp184 = tl.full([1], 16, tl.int64)
    tmp185 = tmp183 < tmp184
    tmp186 = tmp180 & tmp185
    tmp187 = tl.load(in_ptr0 + (11904 + x3), xmask, eviction_policy='evict_last'
        )
    tmp188 = tmp187 + tmp1
    tmp189 = triton_helpers.maximum(tmp3, tmp188)
    tmp190 = tl.full([1], 16, tl.int64)
    tmp191 = tmp189 < tmp190
    tmp192 = tmp186 & tmp191
    tmp193 = tl.load(in_ptr0 + (12288 + x3), xmask, eviction_policy='evict_last'
        )
    tmp194 = tmp193 + tmp1
    tmp195 = triton_helpers.maximum(tmp3, tmp194)
    tmp196 = tl.full([1], 16, tl.int64)
    tmp197 = tmp195 < tmp196
    tmp198 = tmp192 & tmp197
    tmp199 = tl.load(in_ptr0 + (12672 + x3), xmask, eviction_policy='evict_last'
        )
    tmp200 = tmp199 + tmp1
    tmp201 = triton_helpers.maximum(tmp3, tmp200)
    tmp202 = tl.full([1], 16, tl.int64)
    tmp203 = tmp201 < tmp202
    tmp204 = tmp198 & tmp203
    tmp205 = tl.load(in_ptr0 + (13056 + x3), xmask, eviction_policy='evict_last'
        )
    tmp206 = tmp205 + tmp1
    tmp207 = triton_helpers.maximum(tmp3, tmp206)
    tmp208 = tl.full([1], 16, tl.int64)
    tmp209 = tmp207 < tmp208
    tmp210 = tmp204 & tmp209
    tmp211 = tl.load(in_ptr0 + (13440 + x3), xmask, eviction_policy='evict_last'
        )
    tmp212 = tmp211 + tmp1
    tmp213 = triton_helpers.maximum(tmp3, tmp212)
    tmp214 = tl.full([1], 16, tl.int64)
    tmp215 = tmp213 < tmp214
    tmp216 = tmp210 & tmp215
    tmp217 = tl.load(in_ptr0 + (13824 + x3), xmask, eviction_policy='evict_last'
        )
    tmp218 = tmp217 + tmp1
    tmp219 = triton_helpers.maximum(tmp3, tmp218)
    tmp220 = tl.full([1], 16, tl.int64)
    tmp221 = tmp219 < tmp220
    tmp222 = tmp216 & tmp221
    tmp223 = tl.load(in_ptr0 + (14208 + x3), xmask, eviction_policy='evict_last'
        )
    tmp224 = tmp223 + tmp1
    tmp225 = triton_helpers.maximum(tmp3, tmp224)
    tmp226 = tl.full([1], 16, tl.int64)
    tmp227 = tmp225 < tmp226
    tmp228 = tmp222 & tmp227
    tmp229 = tl.load(in_ptr0 + (14592 + x3), xmask, eviction_policy='evict_last'
        )
    tmp230 = tmp229 + tmp1
    tmp231 = triton_helpers.maximum(tmp3, tmp230)
    tmp232 = tl.full([1], 16, tl.int64)
    tmp233 = tmp231 < tmp232
    tmp234 = tmp228 & tmp233
    tmp235 = tl.load(in_ptr0 + (14976 + x3), xmask, eviction_policy='evict_last'
        )
    tmp236 = tmp235 + tmp1
    tmp237 = triton_helpers.maximum(tmp3, tmp236)
    tmp238 = tl.full([1], 16, tl.int64)
    tmp239 = tmp237 < tmp238
    tmp240 = tmp234 & tmp239
    tmp241 = tl.load(in_ptr0 + (15360 + x3), xmask, eviction_policy='evict_last'
        )
    tmp242 = tmp241 + tmp1
    tmp243 = triton_helpers.maximum(tmp3, tmp242)
    tmp244 = tl.full([1], 16, tl.int64)
    tmp245 = tmp243 < tmp244
    tmp246 = tmp240 & tmp245
    tmp247 = tl.load(in_ptr0 + (15744 + x3), xmask, eviction_policy='evict_last'
        )
    tmp248 = tmp247 + tmp1
    tmp249 = triton_helpers.maximum(tmp3, tmp248)
    tmp250 = tl.full([1], 16, tl.int64)
    tmp251 = tmp249 < tmp250
    tmp252 = tmp246 & tmp251
    tmp253 = tl.load(in_ptr0 + (16128 + x3), xmask, eviction_policy='evict_last'
        )
    tmp254 = tmp253 + tmp1
    tmp255 = triton_helpers.maximum(tmp3, tmp254)
    tmp256 = tl.full([1], 16, tl.int64)
    tmp257 = tmp255 < tmp256
    tmp258 = tmp252 & tmp257
    tmp259 = tl.load(in_ptr0 + (16512 + x3), xmask, eviction_policy='evict_last'
        )
    tmp260 = tmp259 + tmp1
    tmp261 = triton_helpers.maximum(tmp3, tmp260)
    tmp262 = tl.full([1], 16, tl.int64)
    tmp263 = tmp261 < tmp262
    tmp264 = tmp258 & tmp263
    tmp265 = tl.load(in_ptr0 + (16896 + x3), xmask, eviction_policy='evict_last'
        )
    tmp266 = tmp265 + tmp1
    tmp267 = triton_helpers.maximum(tmp3, tmp266)
    tmp268 = tl.full([1], 16, tl.int64)
    tmp269 = tmp267 < tmp268
    tmp270 = tmp264 & tmp269
    tmp271 = tl.load(in_ptr0 + (17280 + x3), xmask, eviction_policy='evict_last'
        )
    tmp272 = tmp271 + tmp1
    tmp273 = triton_helpers.maximum(tmp3, tmp272)
    tmp274 = tl.full([1], 16, tl.int64)
    tmp275 = tmp273 < tmp274
    tmp276 = tmp270 & tmp275
    tmp277 = tl.load(in_ptr0 + (17664 + x3), xmask, eviction_policy='evict_last'
        )
    tmp278 = tmp277 + tmp1
    tmp279 = triton_helpers.maximum(tmp3, tmp278)
    tmp280 = tl.full([1], 16, tl.int64)
    tmp281 = tmp279 < tmp280
    tmp282 = tmp276 & tmp281
    tmp283 = tl.load(in_ptr0 + (18048 + x3), xmask, eviction_policy='evict_last'
        )
    tmp284 = tmp283 + tmp1
    tmp285 = triton_helpers.maximum(tmp3, tmp284)
    tmp286 = tl.full([1], 16, tl.int64)
    tmp287 = tmp285 < tmp286
    tmp288 = tmp282 & tmp287
    tmp289 = tl.load(in_ptr0 + (18432 + x3), xmask, eviction_policy='evict_last'
        )
    tmp290 = tmp289 + tmp1
    tmp291 = triton_helpers.maximum(tmp3, tmp290)
    tmp292 = tl.full([1], 16, tl.int64)
    tmp293 = tmp291 < tmp292
    tmp294 = tmp288 & tmp293
    tmp295 = tl.load(in_ptr0 + (18816 + x3), xmask, eviction_policy='evict_last'
        )
    tmp296 = tmp295 + tmp1
    tmp297 = triton_helpers.maximum(tmp3, tmp296)
    tmp298 = tl.full([1], 16, tl.int64)
    tmp299 = tmp297 < tmp298
    tmp300 = tmp294 & tmp299
    tmp301 = tl.load(in_ptr0 + (19200 + x3), xmask, eviction_policy='evict_last'
        )
    tmp302 = tmp301 + tmp1
    tmp303 = triton_helpers.maximum(tmp3, tmp302)
    tmp304 = tl.full([1], 16, tl.int64)
    tmp305 = tmp303 < tmp304
    tmp306 = tmp300 & tmp305
    tmp307 = tl.load(in_ptr0 + (19584 + x3), xmask, eviction_policy='evict_last'
        )
    tmp308 = tmp307 + tmp1
    tmp309 = triton_helpers.maximum(tmp3, tmp308)
    tmp310 = tl.full([1], 16, tl.int64)
    tmp311 = tmp309 < tmp310
    tmp312 = tmp306 & tmp311
    tmp313 = tl.load(in_ptr0 + (19968 + x3), xmask, eviction_policy='evict_last'
        )
    tmp314 = tmp313 + tmp1
    tmp315 = triton_helpers.maximum(tmp3, tmp314)
    tmp316 = tl.full([1], 16, tl.int64)
    tmp317 = tmp315 < tmp316
    tmp318 = tmp312 & tmp317
    tmp319 = tl.load(in_ptr0 + (20352 + x3), xmask, eviction_policy='evict_last'
        )
    tmp320 = tmp319 + tmp1
    tmp321 = triton_helpers.maximum(tmp3, tmp320)
    tmp322 = tl.full([1], 16, tl.int64)
    tmp323 = tmp321 < tmp322
    tmp324 = tmp318 & tmp323
    tmp325 = tl.load(in_ptr0 + (20736 + x3), xmask, eviction_policy='evict_last'
        )
    tmp326 = tmp325 + tmp1
    tmp327 = triton_helpers.maximum(tmp3, tmp326)
    tmp328 = tl.full([1], 16, tl.int64)
    tmp329 = tmp327 < tmp328
    tmp330 = tmp324 & tmp329
    tmp331 = tl.load(in_ptr0 + (21120 + x3), xmask, eviction_policy='evict_last'
        )
    tmp332 = tmp331 + tmp1
    tmp333 = triton_helpers.maximum(tmp3, tmp332)
    tmp334 = tl.full([1], 16, tl.int64)
    tmp335 = tmp333 < tmp334
    tmp336 = tmp330 & tmp335
    tmp337 = tl.load(in_ptr0 + (21504 + x3), xmask, eviction_policy='evict_last'
        )
    tmp338 = tmp337 + tmp1
    tmp339 = triton_helpers.maximum(tmp3, tmp338)
    tmp340 = tl.full([1], 16, tl.int64)
    tmp341 = tmp339 < tmp340
    tmp342 = tmp336 & tmp341
    tmp343 = tl.load(in_ptr0 + (21888 + x3), xmask, eviction_policy='evict_last'
        )
    tmp344 = tmp343 + tmp1
    tmp345 = triton_helpers.maximum(tmp3, tmp344)
    tmp346 = tl.full([1], 16, tl.int64)
    tmp347 = tmp345 < tmp346
    tmp348 = tmp342 & tmp347
    tmp349 = tl.load(in_ptr0 + (22272 + x3), xmask, eviction_policy='evict_last'
        )
    tmp350 = tmp349 + tmp1
    tmp351 = triton_helpers.maximum(tmp3, tmp350)
    tmp352 = tl.full([1], 16, tl.int64)
    tmp353 = tmp351 < tmp352
    tmp354 = tmp348 & tmp353
    tmp355 = tl.load(in_ptr0 + (22656 + x3), xmask, eviction_policy='evict_last'
        )
    tmp356 = tmp355 + tmp1
    tmp357 = triton_helpers.maximum(tmp3, tmp356)
    tmp358 = tl.full([1], 16, tl.int64)
    tmp359 = tmp357 < tmp358
    tmp360 = tmp354 & tmp359
    tmp361 = tl.load(in_ptr0 + (23040 + x3), xmask, eviction_policy='evict_last'
        )
    tmp362 = tmp361 + tmp1
    tmp363 = triton_helpers.maximum(tmp3, tmp362)
    tmp364 = tl.full([1], 16, tl.int64)
    tmp365 = tmp363 < tmp364
    tmp366 = tmp360 & tmp365
    tmp367 = tl.load(in_ptr0 + (23424 + x3), xmask, eviction_policy='evict_last'
        )
    tmp368 = tmp367 + tmp1
    tmp369 = triton_helpers.maximum(tmp3, tmp368)
    tmp370 = tl.full([1], 16, tl.int64)
    tmp371 = tmp369 < tmp370
    tmp372 = tmp366 & tmp371
    tmp373 = tl.load(in_ptr0 + (23808 + x3), xmask, eviction_policy='evict_last'
        )
    tmp374 = tmp373 + tmp1
    tmp375 = triton_helpers.maximum(tmp3, tmp374)
    tmp376 = tl.full([1], 16, tl.int64)
    tmp377 = tmp375 < tmp376
    tmp378 = tmp372 & tmp377
    tmp379 = tl.load(in_ptr0 + (24192 + x3), xmask, eviction_policy='evict_last'
        )
    tmp380 = tmp379 + tmp1
    tmp381 = triton_helpers.maximum(tmp3, tmp380)
    tmp382 = tl.full([1], 16, tl.int64)
    tmp383 = tmp381 < tmp382
    tmp384 = tmp378 & tmp383
    tmp385 = tl.load(in_ptr0 + (24576 + x3), xmask, eviction_policy='evict_last'
        )
    tmp386 = tmp385 + tmp1
    tmp387 = triton_helpers.maximum(tmp3, tmp386)
    tmp388 = tl.full([1], 16, tl.int64)
    tmp389 = tmp387 < tmp388
    tmp390 = tmp384 & tmp389
    tmp391 = tl.load(in_ptr0 + (24960 + x3), xmask, eviction_policy='evict_last'
        )
    tmp392 = tmp391 + tmp1
    tmp393 = triton_helpers.maximum(tmp3, tmp392)
    tmp394 = tl.full([1], 16, tl.int64)
    tmp395 = tmp393 < tmp394
    tmp396 = tmp390 & tmp395
    tmp397 = tl.load(in_ptr0 + (25344 + x3), xmask, eviction_policy='evict_last'
        )
    tmp398 = tmp397 + tmp1
    tmp399 = triton_helpers.maximum(tmp3, tmp398)
    tmp400 = tl.full([1], 16, tl.int64)
    tmp401 = tmp399 < tmp400
    tmp402 = tmp396 & tmp401
    tmp403 = tl.load(in_ptr0 + (25728 + x3), xmask, eviction_policy='evict_last'
        )
    tmp404 = tmp403 + tmp1
    tmp405 = triton_helpers.maximum(tmp3, tmp404)
    tmp406 = tl.full([1], 16, tl.int64)
    tmp407 = tmp405 < tmp406
    tmp408 = tmp402 & tmp407
    tmp409 = tl.load(in_ptr0 + (26112 + x3), xmask, eviction_policy='evict_last'
        )
    tmp410 = tmp409 + tmp1
    tmp411 = triton_helpers.maximum(tmp3, tmp410)
    tmp412 = tl.full([1], 16, tl.int64)
    tmp413 = tmp411 < tmp412
    tmp414 = tmp408 & tmp413
    tmp415 = tl.load(in_ptr0 + (26496 + x3), xmask, eviction_policy='evict_last'
        )
    tmp416 = tmp415 + tmp1
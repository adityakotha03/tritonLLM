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
def triton_per_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(out_ptr2 + x0, tmp22, xmask)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 14400 % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_relu_2(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr1 + (1 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr2 + (1 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp8 = tl.load(in_ptr1 + (2 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr2 + (2 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr2 + (3 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr1 + (4 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr2 + (4 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr1 + (5 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr2 + (5 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr1 + (6 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr2 + (6 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr1 + (7 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr2 + (7 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr1 + (8 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr2 + (8 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr1 + (9 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr2 + (9 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr1 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr2 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr1 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr2 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr1 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr2 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr1 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr2 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr1 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr2 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr1 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr2 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 + tmp4
    tmp5 = tmp5 + tmp4
    tmp6 = tmp5 + tmp7
    tmp8 = tmp6 + tmp8
    tmp9 = tmp9 + tmp8
    tmp10 = tmp10 + tmp9
    tmp11 = tmp11 + tmp10
    tmp12 = tmp12 + tmp11
    tmp13 = tmp13 + tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp14 + tmp13
    tmp16 = tmp16 + tmp15
    tmp17 = tmp17 + tmp16
    tmp18 = tmp18 + tmp17
    tmp19 = tmp19 + tmp18
    tmp20 = tmp20 + tmp19
    tmp21 = tmp21 + tmp20
    tmp22 = tmp22 + tmp21
    tmp23 = tmp23 + tmp22
    tmp24 = tmp24 + tmp23
    tmp25 = tmp25 + tmp24
    tmp26 = tmp26 + tmp25
    tmp27 = tmp27 + tmp26
    tmp28 = tmp28 + tmp27
    tmp29 = tmp29 + tmp28
    tmp30 = tmp30 + tmp29
    tmp31 = tmp31 + tmp30
    tmp32 = tmp32 + tmp31
    tmp33 = tmp33 + tmp32
    tmp34 = tmp34 + tmp33
    tmp35 = tmp35 + tmp34
    tmp36 = tmp36 + tmp35
    tmp37 = tmp37 + tmp36
    tmp38 = tmp38 + tmp37
    tmp39 = tmp39 + tmp38
    tmp40 = tmp40 + tmp39
    tmp41 = tmp41 + tmp40
    tmp42 = tmp42 + tmp41
    tmp43 = tmp43 + tmp42
    tmp44 = tmp44 + tmp43
    tmp45 = tmp45 + tmp44
    tmp46 = tmp46 + tmp45
    tmp47 = tmp47 + tmp46
    tmp48 = tmp48 + tmp47
    tmp49 = tmp49 + tmp48
    tmp50 = tmp50 + tmp49
    tmp51 = tmp51 + tmp50
    tmp52 = tmp52 + tmp51
    tmp53 = tmp53 + tmp52
    tmp54 = tmp54 + tmp53
    tmp55 = tmp55 + tmp54
    tmp56 = tmp56 + tmp55
    tmp57 = tmp57 + tmp56
    tmp58 = tmp58 + tmp57
    tmp59 = tmp59 + tmp58
    tmp60 = tmp60 + tmp59
    tmp61 = tmp61 + tmp60
    tmp62 = tmp62 + tmp61
    tmp63 = tmp63 + tmp62
    tmp64 = tmp64 + tmp63
    tmp65 = tmp65 + tmp64
    tmp66 = tmp66 + tmp65
    tmp67 = tmp67 + tmp66
    tmp68 = tmp68 + tmp67
    tmp69 = tmp69 + tmp68
    tmp70 = tmp70 + tmp69
    tmp71 = tmp71 + tmp70
    tmp72 = tmp72 + tmp71
    tmp73 = tmp73 + tmp72
    tmp74 = tmp73 + tmp73
    tmp75 = tmp74 + tmp74
    tmp76 = tmp75 + tmp75
    tmp77 = tmp76 + tmp76
    tmp78 = tmp77 + tmp77
    tmp79 = tmp78 + tmp78
    tmp80 = tmp79 + tmp79
    tmp81 = tmp80 + tmp80
    tmp82 = tmp81 + tmp81
    tmp83 = tmp82 + tmp82
    tmp84 = tmp83 + tmp83
    tmp85 = tmp84 + tmp84
    tmp86 = tmp85 + tmp85
    tmp87 = tmp86 + tmp86
    tmp88 = tmp87 + tmp87
    tmp89 = tmp88 + tmp88
    tmp90 = tmp89 + tmp89
    tmp91 = tmp90 + tmp90
    tmp92 = tmp91 + tmp91
    tmp93 = tmp92 + tmp92
    tmp94 = tmp93 + tmp93
    tmp95 = tmp94 + tmp94
    tmp96 = tmp95 + tmp95
    tmp97 = tmp96 + tmp96
    tmp98 = tmp97 + tmp97
    tmp99 = tmp98 + tmp98
    tmp100 = tmp99 + tmp99
    tmp101 = tmp100 + tmp100
    tmp102 = tmp101 + tmp101
    tmp103 = tmp102 + tmp102
    tmp104 = tmp103 + tmp103
    tmp105 = tmp104 + tmp104
    tmp106 = tmp105 + tmp105
    tmp107 = tmp106 + tmp106
    tmp108 = tmp107 + tmp107
    tmp109 = tmp108 + tmp108
    tmp110 = tmp109 + tmp109
    tmp111 = tmp110 + tmp110
    tmp112 = tmp111 + tmp111
    tmp113 = tmp112 + tmp112
    tmp114 = tmp113 + tmp113
    tmp115 = tmp114 + tmp114
    tmp116 = tmp115 + tmp115
    tmp117 = tmp116 + tmp116
    tmp118 = tmp117 + tmp117
    tmp119 = tmp118 + tmp118
    tmp120 = tmp119 + tmp119
    tmp121 = tmp120 + tmp120
    tmp122 = tmp121 + tmp121
    tmp123 = tmp122 + tmp122
    tmp124 = tmp123 + tmp123
    tmp125 = tmp124 + tmp124
    tmp126 = tmp125 + tmp125
    tmp127 = tmp126 + tmp126
    tmp128 = tmp127 + tmp127
    tmp129 = tmp128 + tmp128
    tmp130 = tmp129 + tmp129
    tmp131 = tmp130 + tmp130
    tmp132 = tmp131 + tmp131
    tmp133 = tmp132 + tmp132
    tmp134 = tmp133 + tmp133
    tmp135 = tmp134 + tmp134
    tmp136 = tmp135 + tmp135
    tmp137 = tmp136 + tmp136
    tmp138 = tmp137 + tmp137
    tmp139 = tmp138 + tmp138
    tmp140 = tmp139 + tmp139
    tmp141 = tmp140 + tmp140
    tmp142 = tmp141 + tmp141
    tmp143 = tmp142 + tmp142
    tmp144 = tmp143 + tmp143
    tmp145 = tmp144 + tmp144
    tmp146 = tmp145 + tmp145
    tmp147 = tmp146 + tmp146
    tmp148 = tmp147 + tmp147
    tmp149 = tmp148 + tmp148
    tmp150 = tmp149 + tmp149
    tmp151 = tmp150 + tmp150
    tmp152 = tmp151 + tmp151
    tmp153 = tmp152 + tmp152
    tmp154 = tmp153 + tmp153
    tmp155 = tmp154 + tmp154
    tmp156 = tmp155 + tmp155
    tmp157 = tmp156 + tmp156
    tmp158 = tmp157 + tmp157
    tmp159 = tmp158 + tmp158
    tmp160 = tmp159 + tmp159
    tmp161 = tmp160 + tmp160
    tmp162 = tmp161 + tmp161
    tmp163 = tmp162 + tmp162
    tmp164 = tmp163 + tmp163
    tmp165 = tmp164 + tmp164
    tmp166 = tmp165 + tmp165
    tmp167 = tmp166 + tmp166
    tmp168 = tmp167 + tmp167
    tmp169 = tmp168 + tmp168
    tmp170 = tmp169 + tmp169
    tmp171 = tmp170 + tmp170
    tmp172 = tmp171 + tmp171
    tmp173 = tmp172 + tmp172
    tmp174 = tmp173 + tmp173
    tmp175 = tmp174 + tmp174
    tmp176 = tmp175 + tmp175
    tmp177 = tmp176 + tmp176
    tmp178 = tmp177 + tmp177
    tmp179 = tmp178 + tmp178
    tmp180 = tmp179 + tmp179
    tmp181 = tmp180 + tmp180
    tmp182 = tmp181 + tmp181
    tmp183 = tmp182 + tmp182
    tmp184 = tmp183 + tmp183
    tmp185 = tmp184 + tmp184
    tmp186 = tmp185 + tmp185
    tmp187 = tmp186 + tmp186
    tmp188 = tmp187 + tmp187
    tmp189 = tmp188 + tmp188
    tmp190 = tmp189 + tmp189
    tmp191 = tmp190 + tmp190
    tmp192 = tmp191 + tmp191
    tmp193 = tmp192 + tmp192
    tmp194 = tmp193 + tmp193
    tmp195 = tmp194 + tmp194
    tmp196 = tmp195 + tmp195
    tmp197 = tmp196 + tmp196
    tmp198 = tmp197 + tmp197
    tmp199 = tmp198 + tmp198
    tmp200 = tmp199 + tmp199
    tmp201 = tmp200 + tmp200
    tmp202 = tmp201 + tmp201
    tmp203 = tmp202 + tmp202
    tmp204 = tmp203 + tmp203
    tmp205 = tmp204 + tmp204
    tmp206 = tmp205 + tmp205
    tmp207 = tmp206 + tmp206
    tmp208 = tmp207 + tmp207
    tmp209 = tmp208 + tmp208
    tmp210 = tmp209 + tmp209
    tmp211 = tmp210 + tmp210
    tmp212 = tmp211 + tmp211
    tmp213 = tmp212 + tmp212
    tmp214 = tmp213 + tmp213
    tmp215 = tmp214 + tmp214
    tmp216 = tmp215 + tmp215
    tmp217 = tmp216 + tmp216
    tmp218 = tmp217 + tmp217
    tmp219 = tmp218 + tmp218
    tmp220 = tmp219 + tmp219
    tmp221 = tmp220 + tmp220
    tmp222 = tmp221 + tmp221
    tmp223 = tmp222 + tmp222
    tmp224 = tmp223 + tmp223
    tmp225 = tmp224 + tmp224
    tmp226 = tmp225 + tmp225
    tmp227 = tmp226 + tmp226
    tmp228 = tmp227 + tmp227
    tmp229 = tmp228 + tmp228
    tmp230 = tmp229 + tmp229
    tmp231 = tmp230 + tmp230
    tmp232 = tmp231 + tmp231
    tmp233 = tmp232 + tmp232
    tmp234 = tmp233 + tmp233
    tmp235 = tmp234 + tmp234
    tmp236 = tmp235 + tmp235
    tmp237 = tmp236 + tmp236
    tmp238 = tmp237 + tmp237
    tmp239 = tmp238 + tmp238
    tmp240 = tmp239 + tmp239
    tmp241 = tmp240 + tmp240
    tmp242 = tmp241 + tmp241
    tmp243 = tmp242 + tmp242
    tmp244 = tmp243 + tmp243
    tmp245 = tmp244 + tmp244
    tmp246 = tmp245 + tmp245
    tmp247 = tmp246 + tmp246
    tmp248 = tmp247 + tmp247
    tmp249 = tmp248 + tmp248
    tmp250 = tmp249 + tmp249
    tmp251 = tmp250 + tmp250
    tmp252 = tmp251 + tmp251
    tmp253 = tmp252 + tmp252
    tmp254 = tmp253 + tmp253
    tmp255 = tmp254 + tmp254
    tmp256 = tmp255 + tmp255
    tmp257 = tmp256 + tmp256
    tmp258 = tmp257 + tmp257
    tmp259 = tmp258 + tmp258
    tmp260 = tmp259 + tmp259
    tmp261 = tmp260 + tmp260
    tmp262 = tmp261 + tmp261
    tmp263 = tmp262 + tmp262
    tmp264 = tmp263 + tmp263
    tmp265 = tmp264 + tmp264
    tmp266 = tmp265 + tmp265
    tmp267 = tmp266 + tmp266
    tmp268 = tmp267 + tmp267
    tmp269 = tmp268 + tmp268
    tmp270 = tmp269 + tmp269
    tmp271 = tmp270 + tmp270
    tmp272 = tmp271 + tmp271
    tmp273 = tmp272 + tmp272
    tmp274 = tmp273 + tmp273
    tmp275 = tmp274 + tmp274
    tmp276 = tmp275 + tmp275
    tmp277 = tmp276 + tmp276
    tmp278 = tmp277 + tmp277
    tmp279 = tmp278 + tmp278
    tmp280 = tmp279 + tmp279
    tmp281 = tmp280 + tmp280
    tmp282 = tmp281 + tmp281
    tmp283 = tmp282 + tmp282
    tmp284 = tmp283 + tmp283
    tmp285 = tmp284 + tmp284
    tmp286 = tmp285 + tmp285
    tmp287 = tmp286 + tmp286
    tmp288 = tmp287 + tmp287
    tmp289 = tmp288 + tmp288
    tmp290 = tmp289 + tmp289
    tmp291 = tmp290 + tmp290
    tmp292 = tmp291 + tmp291
    tmp293 = tmp292 + tmp292
    tmp294 = tmp293 + tmp293
    tmp295 = tmp294 + tmp294
    tmp296 = tmp295 + tmp295
    tmp297 = tmp296 + tmp296
    tmp298 = tmp297 + tmp297
    tmp299 = tmp298 + tmp298
    tmp300 = tmp299 + tmp299
    tmp301 = tmp300 + tmp300
    tmp302 = tmp301 + tmp301
    tmp303 = tmp302 + tmp302
    tmp304 = tmp303 + tmp303
    tmp305 = tmp304 + tmp304
    tmp306 = tmp305 + tmp305
    tmp307 = tmp306 + tmp306
    tmp308 = tmp307 + tmp307
    tmp309 = tmp308 + tmp308
    tmp310 = tmp309 + tmp309
    tmp311 = tmp310 + tmp310
    tmp312 = tmp311 + tmp311
    tmp313 = tmp312 + tmp312
    tmp314 = tmp313 + tmp313
    tmp315 = tmp314 + tmp314
    tmp316 = tmp315 + tmp315
    tmp317 = tmp316 + tmp316
    tmp318 = tmp317 + tmp317
    tmp319 = tmp318 + tmp318
    tmp320 = tmp319 + tmp319
    tmp321 = tmp320 + tmp320
    tmp322 = tmp321 + tmp321
    tmp323 = tmp322 + tmp322
    tmp324 = tmp323 + tmp323
    tmp325 = tmp324 + tmp324
    tmp326 = tmp325 + tmp325
    tmp327 = tmp326 + tmp326
    tmp328 = tmp327 + tmp327
    tmp329 = tmp328 + tmp328
    tmp330 = tmp329 + tmp329
    tmp331 = tmp330 + tmp330
    tmp332 = tmp331 + tmp331
    tmp333 = tmp332 + tmp332
    tmp334 = tmp333 + tmp333
    tmp335 = tmp334 + tmp334
    tmp336 = tmp335 + tmp335
    tmp337 = tmp336 + tmp336
    tmp338 = tmp337 + tmp337
    tmp339 = tmp338 + tmp338
    tmp340 = tmp339 + tmp339
    tmp341 = tmp340 + tmp340
    tmp342 = tmp341 + tmp341
    tmp343 = tmp342 + tmp342
    tmp344 = tmp343 + tmp343
    tmp345 = tmp344 + tmp344
    tmp346 = tmp345 + tmp345
    tmp347 = tmp346 + tmp346
    tmp348 = tmp347 + tmp347
    tmp349 = tmp348 + tmp348
    tmp350 = tmp349 + tmp349
    tmp351 = tmp350 + tmp350
    tmp352 = tmp351 + tmp351
    tmp353 = tmp352 + tmp352
    tmp354 = tmp353 + tmp353
    tmp355 = tmp354 + tmp354
    tmp356 = tmp355 + tmp355
    tmp357 = tmp356 + tmp356
    tmp358 = tmp357 + tmp357
    tmp359 = tmp358 + tmp358
    tmp360 = tmp359 + tmp359
    tmp361 = tmp360 + tmp360
    tmp362 = tmp361 + tmp361
    tmp363 = tmp362 + tmp362
    tmp364 = tmp363 + tmp363
    tmp365 = tmp364 + tmp364
    tmp366 = tmp365 + tmp365
    tmp367 = tmp366 + tmp366
    tmp368 = tmp367 + tmp367
    tmp369 = tmp368 + tmp368
    tmp370 = tmp369 + tmp369
    tmp371 = tmp370 + tmp370
    tmp372 = tmp371 + tmp371
    tmp373 = tmp372 + tmp372
    tmp374 = tmp3
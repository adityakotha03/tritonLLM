import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__log_sum_exp_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 == tmp1
    tmp3 = tmp0 + tmp2
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp4 == tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8 == tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp7 - tmp10
    tmp12 = tmp11.to(tl.int32)
    tmp13 = tmp12 == tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 - tmp14
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tmp16 == tmp16
    tmp18 = tmp15 - tmp17
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tmp19 == tmp19
    tmp21 = tmp18 - tmp20
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tmp22 == tmp22
    tmp24 = tmp21 - tmp23
    tmp25 = tmp24.to(tl.int32)
    tmp26 = tmp25 == tmp25
    tmp27 = tmp24 - tmp26
    tmp28 = tmp27.to(tl.int32)
    tmp29 = tmp28 == tmp28
    tmp30 = tmp27 - tmp29
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tmp31 == tmp31
    tmp33 = tmp30 - tmp32
    tmp34 = tmp33.to(tl.int32)
    tmp35 = tmp34 == tmp34
    tmp36 = tmp33 - tmp35
    tmp37 = tmp36.to(tl.int32)
    tmp38 = tmp37 == tmp37
    tmp39 = tmp36 - tmp38
    tmp40 = tmp39.to(tl.int32)
    tmp41 = tmp40 == tmp40
    tmp42 = tmp39 - tmp41
    tmp43 = tmp42.to(tl.int32)
    tmp44 = tmp43 == tmp43
    tmp45 = tmp42 - tmp44
    tmp46 = tmp45.to(tl.int32)
    tmp47 = tmp46 == tmp46
    tmp48 = tmp45 - tmp47
    tmp49 = tmp48.to(tl.int32)
    tmp50 = tmp49 == tmp49
    tmp51 = tmp48 - tmp50
    tmp52 = tmp51.to(tl.int32)
    tmp53 = tmp52 == tmp52
    tmp54 = tmp51 - tmp53
    tmp55 = tmp54.to(tl.int32)
    tmp56 = tmp55 == tmp55
    tmp57 = tmp54 - tmp56
    tmp58 = tmp57.to(tl.int32)
    tmp59 = tmp58 == tmp58
    tmp60 = tmp57 - tmp59
    tmp61 = tmp60.to(tl.int32)
    tmp62 = tmp61 == tmp61
    tmp63 = tmp60 - tmp62
    tmp64 = tmp63.to(tl.int32)
    tmp65 = tmp64 == tmp64
    tmp66 = tmp63 - tmp65
    tmp67 = tmp66.to(tl.int32)
    tmp68 = tmp67 == tmp67
    tmp69 = tmp66 - tmp68
    tmp70 = tmp69.to(tl.int32)
    tmp71 = tmp70 == tmp70
    tmp72 = tmp69 - tmp71
    tmp73 = tmp72.to(tl.int32)
    tmp74 = tmp73 == tmp73
    tmp75 = tmp72 - tmp74
    tmp76 = tmp75.to(tl.int32)
    tmp77 = tmp76 == tmp76
    tmp78 = tmp75 - tmp77
    tmp79 = tmp78.to(tl.int32)
    tmp80 = tmp79 == tmp79
    tmp81 = tmp78 - tmp80
    tmp82 = tmp81.to(tl.int32)
    tmp83 = tmp82 == tmp82
    tmp84 = tmp81 - tmp83
    tmp85 = tmp84.to(tl.int32)
    tmp86 = tmp85 == tmp85
    tmp87 = tmp84 - tmp86
    tmp88 = tmp87.to(tl.int32)
    tmp89 = tmp88 == tmp88
    tmp90 = tmp87 - tmp89
    tmp91 = tmp90.to(tl.int32)
    tmp92 = tmp91 == tmp91
    tmp93 = tmp90 - tmp92
    tmp94 = tmp93.to(tl.int32)
    tmp95 = tmp94 == tmp94
    tmp96 = tmp93 - tmp95
    tmp97 = tmp96.to(tl.int32)
    tmp98 = tmp97 == tmp97
    tmp99 = tmp96 - tmp98
    tmp100 = tmp99.to(tl.int32)
    tmp101 = tmp100 == tmp100
    tmp102 = tmp99 - tmp101
    tmp103 = tmp102.to(tl.int32)
    tmp104 = tmp103 == tmp103
    tmp105 = tmp102 - tmp104
    tmp106 = tmp105.to(tl.int32)
    tmp107 = tmp106 == tmp106
    tmp108 = tmp105 - tmp107
    tmp109 = tmp108.to(tl.int32)
    tmp110 = tmp109 == tmp109
    tmp111 = tmp108 - tmp110
    tmp112 = tmp111.to(tl.int32)
    tmp113 = tmp112 == tmp112
    tmp114 = tmp111 - tmp113
    tmp115 = tmp114.to(tl.int32)
    tmp116 = tmp115 == tmp115
    tmp117 = tmp114 - tmp116
    tmp118 = tmp117.to(tl.int32)
    tmp119 = tmp118 == tmp118
    tmp120 = tmp117 - tmp119
    tmp121 = tmp120.to(tl.int32)
    tmp122 = tmp121 == tmp121
    tmp123 = tmp120 - tmp122
    tmp124 = tmp123.to(tl.int32)
    tmp125 = tmp124 == tmp124
    tmp126 = tmp123 - tmp125
    tmp127 = tmp126.to(tl.int32)
    tmp128 = tmp127 == tmp127
    tmp129 = tmp126 - tmp128
    tmp130 = tmp129.to(tl.int32)
    tmp131 = tmp130 == tmp130
    tmp132 = tmp129 - tmp131
    tmp133 = tmp132.to(tl.int32)
    tmp134 = tmp133 == tmp133
    tmp135 = tmp132 - tmp134
    tmp136 = tmp135.to(tl.int32)
    tmp137 = tmp136 == tmp136
    tmp138 = tmp135 - tmp137
    tmp139 = tmp138.to(tl.int32)
    tmp140 = tmp139 == tmp139
    tmp141 = tmp138 - tmp140
    tmp142 = tmp141.to(tl.int32)
    tmp143 = tmp142 == tmp142
    tmp144 = tmp141 - tmp143
    tmp145 = tmp144.to(tl.int32)
    tmp146 = tmp145 == tmp145
    tmp147 = tmp144 - tmp146
    tmp148 = tmp147.to(tl.int32)
    tmp149 = tmp148 == tmp148
    tmp150 = tmp147 - tmp149
    tmp151 = tmp150.to(tl.int32)
    tmp152 = tmp151 == tmp151
    tmp153 = tmp150 - tmp152
    tmp154 = tmp153.to(tl.int32)
    tmp155 = tmp154 == tmp154
    tmp156 = tmp153 - tmp155
    tmp157 = tmp156.to(tl.int32)
    tmp158 = tmp157 == tmp157
    tmp159 = tmp156 - tmp158
    tmp160 = tmp159.to(tl.int32)
    tmp161 = tmp160 == tmp160
    tmp162 = tmp159 - tmp161
    tmp163 = tmp162.to(tl.int32)
    tmp164 = tmp163 == tmp163
    tmp165 = tmp162 - tmp164
    tmp166 = tmp165.to(tl.int32)
    tmp167 = tmp166 == tmp166
    tmp168 = tmp165 - tmp167
    tmp169 = tmp168.to(tl.int32)
    tmp170 = tmp169 == tmp169
    tmp171 = tmp168 - tmp170
    tmp172 = tmp171.to(tl.int32)
    tmp173 = tmp172 == tmp172
    tmp174 = tmp171 - tmp173
    tmp175 = tmp174.to(tl.int32)
    tmp176 = tmp175 == tmp175
    tmp177 = tmp174 - tmp176
    tmp178 = tmp177.to(tl.int32)
    tmp179 = tmp178 == tmp178
    tmp180 = tmp177 - tmp179
    tmp181 = tmp180.to(tl.int32)
    tmp182 = tmp181 == tmp181
    tmp183 = tmp180 - tmp182
    tmp184 = tmp183.to(tl.int32)
    tmp185 = tmp184 == tmp184
    tmp186 = tmp183 - tmp185
    tmp187 = tmp186.to(tl.int32)
    tmp188 = tmp187 == tmp187
    tmp189 = tmp186 - tmp188
    tmp190 = tmp189.to(tl.int32)
    tmp191 = tmp190 == tmp190
    tmp192 = tmp189 - tmp191
    tmp193 = tmp192.to(tl.int32)
    tmp194 = tmp193 == tmp193
    tmp195 = tmp192 - tmp194
    tmp196 = tmp195.to(tl.int32)
    tmp197 = tmp196 == tmp196
    tmp198 = tmp195 - tmp197
    tmp199 = tmp198.to(tl.int32)
    tmp200 = tmp199 == tmp199
    tmp201 = tmp198 - tmp200
    tmp202 = tmp201.to(tl.int32)
    tmp203 = tmp202 == tmp202
    tmp204 = tmp201 - tmp203
    tmp205 = tmp204.to(tl.int32)
    tmp206 = tmp205 == tmp205
    tmp207 = tmp204 - tmp206
    tmp208 = tmp207.to(tl.int32)
    tmp209 = tmp208 == tmp208
    tmp210 = tmp207 - tmp209
    tmp211 = tmp210.to(tl.int32)
    tmp212 = tmp211 == tmp211
    tmp213 = tmp210 - tmp212
    tmp214 = tmp213.to(tl.int32)
    tmp215 = tmp214 == tmp214
    tmp216 = tmp213 - tmp215
    tmp217 = tmp216.to(tl.int32)
    tmp218 = tmp217 == tmp217
    tmp219 = tmp216 - tmp218
    tmp220 = tmp219.to(tl.int32)
    tmp221 = tmp220 == tmp220
    tmp222 = tmp219 - tmp221
    tmp223 = tmp222.to(tl.int32)
    tmp224 = tmp223 == tmp223
    tmp225 = tmp222 - tmp224
    tmp226 = tmp225.to(tl.int32)
    tmp227 = tmp226 == tmp226
    tmp228 = tmp225 - tmp227
    tmp229 = tmp228.to(tl.int32)
    tmp230 = tmp229 == tmp229
    tmp231 = tmp228 - tmp230
    tmp232 = tmp231.to(tl.int32)
    tmp233 = tmp232 == tmp232
    tmp234 = tmp231 - tmp233
    tmp235 = tmp234.to(tl.int32)
    tmp236 = tmp235 == tmp235
    tmp237 = tmp234 - tmp236
    tmp238 = tmp237.to(tl.int32)
    tmp239 = tmp238 == tmp238
    tmp240 = tmp237 - tmp239
    tmp241 = tmp240.to(tl.int32)
    tmp242 = tmp241 == tmp241
    tmp243 = tmp240 - tmp242
    tmp244 = tmp243.to(tl.int32)
    tmp245 = tmp244 == tmp244
    tmp246 = tmp243 - tmp245
    tmp247 = tmp246.to(tl.int32)
    tmp248 = tmp247 == tmp247
    tmp249 = tmp246 - tmp248
    tmp250 = tmp249.to(tl.int32)
    tmp251 = tmp250 == tmp250
    tmp252 = tmp249 - tmp251
    tmp253 = tmp252.to(tl.int32)
    tmp254 = tmp253 == tmp253
    tmp255 = tmp252 - tmp254
    tmp256 = tmp255.to(tl.int32)
    tmp257 = tmp256 == tmp256
    tmp258 = tmp255 - tmp257
    tmp259 = tmp258.to(tl.int32)
    tmp260 = tmp259 == tmp259
    tmp261 = tmp258 - tmp260
    tmp262 = tmp261.to(tl.int32)
    tmp263 = tmp262 == tmp262
    tmp264 = tmp261 - tmp263
    tmp265 = tmp264.to(tl.int32)
    tmp266 = tmp265 == tmp265
    tmp267 = tmp264 - tmp266
    tmp268 = tmp267.to(tl.int32)
    tmp269 = tmp268 == tmp268
    tmp270 = tmp267 - tmp269
    tmp271 = tmp270.to(tl.int32)
    tmp272 = tmp271 == tmp271
    tmp273 = tmp270 - tmp272
    tmp274 = tmp273.to(tl.int32)
    tmp275 = tmp274 == tmp274
    tmp276 = tmp273 - tmp275
    tmp277 = tmp276.to(tl.int32)
    tmp278 = tmp277 == tmp277
    tmp279 = tmp276 - tmp278
    tmp280 = tmp279.to(tl.int32)
    tmp281 = tmp280 == tmp280
    tmp282 = tmp279 - tmp281
    tmp283 = tmp282.to(tl.int32)
    tmp284 = tmp283 == tmp283
    tmp285 = tmp282 - tmp284
    tmp286 = tmp285.to(tl.int32)
    tmp287 = tmp286 == tmp286
    tmp288 = tmp285 - tmp287
    tmp289 = tmp288.to(tl.int32)
    tmp290 = tmp289 == tmp289
    tmp291 = tmp288 - tmp290
    tmp292 = tmp291.to(tl.int32)
    tmp293 = tmp292 == tmp292
    tmp294 = tmp291 - tmp293
    tmp295 = tmp294.to(tl.int32)
    tmp296 = tmp295 == tmp295
    tmp297 = tmp294 - tmp296
    tmp298 = tmp297.to(tl.int32)
    tmp299 = tmp298 == tmp298
    tmp300 = tmp297 - tmp299
    tmp301 = tmp300.to(tl.int32)
    tmp302 = tmp301 == tmp301
    tmp303 = tmp300 - tmp302
    tmp304 = tmp303.to(tl.int32)
    tmp305 = tmp304 == tmp304
    tmp306 = tmp303 - tmp305
    tmp307 = tmp306.to(tl.int32)
    tmp308 = tmp307 == tmp307
    tmp309 = tmp306 - tmp308
    tmp310 = tmp309.to(tl.int32)
    tmp311 = tmp310 == tmp310
    tmp312 = tmp309 - tmp311
    tmp313 = tmp312.to(tl.int32)
    tmp314 = tmp313 == tmp313
    tmp315 = tmp312 - tmp314
    tmp316 = tmp315.to(tl.int32)
    tmp317 = tmp316 == tmp316
    tmp318 = tmp315 - tmp317
    tmp319 = tmp318.to(tl.int32)
    tmp320 = tmp319 == tmp319
    tmp321 = tmp318 - tmp320
    tmp322 = tmp321.to(tl.int32)
    tmp323 = tmp322 == tmp322
    tmp324 = tmp321 - tmp323
    tmp325 = tmp324.to(tl.int32)
    tmp326 = tmp325 == tmp325
    tmp327 = tmp324 - tmp326
    tmp328 = tmp327.to(tl.int32)
    tmp329 = tmp328 == tmp328
    tmp330 = tmp327 - tmp329
    tmp331 = tmp330.to(tl.int32)
    tmp332 = tmp331 == tmp331
    tmp333 = tmp330 - tmp332
    tmp334 = tmp333.to(tl.int32)
    tmp335 = tmp334 == tmp334
    tmp336 = tmp333 - tmp335
    tmp337 = tmp336.to(tl.int32)
    tmp338 = tmp337 == tmp337
    tmp339 = tmp336 - tmp338
    tmp340 = tmp339.to(tl.int32)
    tmp341 = tmp340 == tmp340
    tmp342 = tmp339 - tmp341
    tmp343 = tmp342.to(tl.int32)
    tmp344 = tmp343 == tmp343
    tmp345 = tmp342 - tmp344
    tmp346 = tmp345.to(tl.int32)
    tmp347 = tmp346 == tmp346
    tmp348 = tmp345 - tmp347
    tmp349 = tmp348.to(tl.int32)
    tmp350 = tmp349 == tmp349
    tmp351 = tmp348 - tmp350
    tmp352 = tmp351.to(tl.int32)
    tmp353 = tmp352 == tmp352
    tmp354 = tmp351 - tmp353
    tmp355 = tmp354.to(tl.int32)
    tmp356 = tmp355 == tmp355
    tmp357 = tmp354 - tmp356
    tmp358 = tmp357.to(tl.int32)
    tmp359 = tmp358 == tmp358
    tmp360 = tmp357 - tmp359
    tmp361 = tmp360.to(tl.int32)
    tmp362 = tmp361 == tmp361
    tmp363 = tmp360 - tmp362
    tmp364 = tmp363.to(tl.int32)
    tmp365 = tmp364 == tmp364
    tmp366 = tmp363 - tmp365
    tmp367 = tmp366.to(tl.int32)
    tmp368 = tmp367 == tmp367
    tmp369 = tmp366 - tmp368
    tmp370 = tmp369.to(tl.int32)
    tmp371 = tmp370 == tmp370
    tmp372 = tmp369 - tmp371
    tmp373 = tmp372.to(tl.int32)
    tmp374 = tmp373 == tmp373
    tmp375 = tmp372 - tmp374
    tmp376 = tmp375.to(tl.int32)
    tmp377 = tmp376 == tmp376
    tmp378 = tmp375 - tmp377
    tmp379 = tmp378.to(tl.int32)
    tmp380 = tmp379 == tmp379
    tmp381 = tmp378 - tmp380
    tmp382 = tmp381.to(tl.int32)
    tmp383 = tmp382 == tmp382
    tmp384 = tmp381 - tmp383
    tmp385 = tmp384.to(tl.int32)
    tmp386 = tmp385 == tmp385
    tmp387 = tmp384 - tmp386
    tmp388 = tmp387.to(tl.int32)
    tmp389 = tmp388 == tmp388
    tmp390 = tmp387 - tmp389
    tmp391 = tmp390.to(tl.int32)
    tmp392 = tmp391 == tmp391
    tmp393 = tmp390 - tmp392
    tmp394 = tmp393.to(tl.int32)
    tmp395 = tmp394 == tmp394
    tmp396 = tmp393 - tmp395
    tmp397 = tmp396.to(tl.int32)
    tmp398 = tmp397 == tmp397
    tmp399 = tmp396 - tmp398
    tmp400 = tmp399.to(tl.int32)
    tmp401 = tmp400 == tmp400
    tmp402 = tmp399 - tmp401
    tmp403 = tmp402.to(tl.int32)
    tmp404 = tmp403 == tmp403
    tmp405 = tmp402 - tmp404
    tmp406 = tmp405.to(tl.int32)
    tmp407 = tmp406 == tmp406
    tmp408 = tmp405 - tmp407
    tmp409 = tmp408.to(tl.int32)
    tmp410 = tmp409 == tmp409
    tmp411 = tmp408 - tmp410
    tmp412 = tmp411.to(tl.int32)
    tmp413 = tmp412 == tmp412
    tmp414 = tmp411 - tmp413
    tmp415 = tmp414.to(tl.int32)
    tmp416 = tmp415 == tmp415
    tmp417 = tmp414 - tmp416
    tmp418 = tmp417.to(tl.int32)
    tmp419 = tmp418 == tmp418
    tmp420 = tmp417 - tmp419
    tmp421 = tmp420.to(tl.int32)
    tmp422 = tmp421 == tmp421
    tmp423 = tmp420 - tmp422
    tmp424 = tmp423.to(tl.int32)
    tmp425 = tmp424 == tmp424
    tmp426 = tmp423 - tmp425
    tmp427 = tmp426.to(tl.int32)
    tmp428 = tmp427 == tmp427
    tmp429 = tmp426 - tmp428
    tmp430 = tmp429.to(tl.int32)
    tmp431 = tmp430 == tmp430
    tmp432 = tmp429 - tmp431
    tmp433 = tmp432.to(tl.int32)
    tmp434 = tmp433 == tmp433
    tmp435 = tmp432 - tmp434
    tmp436 = tmp435.to(tl.int32)
    tmp437 = tmp436 == tmp436
    tmp438 = tmp435 - tmp437
    tmp439 = tmp438.to(tl.int32)
    tmp440 = tmp439 == tmp439
    tmp441 = tmp438 - tmp440
    tmp442 = tmp441.to(tl.int32)
    tmp443 = tmp442 == tmp442
    tmp444 = tmp441 - tmp443
    tmp445 = tmp444.to(tl.int32)
    tmp446 = tmp445 == tmp445
    tmp447 = tmp444 - tmp446
    tmp448 = tmp447.to(tl.int32)
    tmp449 = tmp448 == tmp448
    tmp450 = tmp447 - tmp449
    tmp451 = tmp450.to(tl.int32)
    tmp452 = tmp451 == tmp451
    tmp453 = tmp450 - tmp452
    tmp454 = tmp453.to(tl.int32)
    tmp455 = tmp454 == tmp454
    tmp456 = tmp453 - tmp455
    tmp457 = tmp456.to(tl.int32)
    tmp458 = tmp457 == tmp457
    tmp459 = tmp456 - tmp458
    tmp460 = tmp459.to(tl.int32)
    tmp461 = tmp460 == tmp460
    tmp462 = tmp459 - tmp461
    tmp463 = tmp462.to(tl.int32)
    tmp464 = tmp463 == tmp463
    tmp465 = tmp462 - tmp464
    tmp466 = tmp465.to(tl.int32)
    tmp467 = tmp466 == tmp466
    tmp468 = tmp465 - tmp467
    tmp469 = tmp468.to(tl.int32)
    tmp470 = tmp469 == tmp469
    tmp471 = tmp468 - tmp470
    tmp472 = tmp471.to(tl.int32)
    tmp473 = tmp472 == tmp472
    tmp474 = tmp471 - tmp473
    tmp475 = tmp474.to(tl.int32)
    tmp476 = tmp475 == tmp475
    tmp477 = tmp474 - tmp476
    tmp478 = tmp477.to(tl.int32)
    tmp479 = tmp478 == tmp478
    tmp480 = tmp477 - tmp479
    tmp481 = tmp480.to(tl.int32)
    tmp482 = tmp481 == tmp481
    tmp483 = tmp480 - tmp482
    tmp484 = tmp483.to(tl.int32)
    tmp485 = tmp484 == tmp484
    tmp486 = tmp483 - tmp485
    tmp487 = tmp486.to(tl.int32)
    tmp488 = tmp487 == tmp487
    tmp489 = tmp486 - tmp488
    tmp490 = tmp489.to(tl.int32)
    tmp491 = tmp490 == tmp490
    tmp492 = tmp489 - tmp491
    tmp493 = tmp492.to(tl.int32)
    tmp494 = tmp493 == tmp493
    tmp495 = tmp492 - tmp494
    tmp496 = tmp495.to(tl.int32)
    tmp497 = tmp496 == tmp496
    tmp498 = tmp495 - tmp497
    tmp499 = tmp498.to(tl.int32)
    tmp500 = tmp499 == tmp499
    tmp501 = tmp498 - tmp500
    tmp502 = tmp501.to(tl.int32)
    tmp503 = tmp502 == tmp502
    tmp504 = tmp501 - tmp503
    tmp505 = tmp504.to(tl.int32)
    tmp506 = tmp505 == tmp505
    tmp507 = tmp504 - tmp506
    tmp508 = tmp507.to
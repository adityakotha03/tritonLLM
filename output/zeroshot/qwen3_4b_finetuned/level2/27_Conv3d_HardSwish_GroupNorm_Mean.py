import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_hardtanh_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 6.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused__native_group_norm_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex % 16
    r2 = rindex // 16
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 16 * x0 + 256 * r2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 64.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr2 + x0, tmp23, xmask)
    tl.store(out_ptr0 + x0, tmp12, xmask)
    tl.store(out_ptr1 + x0, tmp18, xmask)


@triton.jit
def triton_poi_fused__native_group_norm_hardtanh_hardtanh_backward_2(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + 0)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr4 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + 1)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr1 + 1)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr3 + 2)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp23 = tl.load(in_ptr4 + (1 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr3 + 3)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp28 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr1 + 2)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp32 = tl.load(in_ptr2 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr3 + 4)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp37 = tl.load(in_ptr4 + (2 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp38 = tl.load(in_ptr3 + 5)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr1 + 3)
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp46 = tl.load(in_ptr2 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr3 + 6)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp51 = tl.load(in_ptr4 + (3 + 4 * x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr3 + 7)
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp14 + tmp16
    tmp13 = tmp8 + tmp12
    tmp21 = tmp18 + tmp20
    tmp22 = tmp13 + tmp21
    tmp26 = tmp28 + tmp30
    tmp27 = tmp22 + tmp26
    tmp31 = tmp32 + tmp34
    tmp35 = tmp27 + tmp31
    tmp40 = tmp42 + tmp44
    tmp41 = tmp35 + tmp40
    tmp45 = tmp46 + tmp48
    tmp49 = tmp41 + tmp45
    tmp50 = tmp9 + tmp11
    tmp54 = tmp37 + tmp39
    tmp55 = tmp49 + tmp54
    tmp56 = tmp51 + tmp53
    tmp57 = tmp55 + tmp56
    tmp58 = tmp50 + tmp57
    tmp59 = 64.0
    tmp60 = tmp58 / tmp59
    tmp61 = tmp8 - tmp60
    tmp62 = tmp61 * tmp61
    tmp63 = tmp12 - tmp60
    tmp64 = tmp63 * tmp63
    tmp65 = tmp62 + tmp64
    tmp66 = tmp21 - tmp60
    tmp67 = tmp66 * tmp66
    tmp68 = tmp65 + tmp67
    tmp69 = tmp26 - tmp60
    tmp70 = tmp69 * tmp69
    tmp71 = tmp68 + tmp70
    tmp72 = tmp31 - tmp60
    tmp73 = tmp72 * tmp72
    tmp74 = tmp71 + tmp73
    tmp75 = tmp35 - tmp60
    tmp76 = tmp75 * tmp75
    tmp77 = tmp74 + tmp76
    tmp78 = tmp40 - tmp60
    tmp79 = tmp78 * tmp78
    tmp80 = tmp77 + tmp79
    tmp81 = tmp45 - tmp60
    tmp82 = tmp81 * tmp81
    tmp83 = tmp80 + tmp82
    tmp84 = tmp54 - tmp60
    tmp85 = tmp84 * tmp84
    tmp86 = tmp83 + tmp85
    tmp87 = tmp56 - tmp60
    tmp88 = tmp87 * tmp87
    tmp89 = tmp86 + tmp88
    tmp90 = tmp57 - tmp60
    tmp91 = tmp90 * tmp90
    tmp92 = tmp89 + tmp91
    tmp93 = tmp58 / tmp59
    tmp94 = 1e-05
    tmp95 = tmp93 + tmp94
    tmp96 = libdevice.rsqrt(tmp95)
    tmp97 = tmp61 * tmp96
    tmp98 = 0.0
    tmp99 = tmp3 < tmp98
    tmp100 = tmp3 > tmp98
    tmp101 = tmp99 & tmp100
    tmp102 = 1.0
    tmp103 = tmp102 / tmp98
    tmp104 = tmp103 * tmp102
    tmp105 = tmp101 * tmp104
    tmp106 = tmp105 * tmp3
    tmp107 = tmp106 * tmp102
    tmp108 = tmp107 * tmp102
    tmp109 = tmp7 - tmp60
    tmp110 = tmp109 * tmp96
    tmp111 = tmp99 & tmp100
    tmp112 = tmp111 * tmp104
    tmp113 = tmp112 * tmp7
    tmp114 = tmp113 * tmp102
    tmp115 = tmp114 * tmp102
    tmp116 = tmp116 * tmp102
    tmp117 = tmp10 + tmp98
    tmp118 = tmp117 * tmp102
    tmp119 = tmp116 * tmp118
    tmp120 = tmp11 + tmp98
    tmp121 = tmp120 * tmp102
    tmp122 = tmp119 * tmp121
    tmp123 = tmp122 * tmp102
    tmp124 = tmp123 * tmp102
    tmp125 = tmp15 + tmp98
    tmp126 = tmp125 * tmp102
    tmp127 = tmp124 * tmp126
    tmp128 = tmp127 * tmp102
    tmp129 = tmp128 * tmp102
    tmp130 = tmp16 + tmp98
    tmp131 = tmp130 * tmp102
    tmp132 = tmp129 * tmp131
    tmp133 = tmp132 * tmp102
    tmp134 = tmp133 * tmp102
    tmp135 = tmp19 + tmp98
    tmp136 = tmp135 * tmp102
    tmp137 = tmp134 * tmp136
    tmp138 = tmp137 * tmp102
    tmp139 = tmp138 * tmp102
    tmp140 = tmp20 + tmp98
    tmp141 = tmp140 * tmp102
    tmp142 = tmp139 * tmp141
    tmp143 = tmp142 * tmp102
    tmp144 = tmp143 * tmp102
    tmp145 = tmp22 + tmp98
    tmp146 = tmp145 * tmp102
    tmp147 = tmp144 * tmp146
    tmp148 = tmp147 * tmp102
    tmp149 = tmp148 * tmp102
    tmp150 = tmp23 + tmp98
    tmp151 = tmp150 * tmp102
    tmp152 = tmp149 * tmp151
    tmp153 = tmp152 * tmp102
    tmp154 = tmp153 * tmp102
    tmp155 = tmp24 + tmp98
    tmp156 = tmp155 * tmp102
    tmp157 = tmp154 * tmp156
    tmp158 = tmp157 * tmp102
    tmp159 = tmp158 * tmp102
    tmp160 = tmp29 + tmp98
    tmp161 = tmp160 * tmp102
    tmp162 = tmp159 * tmp161
    tmp163 = tmp162 * tmp102
    tmp164 = tmp163 * tmp102
    tmp165 = tmp30 + tmp98
    tmp166 = tmp165 * tmp102
    tmp167 = tmp164 * tmp166
    tmp168 = tmp167 * tmp102
    tmp169 = tmp168 * tmp102
    tmp170 = tmp31 + tmp98
    tmp171 = tmp170 * tmp102
    tmp172 = tmp169 * tmp171
    tmp173 = tmp172 * tmp102
    tmp174 = tmp173 * tmp102
    tmp175 = tmp33 + tmp98
    tmp176 = tmp175 * tmp102
    tmp177 = tmp174 * tmp176
    tmp178 = tmp177 * tmp102
    tmp179 = tmp178 * tmp102
    tmp180 = tmp34 + tmp98
    tmp181 = tmp180 * tmp102
    tmp182 = tmp179 * tmp181
    tmp183 = tmp182 * tmp102
    tmp184 = tmp183 * tmp102
    tmp185 = tmp36 + tmp98
    tmp186 = tmp185 * tmp102
    tmp187 = tmp184 * tmp186
    tmp188 = tmp187 * tmp102
    tmp189 = tmp188 * tmp102
    tmp190 = tmp39 + tmp98
    tmp191 = tmp190 * tmp102
    tmp192 = tmp189 * tmp191
    tmp193 = tmp192 * tmp102
    tmp194 = tmp193 * tmp102
    tmp195 = tmp41 + tmp98
    tmp196 = tmp195 * tmp102
    tmp197 = tmp194 * tmp196
    tmp198 = tmp197 * tmp102
    tmp199 = tmp198 * tmp102
    tmp200 = tmp43 + tmp98
    tmp201 = tmp200 * tmp102
    tmp202 = tmp199 * tmp201
    tmp203 = tmp202 * tmp102
    tmp204 = tmp203 * tmp102
    tmp205 = tmp44 + tmp98
    tmp206 = tmp205 * tmp102
    tmp207 = tmp204 * tmp206
    tmp208 = tmp207 * tmp102
    tmp209 = tmp208 * tmp102
    tmp210 = tmp46 + tmp98
    tmp211 = tmp210 * tmp102
    tmp212 = tmp209 * tmp211
    tmp213 = tmp212 * tmp102
    tmp214 = tmp213 * tmp102
    tmp215 = tmp48 + tmp98
    tmp216 = tmp215 * tmp102
    tmp217 = tmp214 * tmp216
    tmp218 = tmp217 * tmp102
    tmp219 = tmp218 * tmp102
    tmp220 = tmp50 + tmp98
    tmp221 = tmp220 * tmp102
    tmp222 = tmp219 * tmp221
    tmp223 = tmp222 * tmp102
    tmp224 = tmp223 * tmp102
    tmp225 = tmp52 + tmp98
    tmp226 = tmp225 * tmp102
    tmp227 = tmp224 * tmp226
    tmp228 = tmp227 * tmp102
    tmp229 = tmp228 * tmp102
    tmp230 = tmp53 + tmp98
    tmp231 = tmp230 * tmp102
    tmp232 = tmp229 * tmp231
    tmp233 = tmp232 * tmp102
    tmp234 = tmp233 * tmp102
    tmp235 = tmp55 + tmp98
    tmp236 = tmp235 * tmp102
    tmp237 = tmp234 * tmp236
    tmp238 = tmp237 * tmp102
    tmp239 = tmp238 * tmp102
    tmp240 = tmp57 + tmp98
    tmp241 = tmp240 * tmp102
    tmp242 = tmp239 * tmp241
    tmp243 = tmp242 * tmp102
    tmp244 = tmp243 * tmp102
    tmp245 = tmp59 + tmp98
    tmp246 = tmp245 * tmp102
    tmp247 = tmp244 * tmp246
    tmp248 = tmp247 * tmp102
    tmp249 = tmp248 * tmp102
    tmp250 = tmp60 + tmp98
    tmp251 = tmp250 * tmp102
    tmp252 = tmp249 * tmp251
    tmp253 = tmp252 * tmp102
    tmp254 = tmp253 * tmp102
    tmp255 = tmp254 * tmp102
    tmp256 = tmp9 - tmp60
    tmp257 = tmp256 * tmp96
    tmp258 = tmp99 & tmp100
    tmp259 = tmp258 * tmp104
    tmp260 = tmp259 * tmp9
    tmp261 = tmp260 * tmp102
    tmp262 = tmp261 * tmp102
    tmp263 = tmp262 * tmp102
    tmp264 = tmp11 - tmp60
    tmp265 = tmp264 * tmp96
    tmp266 = tmp99 & tmp100
    tmp267 = tmp266 * tmp104
    tmp268 = tmp267 * tmp11
    tmp269 = tmp268 * tmp102
    tmp270 = tmp269 * tmp102
    tmp271 = tmp270 * tmp102
    tmp272 = tmp13 - tmp60
    tmp273 = tmp272 * tmp96
    tmp274 = tmp99 & tmp100
    tmp275 = tmp274 * tmp104
    tmp276 = tmp275 * tmp13
    tmp277 = tmp276 * tmp102
    tmp278 = tmp277 * tmp102
    tmp279 = tmp278 * tmp102
    tmp280 = tmp15 - tmp60
    tmp281 = tmp280 * tmp96
    tmp282 = tmp99 & tmp100
    tmp283 = tmp282 * tmp104
    tmp284 = tmp283 * tmp15
    tmp285 = tmp284 * tmp102
    tmp286 = tmp285 * tmp102
    tmp287 = tmp286 * tmp102
    tmp288 = tmp17 - tmp60
    tmp289 = tmp288 * tmp96
    tmp290 = tmp99 & tmp100
    tmp291 = tmp290 * tmp104
    tmp292 = tmp291 * tmp17
    tmp293 = tmp292 * tmp102
    tmp294 = tmp293 * tmp102
    tmp295 = tmp294 * tmp102
    tmp296 = tmp19 - tmp60
    tmp297 = tmp296 * tmp96
    tmp298 = tmp99 & tmp100
    tmp299 = tmp298 * tmp104
    tmp300 = tmp299 * tmp19
    tmp301 = tmp300 * tmp102
    tmp302 = tmp301 * tmp102
    tmp303 = tmp302 * tmp102
    tmp304 = tmp21 - tmp60
    tmp305 = tmp304 * tmp96
    tmp306 = tmp99 & tmp100
    tmp307 = tmp306 * tmp104
    tmp308 = tmp307 * tmp21
    tmp309 = tmp308 * tmp102
    tmp310 = tmp309 * tmp102
    tmp311 = tmp310 * tmp102
    tmp312 = tmp23 - tmp60
    tmp313 = tmp312 * tmp96
    tmp314 = tmp99 & tmp100
    tmp315 = tmp314 * tmp104
    tmp316 = tmp315 * tmp23
    tmp317 = tmp316 * tmp102
    tmp318 = tmp317 * tmp102
    tmp319 = tmp318 * tmp102
    tmp320 = tmp25 - tmp60
    tmp321 = tmp320 * tmp96
    tmp322 = tmp99 & tmp100
    tmp323 = tmp322 * tmp104
    tmp324 = tmp323 * tmp25
    tmp325 = tmp324 * tmp102
    tmp326 = tmp325 * tmp102
    tmp327 = tmp326 * tmp102
    tmp328 = tmp27 - tmp60
    tmp329 = tmp328 * tmp96
    tmp330 = tmp99 & tmp100
    tmp331 = tmp330 * tmp104
    tmp332 = tmp331 * tmp27
    tmp333 = tmp332 * tmp102
    tmp334 = tmp333 * tmp102
    tmp335 = tmp334 * tmp102
    tmp336 = tmp29 - tmp60
    tmp337 = tmp336 * tmp96
    tmp338 = tmp99 & tmp100
    tmp339 = tmp338 * tmp104
    tmp340 = tmp339 * tmp29
    tmp341 = tmp340 * tmp102
    tmp342 = tmp341 * tmp102
    tmp343 = tmp342 * tmp102
    tmp344 = tmp31 - tmp60
    tmp345 = tmp344 * tmp96
    tmp346 = tmp99 & tmp100
    tmp347 = tmp346 * tmp104
    tmp348 = tmp347 * tmp31
    tmp349 = tmp348 * tmp102
    tmp350 = tmp349 * tmp102
    tmp351 = tmp350 * tmp102
    tmp352 = tmp33 - tmp60
    tmp353 = tmp352 * tmp96
    tmp354 = tmp99 & tmp100
    tmp355 = tmp354 * tmp104
    tmp356 = tmp355 * tmp33
    tmp357 = tmp356 * tmp102
    tmp358 = tmp357 * tmp102
    tmp359 = tmp358 * tmp102
    tmp360 = tmp35 - tmp60
    tmp361 = tmp360 * tmp96
    tmp362 = tmp99 & tmp100
    tmp363 = tmp362 * tmp104
    tmp364 = tmp363 * tmp35
    tmp365 = tmp364 * tmp102
    tmp366 = tmp365 * tmp102
    tmp367 = tmp366 * tmp102
    tmp368 = tmp37 - tmp60
    tmp369 = tmp368 * tmp96
    tmp370 = tmp99 & tmp100
    tmp371 = tmp370 * tmp104
    tmp372 = tmp371 * tmp37
    tmp373 = tmp372 * tmp102
    tmp374 = tmp373 * tmp102
    tmp375 = tmp374 * tmp102
    tmp376 = tmp39 - tmp60
    tmp377 = tmp376 * tmp96
    tmp378 = tmp99 & tmp100
    tmp379 = tmp378 * tmp104
    tmp380 = tmp379 * tmp39
    tmp381 = tmp380 * tmp102
    tmp382 = tmp381 * tmp102
    tmp383 = tmp382 * tmp102
    tmp384 = tmp41 - tmp60
    tmp385 = tmp384 * tmp96
    tmp386 = tmp99 & tmp100
    tmp387 = tmp386 * tmp104
    tmp388 = tmp387 * tmp41
    tmp389 = tmp388 * tmp102
    tmp390 = tmp389 * tmp102
    tmp391 = tmp390 * tmp102
    tmp392 = tmp43 - tmp60
    tmp393 = tmp392 * tmp96
    tmp394 = tmp99 & tmp100
    tmp395 = tmp394 * tmp104
    tmp396 = tmp395 * tmp43
    tmp397 = tmp396 * tmp102
    tmp398 = tmp397 * tmp102
    tmp399 = tmp398 * tmp102
    tmp400 = tmp44 - tmp60
    tmp401 = tmp400 * tmp96
    tmp402 = tmp99 & tmp100
    tmp403 = tmp402 * tmp104
    tmp404 = tmp403 * tmp44
    tmp405 = tmp404 * tmp102
    tmp406 = tmp405 * tmp102
    tmp407 = tmp406 * tmp102
    tmp408 = tmp46 - tmp60
    tmp409 = tmp408 * tmp96
    tmp410 = tmp99 & tmp100
    tmp411 = tmp410 * tmp104
    tmp412 = tmp411 * tmp46
    tmp413 = tmp412 * tmp102
    tmp414 = tmp413 * tmp102
    tmp415 = tmp414 * tmp102
    tmp416 = tmp48 - tmp60
    tmp417 = tmp416 * tmp96
    tmp418 = tmp99 & tmp100
    tmp419 = tmp418 * tmp104
    tmp420 = tmp419 * tmp48
    tmp421 = tmp420 * tmp102
    tmp422 = tmp421 * tmp102
    tmp423 = tmp422 * tmp102
    tmp424 = tmp50 - tmp60
    tmp425 = tmp424 * tmp96
    tmp426 = tmp99 & tmp100
    tmp427 = tmp426 * tmp104
    tmp428 = tmp427 * tmp50
    tmp429 = tmp428 * tmp102
    tmp430 = tmp429 * tmp102
    tmp431 = tmp430 * tmp102
    tmp432 = tmp52 - tmp60
    tmp433 = tmp432 * tmp96
    tmp434 = tmp99 & tmp100
    tmp435 = tmp434 * tmp104
    tmp436 = tmp435 * tmp52
    tmp437 = tmp436 * tmp102
    tmp438 = tmp437 * tmp102
    tmp439 = tmp438 * tmp102
    tmp440 = tmp53 - tmp60
    tmp441 = tmp440 * tmp96
    tmp442 = tmp99 & tmp100
    tmp443 = tmp442 * tmp104
    tmp444 = tmp443 * tmp53
    tmp445 = tmp444 * tmp102
    tmp446 = tmp445 * tmp102
    tmp447 = tmp446 * tmp102
    tmp448 = tmp55 - tmp60
    tmp449 = tmp448 * tmp96
    tmp450 = tmp99 & tmp100
    tmp451 = tmp450 * tmp104
    tmp452 = tmp451 * tmp55
    tmp453 = tmp452 * tmp102
    tmp454 = tmp453 * tmp102
    tmp455 = tmp454 * tmp102
    tmp456 = tmp57 - tmp60
    tmp457 = tmp456 * tmp96
    tmp458 = tmp99 & tmp100
    tmp459 = tmp458 * tmp104
    tmp460 = tmp459 * tmp57
    tmp461 = tmp460 * tmp102
    tmp462 = tmp461 * tmp102
    tmp463 = tmp462 * tmp102
    tmp464 = tmp59 - tmp60
    tmp465 = tmp464 * tmp96
    tmp466 = tmp99 & tmp100
    tmp
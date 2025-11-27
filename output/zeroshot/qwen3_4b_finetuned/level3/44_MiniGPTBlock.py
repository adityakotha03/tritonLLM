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
import math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (1025 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (2049 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (3073 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (1026 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (2050 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (3074 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (1027 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (2051 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (3075 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (4 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (1028 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (2052 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (3076 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (5 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (1029 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (2053 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (3077 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (6 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (1030 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (2054 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (3078 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (7 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (1031 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (2055 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (3079 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (8 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (1032 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (2056 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (3080 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (9 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (1033 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (2057 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (3081 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tmp1 + tmp3
    tmp21 = tmp1 + tmp5
    tmp23 = tmp3 + tmp5
    tmp25 = tmp1 + tmp19
    tmp26 = tmp25 + tmp3
    tmp27 = tmp25 + tmp5
    tmp28 = tmp26 + tmp19
    tmp30 = tmp28 + tmp3
    tmp31 = tmp28 + tmp5
    tmp32 = tmp30 + tmp19
    tmp33 = tmp32 + tmp3
    tmp34 = tmp32 + tmp5
    tmp35 = tmp33 + tmp19
    tmp36 = tmp35 + tmp3
    tmp37 = tmp35 + tmp5
    tmp38 = tmp36 + tmp19
    tmp39 = tmp38 + tmp3
    tmp40 = tmp38 + tmp5
    tmp41 = tmp39 + tmp19
    tmp42 = tmp41 + tmp3
    tmp43 = tmp41 + tmp5
    tmp44 = tmp42 + tmp19
    tmp45 = tmp44 + tmp3
    tmp46 = tmp44 + tmp5
    tmp47 = tmp45 + tmp19
    tmp48 = tmp47 + tmp3
    tmp49 = tmp47 + tmp5
    tmp50 = tmp48 + tmp19
    tmp51 = tmp50 + tmp3
    tmp52 = tmp50 + tmp5
    tmp53 = tmp51 + tmp19
    tmp54 = tmp53 + tmp3
    tmp55 = tmp53 + tmp5
    tmp56 = tmp54 + tmp19
    tmp57 = tmp56 + tmp3
    tmp58 = tmp56 + tmp5
    tmp59 = tmp57 + tmp19
    tmp60 = tmp59 + tmp3
    tmp61 = tmp59 + tmp5
    tmp62 = tmp60 + tmp19
    tmp63 = tmp62 + tmp3
    tmp64 = tmp62 + tmp5
    tmp65 = tmp63 + tmp19
    tmp66 = tmp65 + tmp3
    tmp67 = tmp65 + tmp5
    tmp68 = tmp66 + tmp19
    tmp69 = tmp68 + tmp3
    tmp70 = tmp68 + tmp5
    tmp71 = tmp69 + tmp19
    tmp72 = tmp71 + tmp3
    tmp73 = tmp71 + tmp5
    tmp74 = tmp72 + tmp19
    tmp75 = tmp74 + tmp3
    tmp76 = tmp74 + tmp5
    tmp77 = tmp75 + tmp19
    tmp78 = tmp77 + tmp3
    tmp79 = tmp77 + tmp5
    tmp80 = tmp78 + tmp19
    tmp81 = tmp80 + tmp3
    tmp82 = tmp80 + tmp5
    tmp83 = tmp81 + tmp19
    tmp84 = tmp83 + tmp3
    tmp85 = tmp83 + tmp5
    tmp86 = tmp84 + tmp19
    tmp87 = tmp86 + tmp3
    tmp88 = tmp86 + tmp5
    tmp89 = tmp87 + tmp19
    tmp90 = tmp89 + tmp3
    tmp91 = tmp89 + tmp5
    tmp92 = tmp90 + tmp19
    tmp93 = tmp92 + tmp3
    tmp94 = tmp92 + tmp5
    tmp95 = tmp93 + tmp19
    tmp96 = tmp94 + tmp3
    tmp97 = tmp94 + tmp5
    tmp98 = tmp95 + tmp19
    tmp99 = tmp98 + tmp3
    tmp100 = tmp98 + tmp5
    tmp101 = tmp99 + tmp19
    tmp102 = tmp101 + tmp3
    tmp103 = tmp101 + tmp5
    tmp104 = tmp102 + tmp19
    tmp105 = tmp104 + tmp3
    tmp106 = tmp104 + tmp5
    tmp107 = tmp105 + tmp19
    tmp108 = tmp107 + tmp3
    tmp109 = tmp107 + tmp5
    tmp110 = tmp108 + tmp19
    tmp111 = tmp110 + tmp3
    tmp112 = tmp110 + tmp5
    tmp113 = tmp111 + tmp19
    tmp114 = tmp112 + tmp3
    tmp115 = tmp112 + tmp5
    tmp116 = tmp113 + tmp19
    tmp117 = tmp115 + tmp3
    tmp118 = tmp115 + tmp5
    tmp119 = tmp116 + tmp19
    tmp120 = tmp119 + tmp3
    tmp121 = tmp119 + tmp5
    tmp122 = tmp120 + tmp19
    tmp123 = tmp122 + tmp3
    tmp124 = tmp122 + tmp5
    tmp125 = tmp123 + tmp19
    tmp126 = tmp125 + tmp3
    tmp127 = tmp125 + tmp5
    tmp128 = tmp126 + tmp19
    tmp129 = tmp128 + tmp3
    tmp130 = tmp128 + tmp5
    tmp131 = tmp129 + tmp19
    tmp132 = tmp131 + tmp3
    tmp133 = tmp131 + tmp5
    tmp134 = tmp132 + tmp19
    tmp135 = tmp133 + tmp3
    tmp136 = tmp133 + tmp5
    tmp137 = tmp134 + tmp19
    tmp138 = tmp136 + tmp3
    tmp139 = tmp136 + tmp5
    tmp140 = tmp137 + tmp19
    tmp141 = tmp140 + tmp3
    tmp142 = tmp140 + tmp5
    tmp143 = tmp141 + tmp19
    tmp144 = tmp142 + tmp3
    tmp145 = tmp142 + tmp5
    tmp146 = tmp143 + tmp19
    tmp147 = tmp146 + tmp3
    tmp148 = tmp146 + tmp5
    tmp149 = tmp147 + tmp19
    tmp150 = tmp148 + tmp3
    tmp151 = tmp148 + tmp5
    tmp152 = tmp150 + tmp19
    tmp153 = tmp151 + tmp3
    tmp154 = tmp151 + tmp5
    tmp155 = tmp152 + tmp19
    tmp156 = tmp154 + tmp3
    tmp157 = tmp154 + tmp5
    tmp158 = tmp155 + tmp19
    tmp159 = tmp158 + tmp3
    tmp160 = tmp158 + tmp5
    tmp161 = tmp159 + tmp19
    tmp162 = tmp160 + tmp3
    tmp163 = tmp160 + tmp5
    tmp164 = tmp161 + tmp19
    tmp165 = tmp163 + tmp3
    tmp166 = tmp163 + tmp5
    tmp167 = tmp164 + tmp19
    tmp168 = tmp166 + tmp3
    tmp169 = tmp166 + tmp5
    tmp170 = tmp167 + tmp19
    tmp171 = tmp169 + tmp3
    tmp172 = tmp169 + tmp5
    tmp173 = tmp170 + tmp19
    tmp174 = tmp172 + tmp3
    tmp175 = tmp172 + tmp5
    tmp176 = tmp173 + tmp19
    tmp177 = tmp175 + tmp3
    tmp178 = tmp175 + tmp5
    tmp179 = tmp176 + tmp19
    tmp180 = tmp178 + tmp3
    tmp181 = tmp178 + tmp5
    tmp182 = tmp179 + tmp19
    tmp183 = tmp181 + tmp3
    tmp184 = tmp181 + tmp5
    tmp185 = tmp182 + tmp19
    tmp186 = tmp184 + tmp3
    tmp187 = tmp184 + tmp5
    tmp188 = tmp185 + tmp19
    tmp189 = tmp187 + tmp3
    tmp190 = tmp187 + tmp5
    tmp191 = tmp188 + tmp19
    tmp192 = tmp190 + tmp3
    tmp193 = tmp190 + tmp5
    tmp194 = tmp191 + tmp19
    tmp195 = tmp193 + tmp3
    tmp196 = tmp193 + tmp5
    tmp197 = tmp194 + tmp19
    tmp198 = tmp196 + tmp3
    tmp199 = tmp196 + tmp5
    tmp200 = tmp197 + tmp19
    tmp201 = tmp199 + tmp3
    tmp202 = tmp199 + tmp5
    tmp203 = tmp200 + tmp19
    tmp204 = tmp202 + tmp3
    tmp205 = tmp202 + tmp5
    tmp206 = tmp203 + tmp19
    tmp207 = tmp205 + tmp3
    tmp208 = tmp205 + tmp5
    tmp209 = tmp206 + tmp19
    tmp210 = tmp208 + tmp3
    tmp211 = tmp208 + tmp5
    tmp212 = tmp209 + tmp19
    tmp213 = tmp211 + tmp3
    tmp214 = tmp211 + tmp5
    tmp215 = tmp212 + tmp19
    tmp216 = tmp214 + tmp3
    tmp217 = tmp214 + tmp5
    tmp218 = tmp215 + tmp19
    tmp219 = tmp217 + tmp3
    tmp220 = tmp217 + tmp5
    tmp221 = tmp218 + tmp19
    tmp222 = tmp219 + tmp3
    tmp223 = tmp219 + tmp5
    tmp224 = tmp220 + tmp19
    tmp225 = tmp222 + tmp3
    tmp226 = tmp222 + tmp5
    tmp227 = tmp223 + tmp19
    tmp228 = tmp225 + tmp3
    tmp229 = tmp225 + tmp5
    tmp230 = tmp226 + tmp19
    tmp231 = tmp228 + tmp3
    tmp232 = tmp228 + tmp5
    tmp233 = tmp229 + tmp19
    tmp234 = tmp231 + tmp3
    tmp235 = tmp231 + tmp5
    tmp236 = tmp232 + tmp19
    tmp237 = tmp234 + tmp3
    tmp238 = tmp234 + tmp5
    tmp239 = tmp235 + tmp19
    tmp240 = tmp237 + tmp3
    tmp241 = tmp237 + tmp5
    tmp242 = tmp238 + tmp19
    tmp243 = tmp240 + tmp3
    tmp244 = tmp238 + tmp5
    tmp245 = tmp241 + tmp19
    tmp246 = tmp243 + tmp3
    tmp247 = tmp243 + tmp5
    tmp248 = tmp244 + tmp19
    tmp249 = tmp246 + tmp3
    tmp250 = tmp246 + tmp5
    tmp251 = tmp247 + tmp19
    tmp252 = tmp249 + tmp3
    tmp253 = tmp249 + tmp5
    tmp254 = tmp250 + tmp19
    tmp255 = tmp252 + tmp3
    tmp256 = tmp252 + tmp5
    tmp257 = tmp253 + tmp19
    tmp258 = tmp255 + tmp3
    tmp259 = tmp255 + tmp5
    tmp260 = tmp256 + tmp19
    tmp261 = tmp258 + tmp3
    tmp262 = tmp258 + tmp5
    tmp263 = tmp259 + tmp19
    tmp264 = tmp261 + tmp3
    tmp265 = tmp261 + tmp5
    tmp266 = tmp262 + tmp19
    tmp267 = tmp264 + tmp3
    tmp268 = tmp264 + tmp5
    tmp269 = tmp265 + tmp19
    tmp270 = tmp267 + tmp3
    tmp271 = tmp267 + tmp5
    tmp272 = tmp268 + tmp19
    tmp273 = tmp270 + tmp3
    tmp274 = tmp268 + tmp5
    tmp275 = tmp271 + tmp19
    tmp276 = tmp273 + tmp3
    tmp277 = tmp273 + tmp5
    tmp278 = tmp274 + tmp19
    tmp279 = tmp276 + tmp3
    tmp280 = tmp276 + tmp5
    tmp281 = tmp277 + tmp19
    tmp282 = tmp279 + tmp3
    tmp283 = tmp279 + tmp5
    tmp284 = tmp280 + tmp19
    tmp285 = tmp282 + tmp3
    tmp286 = tmp282 + tmp5
    tmp287 = tmp283 + tmp19
    tmp288 = tmp285 + tmp3
    tmp289 = tmp285 + tmp5
    tmp290 = tmp286 + tmp19
    tmp291 = tmp288 + tmp3
    tmp292 = tmp288 + tmp5
    tmp293 = tmp289 + tmp19
    tmp294 = tmp291 + tmp3
    tmp295 = tmp291 + tmp5
    tmp296 = tmp292 + tmp19
    tmp297 = tmp294 + tmp3
    tmp298 = tmp294 + tmp5
    tmp299 = tmp295 + tmp19
    tmp300 = tmp297 + tmp3
    tmp301 = tmp297 + tmp5
    tmp302 = tmp298 + tmp19
    tmp303 = tmp300 + tmp3
    tmp304 = tmp300 + tmp5
    tmp305 = tmp301 + tmp19
    tmp306 = tmp303 + tmp3
    tmp307 = tmp303 + tmp5
    tmp308 = tmp304 + tmp19
    tmp309 = tmp306 + tmp3
    tmp310 = tmp306 + tmp5
    tmp311 = tmp307 + tmp19
    tmp312 = tmp309 + tmp3
    tmp313 = tmp309 + tmp5
    tmp314 = tmp310 + tmp19
    tmp315 = tmp312 + tmp3
    tmp316 = tmp312 + tmp5
    tmp317 = tmp313 + tmp19
    tmp318 = tmp315 + tmp3
    tmp319 = tmp315 + tmp5
    tmp320 = tmp316 + tmp19
    tmp321 = tmp318 + tmp3
    tmp322 = tmp318 + tmp5
    tmp323 = tmp319 + tmp19
    tmp324 = tmp321 + tmp3
    tmp325 = tmp321 + tmp5
    tmp326 = tmp322 + tmp19
    tmp327 = tmp324 + tmp3
    tmp328 = tmp324 + tmp5
    tmp329 = tmp325 + tmp19
    tmp330 = tmp327 + tmp3
    tmp331 = tmp327 + tmp5
    tmp332 = tmp328 + tmp19
    tmp333 = tmp330 + tmp3
    tmp334 = tmp330 + tmp5
    tmp335 = tmp329 + tmp19
    tmp336 = tmp333 + tmp3
    tmp337 = tmp333 + tmp5
    tmp338 = tmp334 + tmp19
    tmp339 = tmp336 + tmp3
    tmp340 = tmp336 + tmp5
    tmp341 = tmp337 + tmp19
    tmp342 = tmp339 + tmp3
    tmp343 = tmp339 + tmp5
    tmp344 = tmp340 + tmp19
    tmp345 = tmp342 + tmp3
    tmp346 = tmp342 + tmp5
    tmp347 = tmp343 + tmp19
    tmp348 = tmp345 + tmp3
    tmp349 = tmp345 + tmp5
    tmp350 = tmp346 + tmp19
    tmp351 = tmp348 + tmp3
    tmp352 = tmp348 + tmp5
    tmp353 = tmp349 + tmp19
    tmp354 = tmp351 + tmp3
    tmp355 = tmp351 + tmp5
    tmp356 = tmp352 + tmp19
    tmp357 = tmp354 + tmp3
    tmp358 = tmp354 + tmp5
    tmp359 = tmp355 + tmp19
    tmp360 = tmp357 + tmp3
    tmp361 = tmp357 + tmp5
    tmp362 = tmp358 + tmp19
    tmp363 = tmp360 + tmp3
    tmp364 = tmp360 + tmp5
    tmp365 = tmp361 + tmp19
    tmp366 = tmp363 + tmp3
    tmp367 = tmp363 + tmp5
    tmp368 = tmp364 + tmp19
    tmp369 = tmp366 + tmp3
    tmp370 = tmp366 + tmp5
    tmp371 = tmp367 + tmp19
    tmp372 = tmp369 + tmp3
    tmp373 = tmp369 + tmp5
    tmp374 = tmp370 + tmp19
    tmp375 = tmp372 + tmp3
    tmp376 = tmp372 + tmp5
    tmp377 = tmp373 + tmp19
    tmp378 = tmp375 + tmp3
    tmp379 = tmp375 + tmp5
    tmp380 = tmp376 + tmp19
    tmp381 = tmp378 + tmp3
    tmp382 = tmp378 + tmp5
    tmp383 = tmp379 + tmp19
    tmp384 = tmp381 + tmp3
    tmp385 = tmp381 + tmp5
    tmp386 = tmp382 + tmp19
    tmp387 = tmp384 + tmp3
    tmp388 = tmp384 + tmp5
    tmp389 = tmp385 + tmp19
    tmp390 = tmp387 + tmp3
    tmp391 = tmp387 + tmp5
    tmp392 = tmp388 + tmp19
    tmp393 = tmp390 + tmp3
    tmp394 = tmp390 + tmp5
    tmp395 = tmp389 + tmp19
    tmp396 = tmp393 + tmp3
    tmp397 = tmp393 + tmp5
    tmp398 = tmp394 + tmp19
    tmp399 = tmp396 + tmp3
    tmp400 = tmp396 + tmp5
    tmp401 = tmp397 + tmp19
    tmp402 = tmp399 + tmp3
    tmp403 = tmp399 + tmp5
    tmp404 = tmp400 + tmp19
    tmp405 = tmp402 + tmp3
    tmp406 = tmp402 + tmp5
    tmp407 = tmp403 + tmp19
    tmp408 = tmp405 + tmp3
    tmp409 = tmp405 + tmp5
    tmp410 = tmp406 + tmp19
    tmp411 = tmp408 + tmp3
    tmp412 = tmp408 + tmp5
    tmp413 = tmp409 + tmp19
    tmp414 = tmp411 + tmp3
    tmp415 = tmp411 + tmp5
    tmp416 = tmp412 + tmp19
    tmp417 = tmp414 + tmp3
    tmp418 = tmp414 + tmp5
    tmp419 = tmp415 + tmp19
    tmp420 = tmp417 + tmp3
    tmp421 = tmp417 + tmp5
    tmp422 = tmp418 + tmp19
    tmp423 = tmp420 + tmp3
    tmp424 = tmp420 + tmp5
    tmp425 = tmp421 + tmp19
    tmp426 = tmp423 + tmp3
    tmp427 = tmp423 + tmp5
    tmp428 = tmp424 + tmp19
    tmp429 = tmp426 + tmp3
    tmp430 = tmp426 + tmp5
    tmp431 = tmp427 + tmp19
    tmp432 = tmp429 + tmp3
    tmp433 = tmp429 + tmp5
    tmp434 = tmp430 + tmp19
    tmp435 = tmp432 + tmp3
    tmp436 = tmp432 + tmp5
    tmp437 = tmp433 + tmp19
    tmp438 = tmp435 + tmp3
    tmp439 = tmp435 + tmp5
    tmp440 = tmp436 + tmp19
    tmp441 = tmp438 + tmp3
    tmp442 = tmp438 + tmp5
    tmp443 = tmp439 + tmp19
    tmp444 = tmp441 + tmp3
    tmp445 = tmp441 + tmp5
    tmp446 = tmp442 + tmp19
    tmp447 = tmp444 + tmp3
    tmp448 = tmp444 + tmp5
    tmp449 = tmp443 + tmp19
    tmp450 = tmp447 + tmp3
    tmp451 = tmp447 + tmp5
    tmp452 = tmp448 + tmp19
    tmp453 = tmp450 + tmp3
    tmp454 = tmp450 + tmp5
    tmp455 = tmp451 + tmp19
    tmp456 = tmp453 + tmp3
    tmp457 = tmp453 + tmp5
    tmp458 = tmp454 + tmp19
    tmp459 = tmp456 + tmp3
    tmp460 = tmp456 + tmp5
    tmp461 = tmp457 + tmp19
    tmp462 = tmp459 + tmp3
    tmp463 = tmp459 + tmp5
    tmp464 = tmp460 + tmp19
    tmp465 = tmp462 + tmp3
    tmp466 = tmp462 + tmp5
    tmp467 = tmp463 + tmp19
    tmp468 = tmp465 + tmp3
    tmp469 = tmp465 + tmp5
    tmp470 = tmp466 + tmp19
    tmp471 = tmp468 + tmp3
    tmp472 = tmp468 + tmp5
    tmp473 = tmp469 + tmp19
    tmp474 = tmp471 + tmp3
    tmp475 = tmp471 + tmp5
    tmp476 = tmp472 + tmp19
    tmp477 = tmp474 + tmp3
    tmp478 = tmp474 + tmp5
    tmp479 = tmp475 + tmp19
    tmp480 = tmp477 + tmp3
    tmp481 = tmp477 + tmp5
    tmp482 = tmp478 + tmp19
    tmp483 = tmp
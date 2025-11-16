import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.11489042502622129
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 12777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (2048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (4096 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (5120 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (6144 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (7168 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8192 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (9216 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10240 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11264 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12288 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13312 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14336 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15360 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16384 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17408 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18432 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19456 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20480 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21504 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22528 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23552 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24576 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25600 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26624 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27648 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28672 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29696 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30720 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31744 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (32768 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (33792 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (34816 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (35840 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (36864 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (37888 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (38912 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (39936 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (40960 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (41984 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (43008 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (44032 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (45056 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (46080 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (47104 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (48128 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (49152 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (50176 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (51200 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (52224 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (53248 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (54272 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (55296 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (56320 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (57344 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (58368 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (59392 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (60416 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (61440 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (62464 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (63488 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (64512 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (65536 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (66560 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (67584 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (68608 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (69632 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (70656 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (71680 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (72704 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (73728 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (74752 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (75776 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (76800 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (77824 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (78848 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (79872 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (80896 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (81920 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (82944 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (83968 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (84992 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (86016 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (87040 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (88064 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (89088 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (90112 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (91136 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (92160 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (93184 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (94208 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (95232 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (96256 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (97280 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (98304 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (99328 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (100352 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (101376 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (102400 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (103424 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (104448 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (105472 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (106496 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (107520 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (108544 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (109568 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (110592 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (111616 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (112640 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (113664 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (114688 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (115712 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (116736 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (117760 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (118784 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (119808 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (120832 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (121856 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (122880 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (123904 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (124928 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (125952 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (126976 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (127999 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp251 = tmp0 + tmp1
    tmp252 = tmp251 + tmp2
    tmp253 = tmp252 + tmp4
    tmp254 = tmp253 + tmp6
    tmp255 = tmp254 + tmp8
    tmp256 = tmp255 + tmp10
    tmp257 = tmp256 + tmp12
    tmp258 = tmp257 + tmp14
    tmp259 = tmp258 + tmp16
    tmp260 = tmp259 + tmp18
    tmp261 = tmp260 + tmp20
    tmp262 = tmp261 + tmp22
    tmp263 = tmp262 + tmp24
    tmp264 = tmp263 + tmp26
    tmp265 = tmp264 + tmp28
    tmp266 = tmp265 + tmp30
    tmp267 = tmp266 + tmp32
    tmp268 = tmp267 + tmp34
    tmp269 = tmp268 + tmp36
    tmp270 = tmp269 + tmp38
    tmp271 = tmp270 + tmp40
    tmp272 = tmp271 + tmp42
    tmp273 = tmp272 + tmp44
    tmp274 = tmp273 + tmp46
    tmp275 = tmp274 + tmp48
    tmp276 = tmp275 + tmp50
    tmp277 = tmp276 + tmp52
    tmp278 = tmp277 + tmp54
    tmp279 = tmp278 + tmp56
    tmp280 = tmp279 + tmp58
    tmp281 = tmp280 + tmp60
    tmp282 = tmp281 + tmp62
    tmp283 = tmp282 + tmp64
    tmp284 = tmp283 + tmp66
    tmp285 = tmp284 + tmp68
    tmp286 = tmp285 + tmp70
    tmp287 = tmp286 + tmp72
    tmp288 = tmp287 + tmp74
    tmp289 = tmp288 + tmp76
    tmp290 = tmp289 + tmp78
    tmp291 = tmp290 + tmp80
    tmp292 = tmp291 + tmp82
    tmp293 = tmp292 + tmp84
    tmp294 = tmp293 + tmp86
    tmp295 = tmp294 + tmp88
    tmp296 = tmp295 + tmp90
    tmp297 = tmp296 + tmp92
    tmp298 = tmp297 + tmp94
    tmp299 = tmp298 + tmp96
    tmp300 = tmp299 + tmp98
    tmp301 = tmp300 + tmp100
    tmp302 = tmp301 + tmp102
    tmp303 = tmp302 + tmp104
    tmp304 = tmp303 + tmp106
    tmp305 = tmp304 + tmp108
    tmp306 = tmp305 + tmp110
    tmp307 = tmp306 + tmp112
    tmp308 = tmp307 + tmp114
    tmp309 = tmp308 + tmp116
    tmp310 = tmp309 + tmp118
    tmp311 = tmp310 + tmp120
    tmp312 = tmp311 + tmp122
    tmp313 = tmp312 + tmp124
    tmp314 = tmp313 + tmp126
    tmp315 = tmp314 + tmp128
    tmp316 = tmp315 + tmp130
    tmp317 = tmp316 + tmp132
    tmp318 = tmp317 + tmp134
    tmp319 = tmp318 + tmp136
    tmp320 = tmp319 + tmp138
    tmp321 = tmp320 + tmp140
    tmp322 = tmp321 + tmp142
    tmp323 = tmp322 + tmp144
    tmp324 = tmp323 + tmp146
    tmp325 = tmp324 + tmp148
    tmp326 = tmp325 + tmp150
    tmp327 = tmp326 + tmp152
    tmp328 = tmp327 + tmp154
    tmp329 = tmp328 + tmp156
    tmp330 = tmp329 + tmp158
    tmp331 = tmp330 + tmp160
    tmp332 = tmp331 + tmp162
    tmp333 = tmp332 + tmp164
    tmp334 = tmp333 + tmp166
    tmp335 = tmp334 + tmp168
    tmp336 = tmp335 + tmp170
    tmp337 = tmp336 + tmp172
    tmp338 = tmp337 + tmp174
    tmp339 = tmp338 + tmp176
    tmp340 = tmp339 + tmp178
    tmp341 = tmp340 + tmp180
    tmp342 = tmp341 + tmp182
    tmp343 = tmp342 + tmp184
    tmp344 = tmp343 + tmp186
    tmp345 = tmp344 + tmp188
    tmp346 = tmp345 + tmp190
    tmp347 = tmp346 + tmp192
    tmp348 = tmp347 + tmp194
    tmp349 = tmp348 + tmp196
    tmp350 = tmp349 + tmp198
    tmp351 = tmp350 + tmp200
    tmp352 = tmp351 + tmp202
    tmp353 = tmp352 + tmp204
    tmp354 = tmp353 + tmp206
    tmp355 = tmp354 + tmp208
    tmp356 = tmp355 + tmp210
    tmp357 = tmp356 + tmp212
    tmp358 = tmp357 + tmp214
    tmp359 = tmp358 + tmp216
    tmp360 = tmp359 + tmp218
    tmp361 = tmp360 + tmp220
    tmp362 = tmp361 + tmp222
    tmp363 = tmp362 + tmp224
    tmp364 = tmp363 + tmp226
    tmp365 = tmp364 + tmp228
    tmp366 = tmp365 + tmp230
    tmp367 = tmp366 + tmp232
    tmp368 = tmp367 + tmp234
    tmp369 = tmp368 + tmp236
    tmp370 = tmp369 + tmp238
    tmp371 = tmp370 + tmp240
    tmp372 = tmp371 + tmp242
    tmp373 = tmp372 + tmp244
    tmp374 = tmp373 + tmp246
    tmp375 = tmp374 + tmp248
    tmp376 = tmp375 + tmp250
    tmp377 = tmp376 + tmp251
    tmp378 = 1.0
    tmp379 = tmp378 / tmp377
    tmp380 = tmp251 * tmp379
    tmp381 = tmp2 * tmp379
    tmp382 = tmp380 + tmp381
    tmp383 = tmp4 * tmp379
    tmp384 = tmp382 + tmp383
    tmp385 = tmp6 * tmp379
    tmp386 = tmp384 + tmp385
    tmp387 = tmp8 * tmp379
    tmp388 = tmp386 + tmp387
    tmp389 = tmp10 * tmp379
    tmp390 = tmp388 + tmp389
    tmp391 = tmp12 * tmp379
    tmp392 = tmp390 + tmp391
    tmp393 = tmp14 * tmp379
    tmp394 = tmp392 + tmp393
    tmp395 = tmp16 * tmp379
    tmp396 = tmp394 + tmp395
    tmp397 = tmp18 * tmp379
    tmp398 = tmp396 + tmp397
    tmp399 = tmp20 * tmp379
    tmp400 = tmp398 + tmp399
    tmp401 = tmp22 * tmp379
    tmp402 = tmp400 + tmp401
    tmp403 = tmp24 * tmp379
    tmp404 = tmp402 + tmp403
    tmp405 = tmp26 * tmp379
    tmp406 = tmp404 + tmp405
    tmp407 = tmp28 * tmp379
    tmp408 = tmp406 + tmp407
    tmp409 = tmp30 * tmp379
    tmp410 = tmp408 + tmp409
    tmp411 = tmp32 * tmp379
    tmp412 = tmp410 + tmp411
    tmp413 = tmp34 * tmp379
    tmp414 = tmp412 + tmp413
    tmp415 = tmp36 * tmp379
    tmp416 = tmp414 + tmp415
    tmp417 = tmp38 * tmp379
    tmp418 = tmp416 + tmp417
    tmp419 = tmp40 * tmp379
    tmp420 = tmp418 + tmp419
    tmp421 = tmp42 * tmp379
    tmp422 = tmp420 + tmp421
    tmp423 = tmp44 * tmp379
    tmp424 = tmp422 + tmp423
    tmp425 = tmp46 * tmp379
    tmp426 = tmp424 + tmp425
    tmp427 = tmp48 * tmp379
    tmp428 = tmp426 + tmp427
    tmp429 = tmp50 * tmp379
    tmp430 = tmp428 + tmp429
    tmp431 = tmp52 * tmp379
    tmp432 = tmp430 + tmp431
    tmp433 = tmp54 * tmp379
    tmp434 = tmp432 + tmp433
    tmp435 = tmp56 * tmp379
    tmp436 = tmp434 + tmp435
    tmp437 = tmp58 * tmp379
    tmp438 = tmp436 + tmp43
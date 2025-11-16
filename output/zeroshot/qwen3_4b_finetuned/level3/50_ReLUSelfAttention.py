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
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 768
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 768 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 768 * x1), xmask, eviction_policy='evict_last'
        )
    tmp6 = tl.load(in_ptr0 + (2 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (4 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (5 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (6 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (7 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (8 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (9 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (10 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (11 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (12 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (13 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (14 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (15 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (16 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (17 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (18 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (19 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (20 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (21 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (22 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (23 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (24 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (25 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (26 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (27 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (28 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (29 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (30 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (31 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (32 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (33 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (34 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (35 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (36 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (37 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (38 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (39 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (40 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (41 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (42 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (43 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (44 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (45 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (46 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (47 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (48 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (49 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (50 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (51 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (52 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (53 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (54 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (55 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (56 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (57 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (58 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (59 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (60 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (61 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (62 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (63 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (64 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (65 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (66 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (67 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (68 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (69 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (70 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (71 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (72 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (73 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (74 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (75 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (76 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (77 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (78 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (79 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (80 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (81 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (82 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (83 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (84 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (85 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (86 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (87 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (88 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (89 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (90 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (91 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (92 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (93 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (94 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (95 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (96 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (97 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (98 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (99 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (100 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (101 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (102 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (103 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (104 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (105 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (106 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (107 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (108 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (109 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (110 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (111 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (112 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (113 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (114 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (115 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (116 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (117 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (118 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (119 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (120 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (121 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (122 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (123 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (124 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (125 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (126 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (127 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (128 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (129 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (130 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (131 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (132 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (133 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (134 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (135 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (136 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (137 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (138 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (139 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (140 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (141 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (142 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (143 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (144 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp435 = tl.load(in_ptr0 + (145 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (146 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp441 = tl.load(in_ptr0 + (147 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp444 = tl.load(in_ptr0 + (148 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp447 = tl.load(in_ptr0 + (149 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp450 = tl.load(in_ptr0 + (150 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp453 = tl.load(in_ptr0 + (151 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp456 = tl.load(in_ptr0 + (152 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp459 = tl.load(in_ptr0 + (153 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp462 = tl.load(in_ptr0 + (154 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp465 = tl.load(in_ptr0 + (155 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp468 = tl.load(in_ptr0 + (156 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp471 = tl.load(in_ptr0 + (157 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp474 = tl.load(in_ptr0 + (158 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp477 = tl.load(in_ptr0 + (159 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp480 = tl.load(in_ptr0 + (160 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp483 = tl.load(in_ptr0 + (161 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp486 = tl.load(in_ptr0 + (162 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp489 = tl.load(in_ptr0 + (163 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp492 = tl.load(in_ptr0 + (164 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp495 = tl.load(in_ptr0 + (165 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp498 = tl.load(in_ptr0 + (166 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp501 = tl.load(in_ptr0 + (167 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp504 = tl.load(in_ptr0 + (168 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp507 = tl.load(in_ptr0 + (169 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp510 = tl.load(in_ptr0 + (170 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp513 = tl.load(in_ptr0 + (171 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp516 = tl.load(in_ptr0 + (172 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp519 = tl.load(in_ptr0 + (173 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp522 = tl.load(in_ptr0 + (174 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp525 = tl.load(in_ptr0 + (175 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp528 = tl.load(in_ptr0 + (176 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp531 = tl.load(in_ptr0 + (177 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp534 = tl.load(in_ptr0 + (178 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp537 = tl.load(in_ptr0 + (179 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp540 = tl.load(in_ptr0 + (180 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp543 = tl.load(in_ptr0 + (181 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp546 = tl.load(in_ptr0 + (182 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp549 = tl.load(in_ptr0 + (183 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp552 = tl.load(in_ptr0 + (184 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp555 = tl.load(in_ptr0 + (185 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp558 = tl.load(in_ptr0 + (186 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp561 = tl.load(in_ptr0 + (187 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp564 = tl.load(in_ptr0 + (188 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp567 = tl.load(in_ptr0 + (189 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp570 = tl.load(in_ptr0 + (190 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp573 = tl.load(in_ptr0 + (191 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp576 = tl.load(in_ptr0 + (192 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp579 = tl.load(in_ptr0 + (193 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp582 = tl.load(in_ptr0 + (194 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp585 = tl.load(in_ptr0 + (195 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp588 = tl.load(in_ptr0 + (196 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp591 = tl.load(in_ptr0 + (197 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp594 = tl.load(in_ptr0 + (198 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp597 = tl.load(in_ptr0 + (199 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp600 = tl.load(in_ptr0 + (200 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp603 = tl.load(in_ptr0 + (201 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp606 = tl.load(in_ptr0 + (202 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp609 = tl.load(in_ptr0 + (203 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp612 = tl.load(in_ptr0 + (204 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp615 = tl.load(in_ptr0 + (205 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp618 = tl.load(in_ptr0 + (206 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp621 = tl.load(in_ptr0 + (207 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp624 = tl.load(in_ptr0 + (208 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp627 = tl.load(in_ptr0 + (209 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp630 = tl.load(in_ptr0 + (210 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp633 = tl.load(in_ptr0 + (211 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp636 = tl.load(in_ptr0 + (212 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp639 = tl.load(in_ptr0 + (213 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp642 = tl.load(in_ptr0 + (214 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp645 = tl.load(in_ptr0 + (215 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp648 = tl.load(in_ptr0 + (216 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp651 = tl.load(in_ptr0 + (217 + 768 * x1), xmask, eviction_policy=
        'evict_last')
    tmp
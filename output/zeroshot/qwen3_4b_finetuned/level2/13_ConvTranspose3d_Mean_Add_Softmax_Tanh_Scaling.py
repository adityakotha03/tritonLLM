import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__unsafe_index_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 63, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x1 + 1024 * x0), tmp4 & xmask, other=0.0)
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.load(in_ptr1 + (x1 + 1024 * x0), tmp9 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp11 = tmp5 * tmp10
    tmp12 = tmp7 + tmp11
    tl.store(out_ptr0 + x2, tmp12, xmask)


@triton.jit
def triton_per_fused_add_mean_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = xindex // 1024 % 64
    x4 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4096 + x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5120 + x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (6144 + x3), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (7168 + x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (8192 + x3), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (9216 + x3), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (10240 + x3), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (11264 + x3), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (12288 + x3), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (13312 + x3), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (14336 + x3), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (15360 + x3), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (16384 + x3), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (17408 + x3), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (18432 + x3), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (19456 + x3), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (20480 + x3), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (21504 + x3), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (22528 + x3), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (23552 + x3), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (24576 + x3), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (25600 + x3), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (26624 + x3), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (27648 + x3), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (28672 + x3), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (29696 + x3), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (30720 + x3), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (31744 + x3), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (32768 + x3), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (33792 + x3), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (34816 + x3), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (35840 + x3), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (36864 + x3), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (37888 + x3), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (38912 + x3), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (39936 + x3), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (40960 + x3), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (41984 + x3), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (43008 + x3), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (44032 + x3), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (45056 + x3), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (46080 + x3), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (47104 + x3), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (48128 + x3), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (49152 + x3), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (50176 + x3), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (51200 + x3), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (52224 + x3), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (53248 + x3), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (54272 + x3), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (55296 + x3), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (56320 + x3), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (57344 + x3), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (58368 + x3), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (59392 + x3), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (60416 + x3), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (61440 + x3), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (62464 + x3), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (63488 + x3), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (64512 + x3), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (65536 + x3), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (66560 + x3), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (67584 + x3), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (68608 + x3), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (69632 + x3), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (70656 + x3), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (71680 + x3), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (72704 + x3), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (73728 + x3), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (74752 + x3), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (75776 + x3), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (76800 + x3), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (77824 + x3), xmask, eviction_policy='evict_last'
        )
    tmp228 = tl.load(in_ptr0 + (78848 + x3), xmask, eviction_policy='evict_last'
        )
    tmp231 = tl.load(in_ptr0 + (79872 + x3), xmask, eviction_policy='evict_last'
        )
    tmp234 = tl.load(in_ptr0 + (80896 + x3), xmask, eviction_policy='evict_last'
        )
    tmp237 = tl.load(in_ptr0 + (81920 + x3), xmask, eviction_policy='evict_last'
        )
    tmp240 = tl.load(in_ptr0 + (82944 + x3), xmask, eviction_policy='evict_last'
        )
    tmp243 = tl.load(in_ptr0 + (83968 + x3), xmask, eviction_policy='evict_last'
        )
    tmp246 = tl.load(in_ptr0 + (84992 + x3), xmask, eviction_policy='evict_last'
        )
    tmp249 = tl.load(in_ptr0 + (86016 + x3), xmask, eviction_policy='evict_last'
        )
    tmp252 = tl.load(in_ptr0 + (87040 + x3), xmask, eviction_policy='evict_last'
        )
    tmp255 = tl.load(in_ptr0 + (88064 + x3), xmask, eviction_policy='evict_last'
        )
    tmp258 = tl.load(in_ptr0 + (89088 + x3), xmask, eviction_policy='evict_last'
        )
    tmp261 = tl.load(in_ptr0 + (90112 + x3), xmask, eviction_policy='evict_last'
        )
    tmp264 = tl.load(in_ptr0 + (91136 + x3), xmask, eviction_policy='evict_last'
        )
    tmp267 = tl.load(in_ptr0 + (92160 + x3), xmask, eviction_policy='evict_last'
        )
    tmp270 = tl.load(in_ptr0 + (93184 + x3), xmask, eviction_policy='evict_last'
        )
    tmp273 = tl.load(in_ptr0 + (94208 + x3), xmask, eviction_policy='evict_last'
        )
    tmp276 = tl.load(in_ptr0 + (95232 + x3), xmask, eviction_policy='evict_last'
        )
    tmp279 = tl.load(in_ptr0 + (96256 + x3), xmask, eviction_policy='evict_last'
        )
    tmp282 = tl.load(in_ptr0 + (97280 + x3), xmask, eviction_policy='evict_last'
        )
    tmp285 = tl.load(in_ptr0 + (98304 + x3), xmask, eviction_policy='evict_last'
        )
    tmp288 = tl.load(in_ptr0 + (99328 + x3), xmask, eviction_policy='evict_last'
        )
    tmp291 = tl.load(in_ptr0 + (100352 + x3), xmask, eviction_policy='evict_last'
        )
    tmp294 = tl.load(in_ptr0 + (101376 + x3), xmask, eviction_policy='evict_last'
        )
    tmp297 = tl.load(in_ptr0 + (102400 + x3), xmask, eviction_policy='evict_last'
        )
    tmp300 = tl.load(in_ptr0 + (103424 + x3), xmask, eviction_policy='evict_last'
        )
    tmp303 = tl.load(in_ptr0 + (104448 + x3), xmask, eviction_policy='evict_last'
        )
    tmp306 = tl.load(in_ptr0 + (105472 + x3), xmask, eviction_policy='evict_last'
        )
    tmp309 = tl.load(in_ptr0 + (106496 + x3), xmask, eviction_policy='evict_last'
        )
    tmp312 = tl.load(in_ptr0 + (107520 + x3), xmask, eviction_policy='evict_last'
        )
    tmp315 = tl.load(in_ptr0 + (108544 + x3), xmask, eviction_policy='evict_last'
        )
    tmp318 = tl.load(in_ptr0 + (109568 + x3), xmask, eviction_policy='evict_last'
        )
    tmp321 = tl.load(in_ptr0 + (110592 + x3), xmask, eviction_policy='evict_last'
        )
    tmp324 = tl.load(in_ptr0 + (111616 + x3), xmask, eviction_policy='evict_last'
        )
    tmp327 = tl.load(in_ptr0 + (112640 + x3), xmask, eviction_policy='evict_last'
        )
    tmp330 = tl.load(in_ptr0 + (113664 + x3), xmask, eviction_policy='evict_last'
        )
    tmp333 = tl.load(in_ptr0 + (114688 + x3), xmask, eviction_policy='evict_last'
        )
    tmp336 = tl.load(in_ptr0 + (115712 + x3), xmask, eviction_policy='evict_last'
        )
    tmp339 = tl.load(in_ptr0 + (116736 + x3), xmask, eviction_policy='evict_last'
        )
    tmp342 = tl.load(in_ptr0 + (117760 + x3), xmask, eviction_policy='evict_last'
        )
    tmp345 = tl.load(in_ptr0 + (118784 + x3), xmask, eviction_policy='evict_last'
        )
    tmp348 = tl.load(in_ptr0 + (119808 + x3), xmask, eviction_policy='evict_last'
        )
    tmp351 = tl.load(in_ptr0 + (120832 + x3), xmask, eviction_policy='evict_last'
        )
    tmp354 = tl.load(in_ptr0 + (121856 + x3), xmask, eviction_policy='evict_last'
        )
    tmp357 = tl.load(in_ptr0 + (122880 + x3), xmask, eviction_policy='evict_last'
        )
    tmp360 = tl.load(in_ptr0 + (123904 + x3), xmask, eviction_policy='evict_last'
        )
    tmp363 = tl.load(in_ptr0 + (124928 + x3), xmask, eviction_policy='evict_last'
        )
    tmp366 = tl.load(in_ptr0 + (125952 + x3), xmask, eviction_policy='evict_last'
        )
    tmp369 = tl.load(in_ptr0 + (126976 + x3), xmask, eviction_policy='evict_last'
        )
    tmp372 = tl.load(in_ptr0 + (127000 + x3), xmask, eviction_policy='evict_last'
        )
    tmp375 = tl.load(in_ptr0 + (128024 + x3), xmask, eviction_policy='evict_last'
        )
    tmp378 = tl.load(in_ptr0 + (129048 + x3), xmask, eviction_policy='evict_last'
        )
    tmp381 = tl.load(in_ptr0 + (130072 + x3), xmask, eviction_policy='evict_last'
        )
    tmp384 = tl.load(in_ptr0 + (131096 + x3), xmask, eviction_policy='evict_last'
        )
    tmp387 = tl.load(in_ptr0 + (132120 + x3), xmask, eviction_policy='evict_last'
        )
    tmp390 = tl.load(in_ptr0 + (133144 + x3), xmask, eviction_policy='evict_last'
        )
    tmp393 = tl.load(in_ptr0 + (134168 + x3), xmask, eviction_policy='evict_last'
        )
    tmp396 = tl.load(in_ptr0 + (135192 + x3), xmask, eviction_policy='evict_last'
        )
    tmp399 = tl.load(in_ptr0 + (136216 + x3), xmask, eviction_policy='evict_last'
        )
    tmp402 = tl.load(in_ptr0 + (137240 + x3), xmask, eviction_policy='evict_last'
        )
    tmp405 = tl.load(in_ptr0 + (138264 + x3), xmask, eviction_policy='evict_last'
        )
    tmp408 = tl.load(in_ptr0 + (139288 + x3), xmask, eviction_policy='evict_last'
        )
    tmp411 = tl.load(in_ptr0 + (140312 + x3), xmask, eviction_policy='evict_last'
        )
    tmp414 = tl.load(in_ptr0 + (141336 + x3), xmask, eviction_policy='evict_last'
        )
    tmp417 = tl.load(in_ptr0 + (142360 + x3), xmask, eviction_policy='evict_last'
        )
    tmp420 = tl.load(in_ptr0 + (143384 + x3), xmask, eviction_policy='evict_last'
        )
    tmp423 = tl.load(in_ptr0 + (144408 + x3), xmask, eviction_policy='evict_last'
        )
    tmp426 = tl.load(in_ptr0 + (145432 + x3), xmask, eviction_policy='evict_last'
        )
    tmp429 = tl.load(in_ptr0 + (146456 + x3), xmask, eviction_policy='evict_last'
        )
    tmp432 = tl.load(in_ptr0 + (147480 + x3), xmask, eviction_policy='evict_last'
        )
    tmp435 = tl.load(in_ptr0 + (148504 + x3), xmask, eviction_policy='evict_last'
        )
    tmp438 = tl.load(in_ptr0 + (149528 + x3), xmask, eviction_policy='evict_last'
        )
    tmp441 = tl.load(in_ptr0 + (150552 + x3), xmask, eviction_policy='evict_last'
        )
    tmp444 = tl.load(in_ptr0 + (151576 + x3), xmask, eviction_policy='evict_last'
        )
    tmp447 = tl.load(in_ptr0 + (152600 + x3), xmask, eviction_policy='evict_last'
        )
    tmp450 = tl.load(in_ptr0 + (153624 + x3), xmask, eviction_policy='evict_last'
        )
    tmp453 = tl.load(in_ptr0 + (154648 + x3), xmask, eviction_policy='evict_last'
        )
    tmp456 = tl.load(in_ptr0 + (155672 + x3), xmask, eviction_policy='evict_last'
        )
    tmp459 = tl.load(in_ptr0 + (156696 + x3), xmask, eviction_policy='evict_last'
        )
    tmp462 = tl.load(in_ptr0 + (157720 + x3), xmask, eviction_policy='evict_last'
        )
    tmp465 = tl.load(in_ptr0 + (158744 + x3), xmask, eviction_policy='evict_last'
        )
    tmp468 = tl.load(in_ptr0 + (159768 + x3), xmask, eviction_policy='evict_last'
        )
    tmp471 = tl.load(in_ptr0 + (160792 + x3), xmask, eviction_policy='evict_last'
        )
    tmp474 = tl.load(in_ptr0 + (161816 + x3), xmask, eviction_policy='evict_last'
        )
    tmp477 = tl.load(in_ptr0 + (162840 + x3), xmask, eviction_policy='evict_last'
        )
    tmp480 = tl.load(in_ptr0 + (163864 + x3), xmask, eviction_policy='evict_last'
        )
    tmp483 = tl.load(in_ptr0 + (164888 + x3), xmask, eviction_policy='evict_last'
        )
    tmp486 = tl.load(in_ptr0 + (165912 + x3), xmask, eviction_policy='evict_last'
        )
    tmp489 = tl.load(in_ptr0 + (166936 + x3), xmask, eviction_policy='evict_last'
        )
    tmp492 = tl.load(in_ptr0 + (167960 + x3), xmask, eviction_policy='evict_last'
        )
    tmp495 = tl.load(in_ptr0 + (168984 + x3), xmask, eviction_policy='evict_last'
        )
    tmp498 = tl.load(in_ptr0 + (169008 + x3), xmask, eviction_policy='evict_last'
        )
    tmp501 = tl.load(in_ptr0 + (170032 + x3), xmask, eviction_policy='evict_last'
        )
    tmp504 = tl.load(in_ptr0 + (171056 + x3), xmask, eviction_policy='evict_last'
        )
    tmp507 = tl.load(in_ptr0 + (172080 + x3), xmask, eviction_policy='evict_last'
        )
    tmp510 = tl.load(in_ptr0 + (173104 + x3), xmask, eviction_policy='evict_last'
        )
    tmp513 = tl.load(in_ptr0 + (174128 + x3), xmask, eviction_policy='evict_last'
        )
    tmp516 = tl.load(in_ptr0 + (175152 + x3), xmask, eviction_policy='evict_last'
        )
    tmp519 = tl.load(in_ptr0 + (176176 + x3), xmask, eviction_policy='evict_last'
        )
    tmp522 = tl.load(in_ptr0 + (177200 + x3), xmask, eviction_policy='evict_last'
        )
    tmp525 = tl.load(in_ptr0 + (178224 + x3), xmask, eviction_policy='evict_last'
        )
    tmp528 = tl.load(in_ptr0 + (179248 + x3), xmask, eviction_policy='evict_last'
        )
    tmp531 = tl.load(in_ptr0 + (180272 + x3), xmask, eviction_policy='evict_last'
        )
    tmp534 = tl.load(in_ptr0 + (181296 + x3), xmask, eviction_policy='evict_last'
        )
    tmp537 = tl.load(in_ptr0 + (182320 + x3), xmask, eviction_policy='evict_last'
        )
    tmp540 = tl.load(in_ptr0 + (183344 + x3), xmask, eviction_policy='evict_last'
        )
    tmp543 = tl.load(in_ptr0 + (184368 + x3), xmask, eviction_policy='evict_last'
        )
    tmp546 = tl.load(in_ptr0 + (185392 + x3), xmask, eviction_policy='evict_last'
        )
    tmp549 = tl.load(in_ptr0 + (186416 + x3), xmask, eviction_policy='evict_last'
        )
    tmp552 = tl.load(in_ptr0 + (187440 + x3), xmask, eviction_policy='evict_last'
        )
    tmp555 = tl.load(in_ptr0 + (188464 + x3), xmask, eviction_policy='evict_last'
        )
    tmp558 = tl.load(in_ptr0 + (189488 + x3), xmask, eviction_policy='evict_last'
        )
    tmp561 = tl.load(in_ptr0 + (190512 + x3), xmask, eviction_policy='evict_last'
        )
    tmp564 = tl.load(in_ptr0 + (191536 + x3), xmask, eviction_policy='evict_last'
        )
    tmp567 = tl.load(in_ptr0 + (192560 + x3), xmask, eviction_policy='evict_last'
        )
    tmp570 = tl.load(in_ptr0 + (193584 + x3), xmask, eviction_policy='evict_last'
        )
    tmp573 = tl.load(in_ptr0 + (194608 + x3), xmask, eviction_policy='evict_last'
        )
    tmp576 = tl.load(in_ptr0 + (195632 + x3), xmask, eviction_policy='evict_last'
        )
    tmp579 = tl.load(in_ptr0 + (196656 + x3), xmask, eviction_policy='evict_last'
        )
    tmp582 = tl.load(in_ptr0 + (197680 + x3), xmask, eviction_policy='evict_last'
        )
    tmp585 = tl.load(in_ptr0 + (198704 + x3), xmask, eviction_policy='evict_last'
        )
    tmp588 = tl.load(in_ptr0 + (199728 + x3), xmask, eviction_policy='evict_last'
        )
    tmp591 = tl.load(in_ptr0 + (200752 + x3), xmask, eviction_policy='evict_last'
        )
    tmp594 = tl.load(in_ptr0 + (201776 + x3), xmask, eviction_policy='evict_last'
        )
    tmp597 = tl.load(in_ptr0 + (202800 + x3), xmask, eviction_policy='evict_last'
        )
    tmp600 = tl.load(in_ptr0 + (203824 + x3), xmask, eviction_policy='evict_last'
        )
    tmp603 = tl.load(in_ptr0 + (204848 + x3), xmask, eviction_policy='evict_last'
        )
    tmp606 = tl.load(in_ptr0 + (205872 + x3), xmask, eviction_policy='evict_last'
        )
    tmp609 = tl.load(in_ptr0 + (206896 + x3), xmask, eviction_policy='evict_last'
        )
    tmp612 = tl.load(in_ptr0 + (207920 + x3), xmask, eviction_policy='evict_last'
        )
    tmp615 = tl.load(in_ptr0 + (208944 + x3), xmask, eviction_policy='evict_last'
        )
    tmp618 = tl.load(in_ptr0 + (209968 + x3), xmask, eviction_policy='evict_last'
        )
    tmp621 = tl.load(in_ptr0 + (210992 + x3), xmask, eviction_policy='evict_last'
        )
    tmp624 = tl.load(in_ptr0 + (212016 + x3), xmask, eviction_policy='evict_last'
        )
    tmp627 = tl.load(in_ptr0 + (213040 + x3), xmask, eviction_policy='evict_last'
        )
    tmp630 = tl.load(in_ptr0 + (214064 + x3), xmask, eviction_policy='evict_last'
        )
    tmp633 = tl.load(in_ptr0 + (215088 + x3), xmask, eviction_policy='evict_last'
        )
    tmp636 = tl.load(in_ptr0 + (216112 + x3), xmask, eviction_policy='evict_last'
        )
    tmp639 = tl.load(in_ptr0 + (217136 + x3), xmask, eviction_policy='evict_last'
        )
    tmp642 = tl.load(in_ptr0 + (218160 + x3), xmask, eviction_policy='evict_last'
        )
    tmp645 = tl.load(in_ptr0 + (219184 + x3), xmask, eviction_policy='evict_last'
        )
    tmp648 = tl.load(in_ptr0 + (220208 + x3), xmask, eviction_policy='evict_last'
        )
    tmp651 = tl.load(in_ptr0 + (221232 + x3), xmask, eviction_policy='evict_last'
        )
    tmp654 = tl.load(in_ptr0 + (222256 + x3), xmask, eviction_policy='evict_last'
        )
    tmp657 = tl.load(in_ptr0 + (223280 + x3), xmask, eviction_policy='evict_last'
        )
    tmp660 = tl.load(in_ptr0
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
import torch as th
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 32
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 2048 * y3), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512 * x2 + 1048576 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_per_fused__softmax_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5120 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7168 + x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8192 + x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9216 + x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10240 + x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11264 + x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (12288 + x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (13312 + x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (14336 + x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (15360 + x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (16384 + x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (17408 + x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (18432 + x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (19456 + x0), xmask, eviction_policy='evict_last'
        )
    tmp40 = tl.load(in_ptr0 + (20480 + x0), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (21504 + x0), xmask, eviction_policy='evict_last'
        )
    tmp44 = tl.load(in_ptr0 + (22528 + x0), xmask, eviction_policy='evict_last'
        )
    tmp46 = tl.load(in_ptr0 + (23552 + x0), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (24576 + x0), xmask, eviction_policy='evict_last'
        )
    tmp50 = tl.load(in_ptr0 + (25600 + x0), xmask, eviction_policy='evict_last'
        )
    tmp52 = tl.load(in_ptr0 + (26624 + x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (27648 + x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr0 + (28672 + x0), xmask, eviction_policy='evict_last'
        )
    tmp58 = tl.load(in_ptr0 + (29696 + x0), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (30720 + x0), xmask, eviction_policy='evict_last'
        )
    tmp62 = tl.load(in_ptr0 + (31744 + x0), xmask, eviction_policy='evict_last'
        )
    tmp64 = tl.load(in_ptr0 + (32768 + x0), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (33792 + x0), xmask, eviction_policy='evict_last'
        )
    tmp68 = tl.load(in_ptr0 + (34816 + x0), xmask, eviction_policy='evict_last'
        )
    tmp70 = tl.load(in_ptr0 + (35840 + x0), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (36864 + x0), xmask, eviction_policy='evict_last'
        )
    tmp74 = tl.load(in_ptr0 + (37888 + x0), xmask, eviction_policy='evict_last'
        )
    tmp76 = tl.load(in_ptr0 + (38912 + x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (39936 + x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr0 + (40960 + x0), xmask, eviction_policy='evict_last'
        )
    tmp82 = tl.load(in_ptr0 + (41984 + x0), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (43008 + x0), xmask, eviction_policy='evict_last'
        )
    tmp86 = tl.load(in_ptr0 + (44032 + x0), xmask, eviction_policy='evict_last'
        )
    tmp88 = tl.load(in_ptr0 + (45056 + x0), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (46080 + x0), xmask, eviction_policy='evict_last'
        )
    tmp92 = tl.load(in_ptr0 + (47104 + x0), xmask, eviction_policy='evict_last'
        )
    tmp94 = tl.load(in_ptr0 + (48128 + x0), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (49152 + x0), xmask, eviction_policy='evict_last'
        )
    tmp98 = tl.load(in_ptr0 + (50176 + x0), xmask, eviction_policy='evict_last'
        )
    tmp100 = tl.load(in_ptr0 + (51200 + x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (52224 + x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr0 + (53248 + x0), xmask, eviction_policy='evict_last'
        )
    tmp106 = tl.load(in_ptr0 + (54272 + x0), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (55296 + x0), xmask, eviction_policy='evict_last'
        )
    tmp110 = tl.load(in_ptr0 + (56320 + x0), xmask, eviction_policy='evict_last'
        )
    tmp112 = tl.load(in_ptr0 + (57344 + x0), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (58368 + x0), xmask, eviction_policy='evict_last'
        )
    tmp116 = tl.load(in_ptr0 + (59392 + x0), xmask, eviction_policy='evict_last'
        )
    tmp118 = tl.load(in_ptr0 + (60416 + x0), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (61440 + x0), xmask, eviction_policy='evict_last'
        )
    tmp122 = tl.load(in_ptr0 + (62464 + x0), xmask, eviction_policy='evict_last'
        )
    tmp124 = tl.load(in_ptr0 + (63488 + x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (64512 + x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr0 + (65536 + x0), xmask, eviction_policy='evict_last'
        )
    tmp130 = tl.load(in_ptr0 + (66560 + x0), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (67584 + x0), xmask, eviction_policy='evict_last'
        )
    tmp134 = tl.load(in_ptr0 + (68608 + x0), xmask, eviction_policy='evict_last'
        )
    tmp136 = tl.load(in_ptr0 + (69632 + x0), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (70656 + x0), xmask, eviction_policy='evict_last'
        )
    tmp140 = tl.load(in_ptr0 + (71680 + x0), xmask, eviction_policy='evict_last'
        )
    tmp142 = tl.load(in_ptr0 + (72704 + x0), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (73728 + x0), xmask, eviction_policy='evict_last'
        )
    tmp146 = tl.load(in_ptr0 + (74752 + x0), xmask, eviction_policy='evict_last'
        )
    tmp148 = tl.load(in_ptr0 + (75776 + x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (76800 + x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr0 + (77824 + x0), xmask, eviction_policy='evict_last'
        )
    tmp154 = tl.load(in_ptr0 + (78848 + x0), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (79872 + x0), xmask, eviction_policy='evict_last'
        )
    tmp158 = tl.load(in_ptr0 + (80896 + x0), xmask, eviction_policy='evict_last'
        )
    tmp160 = tl.load(in_ptr0 + (81920 + x0), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (82944 + x0), xmask, eviction_policy='evict_last'
        )
    tmp164 = tl.load(in_ptr0 + (83968 + x0), xmask, eviction_policy='evict_last'
        )
    tmp166 = tl.load(in_ptr0 + (84992 + x0), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (86016 + x0), xmask, eviction_policy='evict_last'
        )
    tmp170 = tl.load(in_ptr0 + (87040 + x0), xmask, eviction_policy='evict_last'
        )
    tmp172 = tl.load(in_ptr0 + (88064 + x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (89088 + x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr0 + (90112 + x0), xmask, eviction_policy='evict_last'
        )
    tmp178 = tl.load(in_ptr0 + (91136 + x0), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (92160 + x0), xmask, eviction_policy='evict_last'
        )
    tmp182 = tl.load(in_ptr0 + (93184 + x0), xmask, eviction_policy='evict_last'
        )
    tmp184 = tl.load(in_ptr0 + (94208 + x0), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (95232 + x0), xmask, eviction_policy='evict_last'
        )
    tmp188 = tl.load(in_ptr0 + (96256 + x0), xmask, eviction_policy='evict_last'
        )
    tmp190 = tl.load(in_ptr0 + (97280 + x0), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (98304 + x0), xmask, eviction_policy='evict_last'
        )
    tmp194 = tl.load(in_ptr0 + (99328 + x0), xmask, eviction_policy='evict_last'
        )
    tmp196 = tl.load(in_ptr0 + (100352 + x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (101376 + x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr0 + (102400 + x0), xmask, eviction_policy='evict_last'
        )
    tmp202 = tl.load(in_ptr0 + (103424 + x0), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (104448 + x0), xmask, eviction_policy='evict_last'
        )
    tmp206 = tl.load(in_ptr0 + (105472 + x0), xmask, eviction_policy='evict_last'
        )
    tmp208 = tl.load(in_ptr0 + (106496 + x0), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (107520 + x0), xmask, eviction_policy='evict_last'
        )
    tmp212 = tl.load(in_ptr0 + (108544 + x0), xmask, eviction_policy='evict_last'
        )
    tmp214 = tl.load(in_ptr0 + (109568 + x0), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (110592 + x0), xmask, eviction_policy='evict_last'
        )
    tmp218 = tl.load(in_ptr0 + (111616 + x0), xmask, eviction_policy='evict_last'
        )
    tmp220 = tl.load(in_ptr0 + (112640 + x0), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (113664 + x0), xmask, eviction_policy='evict_last'
        )
    tmp224 = tl.load(in_ptr0 + (114688 + x0), xmask, eviction_policy='evict_last'
        )
    tmp226 = tl.load(in_ptr0 + (115712 + x0), xmask, eviction_policy='evict_last'
        )
    tmp228 = tl.load(in_ptr0 + (116736 + x0), xmask, eviction_policy='evict_last'
        )
    tmp230 = tl.load(in_ptr0 + (117760 + x0), xmask, eviction_policy='evict_last'
        )
    tmp232 = tl.load(in_ptr0 + (118784 + x0), xmask, eviction_policy='evict_last'
        )
    tmp234 = tl.load(in_ptr0 + (119808 + x0), xmask, eviction_policy='evict_last'
        )
    tmp236 = tl.load(in_ptr0 + (120832 + x0), xmask, eviction_policy='evict_last'
        )
    tmp238 = tl.load(in_ptr0 + (121856 + x0), xmask, eviction_policy='evict_last'
        )
    tmp240 = tl.load(in_ptr0 + (122880 + x0), xmask, eviction_policy='evict_last'
        )
    tmp242 = tl.load(in_ptr0 + (123904 + x0), xmask, eviction_policy='evict_last'
        )
    tmp244 = tl.load(in_ptr0 + (124928 + x0), xmask, eviction_policy='evict_last'
        )
    tmp246 = tl.load(in_ptr0 + (125952 + x0), xmask, eviction_policy='evict_last'
        )
    tmp248 = tl.load(in_ptr0 + (126976 + x0), xmask, eviction_policy='evict_last'
        )
    tmp250 = tl.load(in_ptr0 + (127999 + x0), xmask, eviction_policy='evict_last'
        )
    tmp251 = tl.load(in_ptr0 + (128000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp252 = tl.load(in_ptr0 + (256000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp253 = tl.load(in_ptr0 + (256001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp254 = tl.load(in_ptr0 + (512000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp255 = tl.load(in_ptr0 + (512001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp256 = tl.load(in_ptr0 + (768000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp257 = tl.load(in_ptr0 + (768001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp258 = tl.load(in_ptr0 + (1024000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp259 = tl.load(in_ptr0 + (1024001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp260 = tl.load(in_ptr0 + (1280000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp261 = tl.load(in_ptr0 + (1280001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp262 = tl.load(in_ptr0 + (1536000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp263 = tl.load(in_ptr0 + (1536001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp264 = tl.load(in_ptr0 + (1792000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp265 = tl.load(in_ptr0 + (1792001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp266 = tl.load(in_ptr0 + (2048000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp267 = tl.load(in_ptr0 + (2048001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp268 = tl.load(in_ptr0 + (2304000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp269 = tl.load(in_ptr0 + (2304001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp270 = tl.load(in_ptr0 + (2560000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp271 = tl.load(in_ptr0 + (2560001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp272 = tl.load(in_ptr0 + (2816000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp273 = tl.load(in_ptr0 + (2816001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp274 = tl.load(in_ptr0 + (3072000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp275 = tl.load(in_ptr0 + (3072001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp276 = tl.load(in_ptr0 + (3328000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp277 = tl.load(in_ptr0 + (3328001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp278 = tl.load(in_ptr0 + (3584000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp279 = tl.load(in_ptr0 + (3584001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp280 = tl.load(in_ptr0 + (3840000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp281 = tl.load(in_ptr0 + (3840001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp282 = tl.load(in_ptr0 + (4096000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp283 = tl.load(in_ptr0 + (4096001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp284 = tl.load(in_ptr0 + (4352000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp285 = tl.load(in_ptr0 + (4352001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp286 = tl.load(in_ptr0 + (4608000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp287 = tl.load(in_ptr0 + (4608001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp288 = tl.load(in_ptr0 + (4864000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp289 = tl.load(in_ptr0 + (4864001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp290 = tl.load(in_ptr0 + (5120000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp291 = tl.load(in_ptr0 + (5120001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp292 = tl.load(in_ptr0 + (5376000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp293 = tl.load(in_ptr0 + (5376001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp294 = tl.load(in_ptr0 + (5632000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp295 = tl.load(in_ptr0 + (5632001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp296 = tl.load(in_ptr0 + (5888000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp297 = tl.load(in_ptr0 + (5888001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp298 = tl.load(in_ptr0 + (6144000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp299 = tl.load(in_ptr0 + (6144001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp300 = tl.load(in_ptr0 + (6400000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp301 = tl.load(in_ptr0 + (6400001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp302 = tl.load(in_ptr0 + (6656000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp303 = tl.load(in_ptr0 + (6656001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp304 = tl.load(in_ptr0 + (6912000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp305 = tl.load(in_ptr0 + (6912001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp306 = tl.load(in_ptr0 + (7168000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp307 = tl.load(in_ptr0 + (7168001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp308 = tl.load(in_ptr0 + (7424000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp309 = tl.load(in_ptr0 + (7424001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp310 = tl.load(in_ptr0 + (7680000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp311 = tl.load(in_ptr0 + (7680001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp312 = tl.load(in_ptr0 + (7936000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp313 = tl.load(in_ptr0 + (7936001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp314 = tl.load(in_ptr0 + (8192000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp315 = tl.load(in_ptr0 + (8192001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp316 = tl.load(in_ptr0 + (8448000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp317 = tl.load(in_ptr0 + (8448001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp318 = tl.load(in_ptr0 + (8694000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp319 = tl.load(in_ptr0 + (8694001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp320 = tl.load(in_ptr0 + (8940000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp321 = tl.load(in_ptr0 + (8940001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp322 = tl.load(in_ptr0 + (9186000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp323 = tl.load(in_ptr0 + (9186001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp324 = tl.load(in_ptr0 + (9432000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp325 = tl.load(in_ptr0 + (9432001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp326 = tl.load(in_ptr0 + (9678000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp327 = tl.load(in_ptr0 + (9678001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp328 = tl.load(in_ptr0 + (9924000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp329 = tl.load(in_ptr0 + (9924001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp330 = tl.load(in_ptr0 + (10170000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp331 = tl.load(in_ptr0 + (10170001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp332 = tl.load(in_ptr0 + (10416000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp333 = tl.load(in_ptr0 + (10416001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp334 = tl.load(in_ptr0 + (10662000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp335 = tl.load(in_ptr0 + (10662001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp336 = tl.load(in_ptr0 + (10908000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp337 = tl.load(in_ptr0 + (10908001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp338 = tl.load(in_ptr0 + (11154000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp339 = tl.load(in_ptr0 + (11154001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp340 = tl.load(in_ptr0 + (11400000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp341 = tl.load(in_ptr0 + (11400001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp342 = tl.load(in_ptr0 + (11646000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp343 = tl.load(in_ptr0 + (11646001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp344 = tl.load(in_ptr0 + (11892000 + x0), xmask, eviction_policy='evict_last'
        )
    tmp345 = tl.load(in_ptr0 + (11892001 + x0), xmask, eviction_policy='evict_last'
        )
    tmp346 = tl.load(in_ptr0 + (12138000 + x0), xmask, eviction_policy='evict_last
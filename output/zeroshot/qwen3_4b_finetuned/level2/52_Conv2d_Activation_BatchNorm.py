import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel,
    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 16384
    xnumel = 36
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 36 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tl.store(out_ptr0 + (y0 + 128 * x2 + 4608 * y1), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_add_mul_tanh_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp3 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.sigmoid(tmp2)
    tmp5 = tmp4 * tmp3
    tl.store(in_out_ptr0 + x0, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_batch_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x2), xmask)
    tmp4 = tl.load(in_ptr1 + (1 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x2), xmask)
    tmp7 = tl.load(in_ptr1 + (2 + x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (384 + x2), xmask)
    tmp11 = tl.load(in_ptr1 + (3 + x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (512 + x2), xmask)
    tmp15 = tl.load(in_ptr1 + (4 + x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (640 + x2), xmask)
    tmp19 = tl.load(in_ptr1 + (5 + x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (768 + x2), xmask)
    tmp23 = tl.load(in_ptr1 + (6 + x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (896 + x2), xmask)
    tmp27 = tl.load(in_ptr1 + (7 + x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (1024 + x2), xmask)
    tmp31 = tl.load(in_ptr1 + (8 + x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (1152 + x2), xmask)
    tmp35 = tl.load(in_ptr1 + (9 + x1), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (1280 + x2), xmask)
    tmp39 = tl.load(in_ptr1 + (10 + x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1408 + x2), xmask)
    tmp43 = tl.load(in_ptr1 + (11 + x1), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (1536 + x2), xmask)
    tmp47 = tl.load(in_ptr1 + (12 + x1), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (1664 + x2), xmask)
    tmp51 = tl.load(in_ptr1 + (13 + x1), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (1792 + x2), xmask)
    tmp55 = tl.load(in_ptr1 + (14 + x1), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (1920 + x2), xmask)
    tmp59 = tl.load(in_ptr1 + (15 + x1), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr0 + (2048 + x2), xmask)
    tmp63 = tl.load(in_ptr1 + (16 + x1), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr0 + (2176 + x2), xmask)
    tmp67 = tl.load(in_ptr1 + (17 + x1), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (2304 + x2), xmask)
    tmp71 = tl.load(in_ptr1 + (18 + x1), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr0 + (2432 + x2), xmask)
    tmp75 = tl.load(in_ptr1 + (19 + x1), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (2560 + x2), xmask)
    tmp79 = tl.load(in_ptr1 + (20 + x1), xmask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr0 + (2688 + x2), xmask)
    tmp83 = tl.load(in_ptr1 + (21 + x1), xmask, eviction_policy='evict_last')
    tmp86 = tl.load(in_ptr0 + (2816 + x2), xmask)
    tmp87 = tl.load(in_ptr1 + (22 + x1), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (2944 + x2), xmask)
    tmp91 = tl.load(in_ptr1 + (23 + x1), xmask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (3072 + x2), xmask)
    tmp95 = tl.load(in_ptr1 + (24 + x1), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr0 + (3200 + x2), xmask)
    tmp99 = tl.load(in_ptr1 + (25 + x1), xmask, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr0 + (3328 + x2), xmask)
    tmp103 = tl.load(in_ptr1 + (26 + x1), xmask, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr0 + (3456 + x2), xmask)
    tmp107 = tl.load(in_ptr1 + (27 + x1), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr0 + (3584 + x2), xmask)
    tmp111 = tl.load(in_ptr1 + (28 + x1), xmask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr0 + (3712 + x2), xmask)
    tmp115 = tl.load(in_ptr1 + (29 + x1), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr0 + (3840 + x2), xmask)
    tmp119 = tl.load(in_ptr1 + (30 + x1), xmask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr0 + (3968 + x2), xmask)
    tmp123 = tl.load(in_ptr1 + (31 + x1), xmask, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr0 + (4096 + x2), xmask)
    tmp127 = tl.load(in_ptr1 + (32 + x1), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr0 + (4224 + x2), xmask)
    tmp131 = tl.load(in_ptr1 + (33 + x1), xmask, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr0 + (4352 + x2), xmask)
    tmp135 = tl.load(in_ptr1 + (34 + x1), xmask, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (4480 + x2), xmask)
    tmp139 = tl.load(in_ptr1 + (35 + x1), xmask, eviction_policy='evict_last')
    tmp142 = tl.load(in_ptr0 + (4608 + x2), xmask)
    tmp143 = tl.load(in_ptr1 + (36 + x1), xmask, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (4736 + x2), xmask)
    tmp147 = tl.load(in_ptr1 + (37 + x1), xmask, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr0 + (4864 + x2), xmask)
    tmp151 = tl.load(in_ptr1 + (38 + x1), xmask, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr0 + (4992 + x2), xmask)
    tmp155 = tl.load(in_ptr1 + (39 + x1), xmask, eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr0 + (5120 + x2), xmask)
    tmp159 = tl.load(in_ptr1 + (40 + x1), xmask, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr0 + (5248 + x2), xmask)
    tmp163 = tl.load(in_ptr1 + (41 + x1), xmask, eviction_policy='evict_last')
    tmp166 = tl.load(in_ptr0 + (5376 + x2), xmask)
    tmp167 = tl.load(in_ptr1 + (42 + x1), xmask, eviction_policy='evict_last')
    tmp170 = tl.load(in_ptr0 + (5504 + x2), xmask)
    tmp171 = tl.load(in_ptr1 + (43 + x1), xmask, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr0 + (5632 + x2), xmask)
    tmp175 = tl.load(in_ptr1 + (44 + x1), xmask, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr0 + (5760 + x2), xmask)
    tmp179 = tl.load(in_ptr1 + (45 + x1), xmask, eviction_policy='evict_last')
    tmp182 = tl.load(in_ptr0 + (5888 + x2), xmask)
    tmp183 = tl.load(in_ptr1 + (46 + x1), xmask, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr0 + (6016 + x2), xmask)
    tmp187 = tl.load(in_ptr1 + (47 + x1), xmask, eviction_policy='evict_last')
    tmp190 = tl.load(in_ptr0 + (6144 + x2), xmask)
    tmp191 = tl.load(in_ptr1 + (48 + x1), xmask, eviction_policy='evict_last')
    tmp194 = tl.load(in_ptr0 + (6272 + x2), xmask)
    tmp195 = tl.load(in_ptr1 + (49 + x1), xmask, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (6400 + x2), xmask)
    tmp199 = tl.load(in_ptr1 + (50 + x1), xmask, eviction_policy='evict_last')
    tmp202 = tl.load(in_ptr0 + (6528 + x2), xmask)
    tmp203 = tl.load(in_ptr1 + (51 + x1), xmask, eviction_policy='evict_last')
    tmp206 = tl.load(in_ptr0 + (6656 + x2), xmask)
    tmp207 = tl.load(in_ptr1 + (52 + x1), xmask, eviction_policy='evict_last')
    tmp210 = tl.load(in_ptr0 + (6784 + x2), xmask)
    tmp211 = tl.load(in_ptr1 + (53 + x1), xmask, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr0 + (6912 + x2), xmask)
    tmp215 = tl.load(in_ptr1 + (54 + x1), xmask, eviction_policy='evict_last')
    tmp218 = tl.load(in_ptr0 + (7040 + x2), xmask)
    tmp219 = tl.load(in_ptr1 + (55 + x1), xmask, eviction_policy='evict_last')
    tmp222 = tl.load(in_ptr0 + (7168 + x2), xmask)
    tmp223 = tl.load(in_ptr1 + (56 + x1), xmask, eviction_policy='evict_last')
    tmp226 = tl.load(in_ptr0 + (7296 + x2), xmask)
    tmp227 = tl.load(in_ptr1 + (57 + x1), xmask, eviction_policy='evict_last')
    tmp230 = tl.load(in_ptr0 + (7424 + x2), xmask)
    tmp231 = tl.load(in_ptr1 + (58 + x1), xmask, eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr0 + (7552 + x2), xmask)
    tmp235 = tl.load(in_ptr1 + (59 + x1), xmask, eviction_policy='evict_last')
    tmp238 = tl.load(in_ptr0 + (7680 + x2), xmask)
    tmp239 = tl.load(in_ptr1 + (60 + x1), xmask, eviction_policy='evict_last')
    tmp242 = tl.load(in_ptr0 + (7808 + x2), xmask)
    tmp243 = tl.load(in_ptr1 + (61 + x1), xmask, eviction_policy='evict_last')
    tmp246 = tl.load(in_ptr0 + (7936 + x2), xmask)
    tmp247 = tl.load(in_ptr1 + (62 + x1), xmask, eviction_policy='evict_last')
    tmp250 = tl.load(in_ptr0 + (8064 + x2), xmask)
    tmp251 = tl.load(in_ptr1 + (63 + x1), xmask, eviction_policy='evict_last')
    tmp254 = tl.load(in_ptr0 + (8192 + x2), xmask)
    tmp255 = tl.load(in_ptr1 + (64 + x1), xmask, eviction_policy='evict_last')
    tmp258 = tl.load(in_ptr0 + (8320 + x2), xmask)
    tmp259 = tl.load(in_ptr1 + (65 + x1), xmask, eviction_policy='evict_last')
    tmp262 = tl.load(in_ptr0 + (8448 + x2), xmask)
    tmp263 = tl.load(in_ptr1 + (66 + x1), xmask, eviction_policy='evict_last')
    tmp266 = tl.load(in_ptr0 + (8576 + x2), xmask)
    tmp267 = tl.load(in_ptr1 + (67 + x1), xmask, eviction_policy='evict_last')
    tmp270 = tl.load(in_ptr0 + (8704 + x2), xmask)
    tmp271 = tl.load(in_ptr1 + (68 + x1), xmask, eviction_policy='evict_last')
    tmp274 = tl.load(in_ptr0 + (8832 + x2), xmask)
    tmp275 = tl.load(in_ptr1 + (69 + x1), xmask, eviction_policy='evict_last')
    tmp278 = tl.load(in_ptr0 + (8960 + x2), xmask)
    tmp279 = tl.load(in_ptr1 + (70 + x1), xmask, eviction_policy='evict_last')
    tmp282 = tl.load(in_ptr0 + (9088 + x2), xmask)
    tmp283 = tl.load(in_ptr1 + (71 + x1), xmask, eviction_policy='evict_last')
    tmp286 = tl.load(in_ptr0 + (9216 + x2), xmask)
    tmp287 = tl.load(in_ptr1 + (72 + x1), xmask, eviction_policy='evict_last')
    tmp290 = tl.load(in_ptr0 + (9344 + x2), xmask)
    tmp291 = tl.load(in_ptr1 + (73 + x1), xmask, eviction_policy='evict_last')
    tmp294 = tl.load(in_ptr0 + (9472 + x2), xmask)
    tmp295 = tl.load(in_ptr1 + (74 + x1), xmask, eviction_policy='evict_last')
    tmp298 = tl.load(in_ptr0 + (9600 + x2), xmask)
    tmp299 = tl.load(in_ptr1 + (75 + x1), xmask, eviction_policy='evict_last')
    tmp302 = tl.load(in_ptr0 + (9728 + x2), xmask)
    tmp303 = tl.load(in_ptr1 + (76 + x1), xmask, eviction_policy='evict_last')
    tmp306 = tl.load(in_ptr0 + (9856 + x2), xmask)
    tmp307 = tl.load(in_ptr1 + (77 + x1), xmask, eviction_policy='evict_last')
    tmp310 = tl.load(in_ptr0 + (9984 + x2), xmask)
    tmp311 = tl.load(in_ptr1 + (78 + x1), xmask, eviction_policy='evict_last')
    tmp314 = tl.load(in_ptr0 + (10112 + x2), xmask)
    tmp315 = tl.load(in_ptr1 + (79 + x1), xmask, eviction_policy='evict_last')
    tmp318 = tl.load(in_ptr0 + (10240 + x2), xmask)
    tmp319 = tl.load(in_ptr1 + (80 + x1), xmask, eviction_policy='evict_last')
    tmp322 = tl.load(in_ptr0 + (10368 + x2), xmask)
    tmp323 = tl.load(in_ptr1 + (81 + x1), xmask, eviction_policy='evict_last')
    tmp326 = tl.load(in_ptr0 + (10496 + x2), xmask)
    tmp327 = tl.load(in_ptr1 + (82 + x1), xmask, eviction_policy='evict_last')
    tmp330 = tl.load(in_ptr0 + (10624 + x2), xmask)
    tmp331 = tl.load(in_ptr1 + (83 + x1), xmask, eviction_policy='evict_last')
    tmp334 = tl.load(in_ptr0 + (10752 + x2), xmask)
    tmp335 = tl.load(in_ptr1 + (84 + x1), xmask, eviction_policy='evict_last')
    tmp338 = tl.load(in_ptr0 + (10880 + x2), xmask)
    tmp339 = tl.load(in_ptr1 + (85 + x1), xmask, eviction_policy='evict_last')
    tmp342 = tl.load(in_ptr0 + (11008 + x2), xmask)
    tmp343 = tl.load(in_ptr1 + (86 + x1), xmask, eviction_policy='evict_last')
    tmp346 = tl.load(in_ptr0 + (11136 + x2), xmask)
    tmp347 = tl.load(in_ptr1 + (87 + x1), xmask, eviction_policy='evict_last')
    tmp350 = tl.load(in_ptr0 + (11264 + x2), xmask)
    tmp351 = tl.load(in_ptr1 + (88 + x1), xmask, eviction_policy='evict_last')
    tmp354 = tl.load(in_ptr0 + (11392 + x2), xmask)
    tmp355 = tl.load(in_ptr1 + (89 + x1), xmask, eviction_policy='evict_last')
    tmp358 = tl.load(in_ptr0 + (11520 + x2), xmask)
    tmp359 = tl.load(in_ptr1 + (90 + x1), xmask, eviction_policy='evict_last')
    tmp362 = tl.load(in_ptr0 + (11648 + x2), xmask)
    tmp363 = tl.load(in_ptr1 + (91 + x1), xmask, eviction_policy='evict_last')
    tmp366 = tl.load(in_ptr0 + (11776 + x2), xmask)
    tmp367 = tl.load(in_ptr1 + (92 + x1), xmask, eviction_policy='evict_last')
    tmp370 = tl.load(in_ptr0 + (11904 + x2), xmask)
    tmp371 = tl.load(in_ptr1 + (93 + x1), xmask, eviction_policy='evict_last')
    tmp374 = tl.load(in_ptr0 + (12032 + x2), xmask)
    tmp375 = tl.load(in_ptr1 + (94 + x1), xmask, eviction_policy='evict_last')
    tmp378 = tl.load(in_ptr0 + (12160 + x2), xmask)
    tmp379 = tl.load(in_ptr1 + (95 + x1), xmask, eviction_policy='evict_last')
    tmp382 = tl.load(in_ptr0 + (12288 + x2), xmask)
    tmp383 = tl.load(in_ptr1 + (96 + x1), xmask, eviction_policy='evict_last')
    tmp386 = tl.load(in_ptr0 + (12416 + x2), xmask)
    tmp387 = tl.load(in_ptr1 + (97 + x1), xmask, eviction_policy='evict_last')
    tmp390 = tl.load(in_ptr0 + (12544 + x2), xmask)
    tmp391 = tl.load(in_ptr1 + (98 + x1), xmask, eviction_policy='evict_last')
    tmp394 = tl.load(in_ptr0 + (12672 + x2), xmask)
    tmp395 = tl.load(in_ptr1 + (99 + x1), xmask, eviction_policy='evict_last')
    tmp398 = tl.load(in_ptr0 + (12800 + x2), xmask)
    tmp399 = tl.load(in_ptr1 + (100 + x1), xmask, eviction_policy='evict_last')
    tmp402 = tl.load(in_ptr0 + (12928 + x2), xmask)
    tmp403 = tl.load(in_ptr1 + (101 + x1), xmask, eviction_policy='evict_last')
    tmp406 = tl.load(in_ptr0 + (13056 + x2), xmask)
    tmp407 = tl.load(in_ptr1 + (102 + x1), xmask, eviction_policy='evict_last')
    tmp410 = tl.load(in_ptr0 + (13184 + x2), xmask)
    tmp411 = tl.load(in_ptr1 + (103 + x1), xmask, eviction_policy='evict_last')
    tmp414 = tl.load(in_ptr0 + (13312 + x2), xmask)
    tmp415 = tl.load(in_ptr1 + (104 + x1), xmask, eviction_policy='evict_last')
    tmp418 = tl.load(in_ptr0 + (13440 + x2), xmask)
    tmp419 = tl.load(in_ptr1 + (105 + x1), xmask, eviction_policy='evict_last')
    tmp422 = tl.load(in_ptr0 + (13568 + x2), xmask)
    tmp423 = tl.load(in_ptr1 + (106 + x1), xmask, eviction_policy='evict_last')
    tmp426 = tl.load(in_ptr0 + (13696 + x2), xmask)
    tmp427 = tl.load(in_ptr1 + (107 + x1), xmask, eviction_policy='evict_last')
    tmp430 = tl.load(in_ptr0 + (13824 + x2), xmask)
    tmp431 = tl.load(in_ptr1 + (108 + x1), xmask, eviction_policy='evict_last')
    tmp434 = tl.load(in_ptr0 + (13952 + x2), xmask)
    tmp435 = tl.load(in_ptr1 + (109 + x1), xmask, eviction_policy='evict_last')
    tmp438 = tl.load(in_ptr0 + (14080 + x2), xmask)
    tmp439 = tl.load(in_ptr1 + (110 + x1), xmask, eviction_policy='evict_last')
    tmp442 = tl.load(in_ptr0 + (14208 + x2), xmask)
    tmp443 = tl.load(in_ptr1 + (111 + x1), xmask, eviction_policy='evict_last')
    tmp446 = tl.load(in_ptr0 + (14336 + x2), xmask)
    tmp447 = tl.load(in_ptr1 + (112 + x1), xmask, eviction_policy='evict_last')
    tmp450 = tl.load(in_ptr0 + (14464 + x2), xmask)
    tmp451 = tl.load(in_ptr1 + (113 + x1), xmask, eviction_policy='evict_last')
    tmp454 = tl.load(in_ptr0 + (14592 + x2), xmask)
    tmp455 = tl.load(in_ptr1 + (114 + x1), xmask, eviction_policy='evict_last')
    tmp458 = tl.load(in_ptr0 + (14720 + x2), xmask)
    tmp459 = tl.load(in_ptr1 + (115 + x1), xmask, eviction_policy='evict_last')
    tmp462 = tl.load(in_ptr0 + (14848 + x2), xmask)
    tmp463 = tl.load(in_ptr1 + (116 + x1), xmask, eviction_policy='evict_last')
    tmp466 = tl.load(in_ptr0 + (14976 + x2), xmask)
    tmp467 = tl.load(in_ptr1 + (117 + x1), xmask, eviction_policy='evict_last')
    tmp470 = tl.load(in_ptr0 + (15104 + x2), xmask)
    tmp471 = tl.load(in_ptr1 + (118 + x1), xmask, eviction_policy='evict_last')
    tmp474 = tl.load(in_ptr0 + (15232 + x2), xmask)
    tmp475 = tl.load(in_ptr1 + (119 + x1), xmask, eviction_policy='evict_last')
    tmp478 = tl.load(in_ptr0 + (15360 + x2), xmask)
    tmp479 = tl.load(in_ptr1 + (120 + x1), xmask, eviction_policy='evict_last')
    tmp482 = tl.load(in_ptr0 + (15488 + x2), xmask)
    tmp483 = tl.load(in_ptr1 + (121 + x1), xmask, eviction_policy='evict_last')
    tmp486 = tl.load(in_ptr0 + (15616 + x2), xmask)
    tmp487 = tl.load(in_ptr1 + (122 + x1), xmask, eviction_policy='evict_last')
    tmp490 = tl.load(in_ptr0 + (15744 + x2), xmask)
    tmp491 = tl.load(in_ptr1 + (123 + x1), xmask, eviction_policy='evict_last')
    tmp494 = tl.load(in_ptr0 + (15872 + x2), xmask)
    tmp495 = tl.load(in_ptr1 + (124 + x1), xmask, eviction_policy='evict_last')
    tmp498 = tl.load(in_ptr0 + (16000 + x2), xmask)
    tmp499 = tl.load(in_ptr1 + (125 + x1), xmask, eviction_policy='evict_last')
    tmp502 = tl.load(in_ptr0 + (16128 + x2), xmask)
    tmp503 = tl.load(in_ptr1 + (126 + x1), xmask, eviction_policy='evict_last')
    tmp506 = tl.load(in_ptr0 + (16256 + x2), xmask)
    tmp507 = tl.load(in_ptr1 + (127 + x1), xmask, eviction_policy='evict_last')
    tmp509 = tmp0 + tmp1
    tmp511 = tmp3 + tmp4
    tmp513 = tmp6 + tmp7
    tmp515 = tmp10 + tmp11
    tmp517 = tmp14 + tmp15
    tmp519 = tmp18 + tmp19
    tmp521 = tmp22 + tmp23
    tmp523 = tmp26 + tmp27
    tmp525 = tmp30 + tmp31
    tmp527 = tmp34 + tmp35
    tmp529 = tmp38 + tmp39
    tmp531 = tmp42 + tmp43
    tmp533 = tmp46 + tmp47
    tmp535 = tmp50 + tmp51
    tmp537 = tmp54 + tmp55
    tmp539 = tmp58 + tmp59
    tmp541 = tmp62 + tmp63
    tmp543 = tmp66 + tmp67
    tmp545 = tmp70 + tmp71
    tmp547 = tmp74 + tmp75
    tmp549 = tmp78 + tmp79
    tmp551 = tmp82 + tmp83
    tmp553 = tmp86 + tmp87
    tmp555 = tmp90 + tmp91
    tmp557 = tmp94 + tmp95
    tmp559 = tmp98 + tmp99
    tmp561 = tmp102 + tmp103
    tmp563 = tmp106 + tmp107
    tmp565 = tmp110 + tmp111
    tmp567 = tmp114 + tmp115
    tmp569 = tmp118 + tmp119
    tmp571 = tmp122 + tmp123
    tmp573 = tmp126 + tmp127
    tmp575 = tmp130 + tmp131
    tmp577 = tmp134 + tmp135
    tmp579 = tmp138 + tmp139
    tmp581 = tmp142 + tmp143
    tmp583 = tmp146 + tmp147
    tmp585 = tmp150 + tmp151
    tmp587 = tmp154 + tmp155
    tmp589 = tmp158 + tmp159
    tmp591 = tmp162 + tmp163
    tmp593 = tmp166 + tmp167
    tmp595 = tmp170 + tmp171
    tmp597 = tmp174 + tmp175
    tmp599 = tmp178 + tmp179
    tmp601 = tmp182 + tmp183
    tmp603 = tmp186 + tmp187
    tmp605 = tmp190 + tmp191
    tmp607
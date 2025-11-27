import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256 % 64
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 4096), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (768 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (1024 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (1280 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (1536 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (1792 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (2048 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (2304 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (2560 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (2816 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (3072 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (3328 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (3584 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (3840 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (4096 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (4352 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (4608 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (4864 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (5120 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (5376 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (5632 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (5888 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (6144 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (6400 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (6656 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (6912 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (7168 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (7424 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (7680 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (7936 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (8192 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (8448 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (8704 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (8960 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (9216 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (9472 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (9728 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (9984 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (10240 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (10496 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (10752 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (11008 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (11264 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (11520 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (11776 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (12032 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (12288 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (12544 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (12800 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (13056 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (13312 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (13568 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (13824 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (14080 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (14336 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (14592 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (14848 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (15104 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (15360 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (15616 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (15872 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (16128 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (16384 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (16640 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (16896 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (17152 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (17408 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (17664 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (17920 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (18176 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (18432 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (18688 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (18944 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (19200 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (19456 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (19712 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (19968 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (20224 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (20480 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (20736 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (20992 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (21248 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (21504 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (21760 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (22016 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (22272 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (22528 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (22784 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (23040 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (23296 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (23552 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (23808 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (24064 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (24320 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (24576 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (24832 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (25088 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (25344 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (25600 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (25856 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (26112 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (26368 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (26624 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (26880 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (27136 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (27392 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (27648 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (27904 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (28160 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (28416 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (28672 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (28928 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (29184 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (29440 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (29696 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (29952 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (30208 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (30464 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (30720 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (30976 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (31232 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (31488 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (31744 + x1 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (31999 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (32255 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (32511 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (32767 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (33023 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (33279 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (33535 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (33791 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (34047 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (34303 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (34559 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (34815 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (35071 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (35327 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (35583 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (35839 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (36095 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (36351 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (36607 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (36863 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp435 = tl.load(in_ptr0 + (37119 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (37375 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp441 = tl.load(in_ptr0 + (37631 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp444 = tl.load(in_ptr0 + (37887 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp447 = tl.load(in_ptr0 + (38143 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp450 = tl.load(in_ptr0 + (38399 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp453 = tl.load(in_ptr0 + (38655 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp456 = tl.load(in_ptr0 + (38911 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp459 = tl.load(in_ptr0 + (39167 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp462 = tl.load(in_ptr0 + (39423 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp465 = tl.load(in_ptr0 + (39679 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp468 = tl.load(in_ptr0 + (39935 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp471 = tl.load(in_ptr0 + (40191 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp474 = tl.load(in_ptr0 + (40447 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp477 = tl.load(in_ptr0 + (40703 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp480 = tl.load(in_ptr0 + (40959 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp483 = tl.load(in_ptr0 + (41215 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp486 = tl.load(in_ptr0 + (41471 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp489 = tl.load(in_ptr0 + (41727 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp492 = tl.load(in_ptr0 + (41983 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp495 = tl.load(in_ptr0 + (42239 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp498 = tl.load(in_ptr0 + (42495 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp501 = tl.load(in_ptr0 + (42751 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp504 = tl.load(in_ptr0 + (43007 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp507 = tl.load(in_ptr0 + (43263 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp510 = tl.load(in_ptr0 + (43519 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp513 = tl.load(in_ptr0 + (43775 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp516 = tl.load(in_ptr0 + (44031 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp519 = tl.load(in_ptr0 + (44287 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp522 = tl.load(in_ptr0 + (44543 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp525 = tl.load(in_ptr0 + (44799 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp528 = tl.load(in_ptr0 + (45055 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp531 = tl.load(in_ptr0 + (45311 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp534 = tl.load(in_ptr0 + (45567 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp537 = tl.load(in_ptr0 + (45823 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp540 = tl.load(in_ptr0 + (46079 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp543 = tl.load(in_ptr0 + (46335 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp546 = tl.load(in_ptr0 + (46591 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp549 = tl.load(in_ptr0 + (46847 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp552 = tl.load(in_ptr0 + (47103 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp555 = tl.load(in_ptr0 + (47359 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp558 = tl.load(in_ptr0 + (47615 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp561 = tl.load(in_ptr0 + (47871 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp564 = tl.load(in_ptr0 + (48127 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp567 = tl.load(in_ptr0 + (48383 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp570 = tl.load(in_ptr0 + (48639 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp573 = tl.load(in_ptr0 + (48895 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp576 = tl.load(in_ptr0 + (49151 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp579 = tl.load(in_ptr0 + (49407 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp582 = tl.load(in_ptr0 + (49663 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp585 = tl.load(in_ptr0 + (49919 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp588 = tl.load(in_ptr0 + (50175 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp591 = tl.load(in_ptr0 + (50431 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp594 = tl.load(in_ptr0 + (50687 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp597 = tl.load(in_ptr0 + (50943 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp600 = tl.load(in_ptr0 + (51199 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp603 = tl.load(in_ptr0 + (51455 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp606 = tl.load(in_ptr0 + (51711 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp609 = tl.load(in_ptr0 + (51967 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp612 = tl.load(in_ptr0 + (52223 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp615 = tl.load(in_ptr0 + (52479 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp618 = tl.load(in_ptr0 + (52735 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp621 = tl
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 64 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 64 * x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (64 + 64 * x1), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (65 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (66 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (67 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (128 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (129 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (130 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (131 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (192 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (193 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (194 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (195 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (256 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (257 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (258 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (259 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (320 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (321 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (322 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (323 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (384 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (385 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (386 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (387 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (448 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (449 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (450 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (451 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (512 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (513 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (514 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (515 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (576 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (577 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (578 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (579 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (640 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (641 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (642 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (643 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (704 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (705 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (706 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (707 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (768 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (769 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (770 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (771 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (832 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (833 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (834 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (835 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (896 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (897 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (898 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (899 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (960 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (961 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (962 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (963 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (1024 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (1025 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (1026 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (1027 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (1088 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (1089 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (1090 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (1091 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (1152 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (1153 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (1154 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (1155 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (1216 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (1217 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (1218 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (1219 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (1280 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (1281 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (1282 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (1283 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (1344 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (1345 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (1346 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (1347 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (1408 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (1409 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (1410 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (1411 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (1472 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (1473 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (1474 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (1475 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (1536 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (1537 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (1538 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (1539 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (1600 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (1601 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (1602 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (1603 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (1664 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (1665 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (1666 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (1667 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (1728 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (1729 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (1730 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (1731 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (1792 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (1793 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (1794 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (1795 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (1856 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (1857 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (1858 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (1859 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (1920 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (1921 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (1922 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (1923 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (1984 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (1985 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (1986 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (1987 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (2048 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (2049 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (2050 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (2051 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (2112 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (2113 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (2114 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (2115 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (2176 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (2177 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (2178 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (2179 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (2240 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (2241 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (2242 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (2243 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (2304 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp435 = tl.load(in_ptr0 + (2305 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (2306 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp441 = tl.load(in_ptr0 + (2307 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp444 = tl.load(in_ptr0 + (2368 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp447 = tl.load(in_ptr0 + (2369 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp450 = tl.load(in_ptr0 + (2370 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp453 = tl.load(in_ptr0 + (2371 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp456 = tl.load(in_ptr0 + (2432 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp459 = tl.load(in_ptr0 + (2433 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp462 = tl.load(in_ptr0 + (2434 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp465 = tl.load(in_ptr0 + (2435 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp468 = tl.load(in_ptr0 + (2496 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp471 = tl.load(in_ptr0 + (2497 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp474 = tl.load(in_ptr0 + (2498 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp477 = tl.load(in_ptr0 + (2499 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp480 = tl.load(in_ptr0 + (2560 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp483 = tl.load(in_ptr0 + (2561 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp486 = tl.load(in_ptr0 + (2562 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp489 = tl.load(in_ptr0 + (2563 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp492 = tl.load(in_ptr0 + (2624 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp495 = tl.load(in_ptr0 + (2625 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp498 = tl.load(in_ptr0 + (2626 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp501 = tl.load(in_ptr0 + (2627 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp504 = tl.load(in_ptr0 + (2688 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp507 = tl.load(in_ptr0 + (2689 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp510 = tl.load(in_ptr0 + (2690 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp513 = tl.load(in_ptr0 + (2691 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp516 = tl.load(in_ptr0 + (2752 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp519 = tl.load(in_ptr0 + (2753 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp522 = tl.load(in_ptr0 + (2754 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp525 = tl.load(in_ptr0 + (2755 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp528 = tl.load(in_ptr0 + (2816 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp531 = tl.load(in_ptr0 + (2817 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp534 = tl.load(in_ptr0 + (2818 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp537 = tl.load(in_ptr0 + (2819 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp540 = tl.load(in_ptr0 + (2880 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp543 = tl.load(in_ptr0 + (2881 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp546 = tl.load(in_ptr0 + (2882 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp549 = tl.load(in_ptr0 + (2883 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp552 = tl.load(in_ptr0 + (2944 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp555 = tl.load(in_ptr0 + (2945 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp558 = tl.load(in_ptr0 + (2946 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp561 = tl.load(in_ptr0 + (2947 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp564 = tl.load(in_ptr0 + (3008 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp567 = tl.load(in_ptr0 + (3009 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp570 = tl.load(in_ptr0 + (3010 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp573 = tl.load(in_ptr0 + (3011 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp576 = tl.load(in_ptr0 + (3072 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp579 = tl.load(in_ptr0 + (3073 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp582 = tl.load(in_ptr0 + (3074 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp585 = tl.load(in_ptr0 + (3075 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp588 = tl.load(in_ptr0 + (3136 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp591 = tl.load(in_ptr0 + (3137 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp594 = tl.load(in_ptr0 + (3138 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp597 = tl.load(in_ptr0 + (3139 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp600 = tl.load(in_ptr0 + (3200 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp603 = tl.load(in_ptr0 + (3201 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp606 = tl.load(in_ptr0 + (3202 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp609 = tl.load(in_ptr0 + (3203 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp612 = tl.load(in_ptr0 + (3264 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp615 = tl.load(in_ptr0 + (3265 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp618 = tl.load(in_ptr0 + (3266 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp621 = tl.load(in_ptr0 + (3267 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp624 = tl.load(in_ptr0 + (3328 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp627 = tl.load(in_ptr0 + (3329 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp630 = tl.load(in_ptr0 + (3330 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp633 = tl.load(in_ptr0 + (3331 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp636 = tl.load(in_ptr0 + (3392 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp639 = tl.load(in_ptr0 + (3393 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp642 = tl.load(in_ptr0 + (3394 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp645 = tl.load(in_ptr0 + (3395 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp648 = tl.load(in_ptr0 + (3456 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp651 = tl.load(in_ptr0 + (3457 + 64 * x1), xmask, eviction_policy=
        'evict_last')
    tmp654
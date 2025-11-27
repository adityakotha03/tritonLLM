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
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.
    constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x2 + 4096 * y1), xmask & ymask,
        eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 64 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_leaky_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256 % 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 16384), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16384 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32768 + x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (49152 + x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (65536 + x1), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (81920 + x1), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (98304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (114688 + x1), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (131072 + x1), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (147456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr0 + (163840 + x1), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (180224 + x1), xmask, eviction_policy='evict_last'
        )
    tmp36 = tl.load(in_ptr0 + (196608 + x1), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (212992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp42 = tl.load(in_ptr0 + (229376 + x1), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (245760 + x1), xmask, eviction_policy='evict_last'
        )
    tmp48 = tl.load(in_ptr0 + (262144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (278528 + x1), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr0 + (294912 + x1), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (311392 + x1), xmask, eviction_policy='evict_last'
        )
    tmp60 = tl.load(in_ptr0 + (327776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (344160 + x1), xmask, eviction_policy='evict_last'
        )
    tmp66 = tl.load(in_ptr0 + (360544 + x1), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (376928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp72 = tl.load(in_ptr0 + (393312 + x1), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (409696 + x1), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr0 + (426080 + x1), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (442464 + x1), xmask, eviction_policy='evict_last'
        )
    tmp84 = tl.load(in_ptr0 + (458848 + x1), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (475232 + x1), xmask, eviction_policy='evict_last'
        )
    tmp90 = tl.load(in_ptr0 + (491616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (507992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp96 = tl.load(in_ptr0 + (524368 + x1), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (540744 + x1), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr0 + (557120 + x1), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (573496 + x1), xmask, eviction_policy='evict_last'
        )
    tmp108 = tl.load(in_ptr0 + (589872 + x1), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (606248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp114 = tl.load(in_ptr0 + (622624 + x1), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (638992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp120 = tl.load(in_ptr0 + (655360 + x1), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (671728 + x1), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr0 + (688096 + x1), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (704464 + x1), xmask, eviction_policy='evict_last'
        )
    tmp132 = tl.load(in_ptr0 + (720832 + x1), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (737200 + x1), xmask, eviction_policy='evict_last'
        )
    tmp138 = tl.load(in_ptr0 + (753568 + x1), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (769936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp144 = tl.load(in_ptr0 + (786304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (802672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr0 + (819040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (835408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp156 = tl.load(in_ptr0 + (851776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (868144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp162 = tl.load(in_ptr0 + (884512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (900880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp168 = tl.load(in_ptr0 + (917248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (933616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr0 + (949984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (966352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp180 = tl.load(in_ptr0 + (982720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (999088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp186 = tl.load(in_ptr0 + (1015456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (1031824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp192 = tl.load(in_ptr0 + (1048192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (1064560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr0 + (1080928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (1097296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp204 = tl.load(in_ptr0 + (1113664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (1129936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp210 = tl.load(in_ptr0 + (1146304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (1162672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp216 = tl.load(in_ptr0 + (1178944 + x1), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (1195312 + x1), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr0 + (1211680 + x1), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (1228048 + x1), xmask, eviction_policy='evict_last'
        )
    tmp228 = tl.load(in_ptr0 + (1244416 + x1), xmask, eviction_policy='evict_last'
        )
    tmp231 = tl.load(in_ptr0 + (1260784 + x1), xmask, eviction_policy='evict_last'
        )
    tmp234 = tl.load(in_ptr0 + (1277152 + x1), xmask, eviction_policy='evict_last'
        )
    tmp237 = tl.load(in_ptr0 + (1293520 + x1), xmask, eviction_policy='evict_last'
        )
    tmp240 = tl.load(in_ptr0 + (1309888 + x1), xmask, eviction_policy='evict_last'
        )
    tmp243 = tl.load(in_ptr0 + (1326256 + x1), xmask, eviction_policy='evict_last'
        )
    tmp246 = tl.load(in_ptr0 + (1342624 + x1), xmask, eviction_policy='evict_last'
        )
    tmp249 = tl.load(in_ptr0 + (1358992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp252 = tl.load(in_ptr0 + (1375360 + x1), xmask, eviction_policy='evict_last'
        )
    tmp255 = tl.load(in_ptr0 + (1391728 + x1), xmask, eviction_policy='evict_last'
        )
    tmp258 = tl.load(in_ptr0 + (1408096 + x1), xmask, eviction_policy='evict_last'
        )
    tmp261 = tl.load(in_ptr0 + (1424464 + x1), xmask, eviction_policy='evict_last'
        )
    tmp264 = tl.load(in_ptr0 + (1440832 + x1), xmask, eviction_policy='evict_last'
        )
    tmp267 = tl.load(in_ptr0 + (1457200 + x1), xmask, eviction_policy='evict_last'
        )
    tmp270 = tl.load(in_ptr0 + (1473568 + x1), xmask, eviction_policy='evict_last'
        )
    tmp273 = tl.load(in_ptr0 + (1489936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp276 = tl.load(in_ptr0 + (1506304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp279 = tl.load(in_ptr0 + (1522672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp282 = tl.load(in_ptr0 + (1539040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp285 = tl.load(in_ptr0 + (1555408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp288 = tl.load(in_ptr0 + (1571776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp291 = tl.load(in_ptr0 + (1588144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp294 = tl.load(in_ptr0 + (1604512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp297 = tl.load(in_ptr0 + (1620880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp300 = tl.load(in_ptr0 + (1637248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp303 = tl.load(in_ptr0 + (1653616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp306 = tl.load(in_ptr0 + (1669984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp309 = tl.load(in_ptr0 + (1686352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp312 = tl.load(in_ptr0 + (1702720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp315 = tl.load(in_ptr0 + (1719088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp318 = tl.load(in_ptr0 + (1735456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp321 = tl.load(in_ptr0 + (1751824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp324 = tl.load(in_ptr0 + (1768192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp327 = tl.load(in_ptr0 + (1784560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp330 = tl.load(in_ptr0 + (1800928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp333 = tl.load(in_ptr0 + (1817296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp336 = tl.load(in_ptr0 + (1833664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp339 = tl.load(in_ptr0 + (1849936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp342 = tl.load(in_ptr0 + (1866304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp345 = tl.load(in_ptr0 + (1882672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp348 = tl.load(in_ptr0 + (1899040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp351 = tl.load(in_ptr0 + (1915408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp354 = tl.load(in_ptr0 + (1931776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp357 = tl.load(in_ptr0 + (1948144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp360 = tl.load(in_ptr0 + (1964512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp363 = tl.load(in_ptr0 + (1980880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp366 = tl.load(in_ptr0 + (1997248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp369 = tl.load(in_ptr0 + (2013616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp372 = tl.load(in_ptr0 + (2029984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp375 = tl.load(in_ptr0 + (2046352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp378 = tl.load(in_ptr0 + (2062720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp381 = tl.load(in_ptr0 + (2079088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp384 = tl.load(in_ptr0 + (2095456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp387 = tl.load(in_ptr0 + (2111824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp390 = tl.load(in_ptr0 + (2128192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp393 = tl.load(in_ptr0 + (2144560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp396 = tl.load(in_ptr0 + (2160928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp399 = tl.load(in_ptr0 + (2177296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp402 = tl.load(in_ptr0 + (2193664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp405 = tl.load(in_ptr0 + (2209936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp408 = tl.load(in_ptr0 + (2226304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp411 = tl.load(in_ptr0 + (2242672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp414 = tl.load(in_ptr0 + (2258944 + x1), xmask, eviction_policy='evict_last'
        )
    tmp417 = tl.load(in_ptr0 + (2275312 + x1), xmask, eviction_policy='evict_last'
        )
    tmp420 = tl.load(in_ptr0 + (2291680 + x1), xmask, eviction_policy='evict_last'
        )
    tmp423 = tl.load(in_ptr0 + (2308048 + x1), xmask, eviction_policy='evict_last'
        )
    tmp426 = tl.load(in_ptr0 + (2324416 + x1), xmask, eviction_policy='evict_last'
        )
    tmp429 = tl.load(in_ptr0 + (2340784 + x1), xmask, eviction_policy='evict_last'
        )
    tmp432 = tl.load(in_ptr0 + (2357152 + x1), xmask, eviction_policy='evict_last'
        )
    tmp435 = tl.load(in_ptr0 + (2373520 + x1), xmask, eviction_policy='evict_last'
        )
    tmp438 = tl.load(in_ptr0 + (2389888 + x1), xmask, eviction_policy='evict_last'
        )
    tmp441 = tl.load(in_ptr0 + (2406256 + x1), xmask, eviction_policy='evict_last'
        )
    tmp444 = tl.load(in_ptr0 + (2422624 + x1), xmask, eviction_policy='evict_last'
        )
    tmp447 = tl.load(in_ptr0 + (2438992 + x1), xmask, eviction_policy='evict_last'
        )
    tmp450 = tl.load(in_ptr0 + (2455360 + x1), xmask, eviction_policy='evict_last'
        )
    tmp453 = tl.load(in_ptr0 + (2471728 + x1), xmask, eviction_policy='evict_last'
        )
    tmp456 = tl.load(in_ptr0 + (2488096 + x1), xmask, eviction_policy='evict_last'
        )
    tmp459 = tl.load(in_ptr0 + (2504464 + x1), xmask, eviction_policy='evict_last'
        )
    tmp462 = tl.load(in_ptr0 + (2520832 + x1), xmask, eviction_policy='evict_last'
        )
    tmp465 = tl.load(in_ptr0 + (2537200 + x1), xmask, eviction_policy='evict_last'
        )
    tmp468 = tl.load(in_ptr0 + (2553568 + x1), xmask, eviction_policy='evict_last'
        )
    tmp471 = tl.load(in_ptr0 + (2569936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp474 = tl.load(in_ptr0 + (2586304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp477 = tl.load(in_ptr0 + (2602672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp480 = tl.load(in_ptr0 + (2619040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp483 = tl.load(in_ptr0 + (2635408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp486 = tl.load(in_ptr0 + (2651776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp489 = tl.load(in_ptr0 + (2668144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp492 = tl.load(in_ptr0 + (2684512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp495 = tl.load(in_ptr0 + (2700880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp498 = tl.load(in_ptr0 + (2717248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp501 = tl.load(in_ptr0 + (2733616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp504 = tl.load(in_ptr0 + (2749984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp507 = tl.load(in_ptr0 + (2766352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp510 = tl.load(in_ptr0 + (2782720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp513 = tl.load(in_ptr0 + (2799088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp516 = tl.load(in_ptr0 + (2815456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp519 = tl.load(in_ptr0 + (2831824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp522 = tl.load(in_ptr0 + (2848192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp525 = tl.load(in_ptr0 + (2864560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp528 = tl.load(in_ptr0 + (2880928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp531 = tl.load(in_ptr0 + (2897296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp534 = tl.load(in_ptr0 + (2913664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp537 = tl.load(in_ptr0 + (2929936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp540 = tl.load(in_ptr0 + (2946304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp543 = tl.load(in_ptr0 + (2962672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp546 = tl.load(in_ptr0 + (2979040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp549 = tl.load(in_ptr0 + (2995408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp552 = tl.load(in_ptr0 + (3011776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp555 = tl.load(in_ptr0 + (3028144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp558 = tl.load(in_ptr0 + (3044512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp561 = tl.load(in_ptr0 + (3060880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp564 = tl.load(in_ptr0 + (3077248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp567 = tl.load(in_ptr0 + (3093616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp570 = tl.load(in_ptr0 + (3109984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp573 = tl.load(in_ptr0 + (3126352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp576 = tl.load(in_ptr0 + (3142720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp579 = tl.load(in_ptr0 + (3159088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp582 = tl.load(in_ptr0 + (3175456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp585 = tl.load(in_ptr0 + (3191824 + x1), xmask, eviction_policy='evict_last'
        )
    tmp588 = tl.load(in_ptr0 + (3208192 + x1), xmask, eviction_policy='evict_last'
        )
    tmp591 = tl.load(in_ptr0 + (3224560 + x1), xmask, eviction_policy='evict_last'
        )
    tmp594 = tl.load(in_ptr0 + (3240928 + x1), xmask, eviction_policy='evict_last'
        )
    tmp597 = tl.load(in_ptr0 + (3257296 + x1), xmask, eviction_policy='evict_last'
        )
    tmp600 = tl.load(in_ptr0 + (3273664 + x1), xmask, eviction_policy='evict_last'
        )
    tmp603 = tl.load(in_ptr0 + (3289936 + x1), xmask, eviction_policy='evict_last'
        )
    tmp606 = tl.load(in_ptr0 + (3306304 + x1), xmask, eviction_policy='evict_last'
        )
    tmp609 = tl.load(in_ptr0 + (3322672 + x1), xmask, eviction_policy='evict_last'
        )
    tmp612 = tl.load(in_ptr0 + (3339040 + x1), xmask, eviction_policy='evict_last'
        )
    tmp615 = tl.load(in_ptr0 + (3355408 + x1), xmask, eviction_policy='evict_last'
        )
    tmp618 = tl.load(in_ptr0 + (3371776 + x1), xmask, eviction_policy='evict_last'
        )
    tmp621 = tl.load(in_ptr0 + (3388144 + x1), xmask, eviction_policy='evict_last'
        )
    tmp624 = tl.load(in_ptr0 + (3404512 + x1), xmask, eviction_policy='evict_last'
        )
    tmp627 = tl.load(in_ptr0 + (3420880 + x1), xmask, eviction_policy='evict_last'
        )
    tmp630 = tl.load(in_ptr0 + (3437248 + x1), xmask, eviction_policy='evict_last'
        )
    tmp633 = tl.load(in_ptr0 + (3453616 + x1), xmask, eviction_policy='evict_last'
        )
    tmp636 = tl.load(in_ptr0 + (3469984 + x1), xmask, eviction_policy='evict_last'
        )
    tmp639 = tl.load(in_ptr0 + (3486352 + x1), xmask, eviction_policy='evict_last'
        )
    tmp642 = tl.load(in_ptr0 + (3502720 + x1), xmask, eviction_policy='evict_last'
        )
    tmp645 = tl.load(in_ptr0 + (3519088 + x1), xmask, eviction_policy='evict_last'
        )
    tmp648 = tl.load(in_ptr0 + (3535456 + x1), xmask, eviction_policy='evict_last'
        )
    tmp651
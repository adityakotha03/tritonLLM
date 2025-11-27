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
def triton_poi_fused__native_batch_norm_legit_0(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 32 % 32
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 20480), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (32 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (96 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (128 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (160 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (192 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (224 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (256 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (288 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (320 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (352 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (384 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (416 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (448 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (480 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (512 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (544 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (576 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (608 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (640 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (672 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (704 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (736 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (768 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (800 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (832 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (864 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (896 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (928 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (960 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (992 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (1024 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (1056 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (1088 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (1120 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (1152 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (1184 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (1216 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (1248 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (1280 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (1312 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (1344 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (1376 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (1408 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (1440 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (1472 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (1504 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (1536 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (1568 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (1600 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (1632 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (1664 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (1696 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (1728 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (1760 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (1792 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (1824 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (1856 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (1888 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (1920 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (1952 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (1984 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (2016 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (2048 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (2080 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (2112 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (2144 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (2176 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (2208 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (2240 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (2272 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (2304 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (2336 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (2368 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (2400 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (2432 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (2464 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (2496 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (2528 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (2560 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (2592 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (2624 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (2656 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (2688 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (2720 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (2752 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (2784 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (2816 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (2848 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (2880 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (2912 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (2944 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (2976 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (3008 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (3040 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (3072 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (3104 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (3136 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (3168 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (3200 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (3232 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (3264 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (3296 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (3328 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (3360 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (3392 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (3424 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (3456 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (3488 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (3520 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (3552 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (3584 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (3616 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (3648 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (3680 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (3712 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (3744 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (3776 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (3808 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (3840 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (3872 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp366 = tl.load(in_ptr0 + (3904 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (3936 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp372 = tl.load(in_ptr0 + (3968 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (4000 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp378 = tl.load(in_ptr0 + (4032 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (4064 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp384 = tl.load(in_ptr0 + (4096 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (4128 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp390 = tl.load(in_ptr0 + (4160 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (4192 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp396 = tl.load(in_ptr0 + (4224 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (4256 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp402 = tl.load(in_ptr0 + (4288 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (4320 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp408 = tl.load(in_ptr0 + (4352 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (4384 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp414 = tl.load(in_ptr0 + (4416 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (4448 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp420 = tl.load(in_ptr0 + (4480 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (4512 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp426 = tl.load(in_ptr0 + (4544 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (4576 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp432 = tl.load(in_ptr0 + (4608 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp435 = tl.load(in_ptr0 + (4640 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp438 = tl.load(in_ptr0 + (4672 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp441 = tl.load(in_ptr0 + (4704 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp444 = tl.load(in_ptr0 + (4736 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp447 = tl.load(in_ptr0 + (4768 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp450 = tl.load(in_ptr0 + (4800 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp453 = tl.load(in_ptr0 + (4832 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp456 = tl.load(in_ptr0 + (4864 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp459 = tl.load(in_ptr0 + (4896 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp462 = tl.load(in_ptr0 + (4928 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp465 = tl.load(in_ptr0 + (4960 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp468 = tl.load(in_ptr0 + (4992 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp471 = tl.load(in_ptr0 + (5024 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp474 = tl.load(in_ptr0 + (5056 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp477 = tl.load(in_ptr0 + (5088 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp480 = tl.load(in_ptr0 + (5120 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp483 = tl.load(in_ptr0 + (5152 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp486 = tl.load(in_ptr0 + (5184 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp489 = tl.load(in_ptr0 + (5216 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp492 = tl.load(in_ptr0 + (5248 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp495 = tl.load(in_ptr0 + (5280 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp498 = tl.load(in_ptr0 + (5312 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp501 = tl.load(in_ptr0 + (5344 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp504 = tl.load(in_ptr0 + (5376 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp507 = tl.load(in_ptr0 + (5408 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp510 = tl.load(in_ptr0 + (5440 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp513 = tl.load(in_ptr0 + (5472 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp516 = tl.load(in_ptr0 + (5504 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp519 = tl.load(in_ptr0 + (5536 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp522 = tl.load(in_ptr0 + (5568 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp525 = tl.load(in_ptr0 + (5600 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp528 = tl.load(in_ptr0 + (5632 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp531 = tl.load(in_ptr0 + (5664 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp534 = tl.load(in_ptr0 + (5696 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp537 = tl.load(in_ptr0 + (5728 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp540 = tl.load(in_ptr0 + (5760 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp543 = tl.load(in_ptr0 + (5792 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp546 = tl.load(in_ptr0 + (5824 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp549 = tl.load(in_ptr0 + (5856 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp552 = tl.load(in_ptr0 + (5888 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp555 = tl.load(in_ptr0 + (5920 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp558 = tl.load(in_ptr0 + (5952 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp561 = tl.load(in_ptr0 + (5984 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp564 = tl.load(in_ptr0 + (6016 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp567 = tl.load(in_ptr0 + (6048 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp570 = tl.load(in_ptr0 + (6080 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp573 = tl.load(in_ptr0 + (6112 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp576 = tl.load(in_ptr0 + (6144 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp579 = tl.load(in_ptr0 + (6176 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp582 = tl.load(in_ptr0 + (6208 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp585 = tl.load(in_ptr0 + (6240 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp588 = tl.load(in_ptr0 + (6272 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp591 = tl.load(in_ptr0 + (6304 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp594 = tl.load(in_ptr0 + (6336 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp597 = tl.load(in_ptr0 + (6368 + x1 + 20480), xmask, eviction_policy=
        'evict_last')
    tmp600 = tl.load(in_ptr0 + (6400 + x1 + 20480), xmask
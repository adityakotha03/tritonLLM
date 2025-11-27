import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 123008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 21952 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_per_fused__native_batch_norm_legit_mul_1(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (64 + r1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (128 + r1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (192 + r1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (256 + r1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (320 + r1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (384 + r1), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (448 + r1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (512 + r1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (576 + r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (640 + r1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr1 + (704 + r1), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (768 + r1), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (832 + r1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (896 + r1), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (960 + r1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr1 + (1024 + r1), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (1088 + r1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (1152 + r1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (1216 + r1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr1 + (1280 + r1), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr1 + (1344 + r1), None, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr1 + (1408 + r1), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr1 + (1472 + r1), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (1536 + r1), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr1 + (1600 + r1), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr1 + (1664 + r1), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr1 + (1728 + r1), None, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr1 + (1792 + r1), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr1 + (1856 + r1), None, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr1 + (1920 + r1), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr1 + (1984 + r1), None, eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr1 + (2048 + r1), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr1 + (2112 + r1), None, eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr1 + (2176 + r1), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr1 + (2240 + r1), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr1 + (2304 + r1), None, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr1 + (2368 + r1), None, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr1 + (2432 + r1), None, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr1 + (2496 + r1), None, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr1 + (2560 + r1), None, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr1 + (2624 + r1), None, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr1 + (2688 + r1), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr1 + (2752 + r1), None, eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr1 + (2816 + r1), None, eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr1 + (2880 + r1), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr1 + (2944 + r1), None, eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr1 + (3008 + r1), None, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr1 + (3072 + r1), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr1 + (3136 + r1), None, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr1 + (3200 + r1), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr1 + (3264 + r1), None, eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr1 + (3328 + r1), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr1 + (3392 + r1), None, eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr1 + (3456 + r1), None, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr1 + (3520 + r1), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr1 + (3584 + r1), None, eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr1 + (3648 + r1), None, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr1 + (3712 + r1), None, eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr1 + (3776 + r1), None, eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr1 + (3840 + r1), None, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr1 + (3904 + r1), None, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr1 + (3968 + r1), None, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr1 + (4032 + r1), None, eviction_policy='evict_last')
    tmp192 = tl.load(in_ptr1 + (4096 + r1), None, eviction_policy='evict_last')
    tmp195 = tl.load(in_ptr1 + (4160 + r1), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr1 + (4224 + r1), None, eviction_policy='evict_last')
    tmp201 = tl.load(in_ptr1 + (4288 + r1), None, eviction_policy='evict_last')
    tmp204 = tl.load(in_ptr1 + (4352 + r1), None, eviction_policy='evict_last')
    tmp207 = tl.load(in_ptr1 + (4416 + r1), None, eviction_policy='evict_last')
    tmp210 = tl.load(in_ptr1 + (4480 + r1), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr1 + (4544 + r1), None, eviction_policy='evict_last')
    tmp216 = tl.load(in_ptr1 + (4608 + r1), None, eviction_policy='evict_last')
    tmp219 = tl.load(in_ptr1 + (4672 + r1), None, eviction_policy='evict_last')
    tmp222 = tl.load(in_ptr1 + (4736 + r1), None, eviction_policy='evict_last')
    tmp225 = tl.load(in_ptr1 + (4800 + r1), None, eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr1 + (4864 + r1), None, eviction_policy='evict_last')
    tmp231 = tl.load(in_ptr1 + (4928 + r1), None, eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr1 + (4992 + r1), None, eviction_policy='evict_last')
    tmp237 = tl.load(in_ptr1 + (5056 + r1), None, eviction_policy='evict_last')
    tmp240 = tl.load(in_ptr1 + (5120 + r1), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr1 + (5184 + r1), None, eviction_policy='evict_last')
    tmp246 = tl.load(in_ptr1 + (5248 + r1), None, eviction_policy='evict_last')
    tmp249 = tl.load(in_ptr1 + (5312 + r1), None, eviction_policy='evict_last')
    tmp252 = tl.load(in_ptr1 + (5376 + r1), None, eviction_policy='evict_last')
    tmp255 = tl.load(in_ptr1 + (5440 + r1), None, eviction_policy='evict_last')
    tmp258 = tl.load(in_ptr1 + (5504 + r1), None, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr1 + (5568 + r1), None, eviction_policy='evict_last')
    tmp264 = tl.load(in_ptr1 + (5632 + r1), None, eviction_policy='evict_last')
    tmp267 = tl.load(in_ptr1 + (5696 + r1), None, eviction_policy='evict_last')
    tmp270 = tl.load(in_ptr1 + (5760 + r1), None, eviction_policy='evict_last')
    tmp273 = tl.load(in_ptr1 + (5824 + r1), None, eviction_policy='evict_last')
    tmp276 = tl.load(in_ptr1 + (5888 + r1), None, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr1 + (5952 + r1), None, eviction_policy='evict_last')
    tmp282 = tl.load(in_ptr1 + (6016 + r1), None, eviction_policy='evict_last')
    tmp285 = tl.load(in_ptr1 + (6080 + r1), None, eviction_policy='evict_last')
    tmp288 = tl.load(in_ptr1 + (6144 + r1), None, eviction_policy='evict_last')
    tmp291 = tl.load(in_ptr1 + (6208 + r1), None, eviction_policy='evict_last')
    tmp294 = tl.load(in_ptr1 + (6272 + r1), None, eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr1 + (6336 + r1), None, eviction_policy='evict_last')
    tmp300 = tl.load(in_ptr1 + (6400 + r1), None, eviction_policy='evict_last')
    tmp303 = tl.load(in_ptr1 + (6464 + r1), None, eviction_policy='evict_last')
    tmp306 = tl.load(in_ptr1 + (6528 + r1), None, eviction_policy='evict_last')
    tmp309 = tl.load(in_ptr1 + (6592 + r1), None, eviction_policy='evict_last')
    tmp312 = tl.load(in_ptr1 + (6656 + r1), None, eviction_policy='evict_last')
    tmp315 = tl.load(in_ptr1 + (6720 + r1), None, eviction_policy='evict_last')
    tmp318 = tl.load(in_ptr1 + (6784 + r1), None, eviction_policy='evict_last')
    tmp321 = tl.load(in_ptr1 + (6848 + r1), None, eviction_policy='evict_last')
    tmp324 = tl.load(in_ptr1 + (6912 + r1), None, eviction_policy='evict_last')
    tmp327 = tl.load(in_ptr1 + (6976 + r1), None, eviction_policy='evict_last')
    tmp330 = tl.load(in_ptr1 + (7040 + r1), None, eviction_policy='evict_last')
    tmp333 = tl.load(in_ptr1 + (7104 + r1), None, eviction_policy='evict_last')
    tmp336 = tl.load(in_ptr1 + (7168 + r1), None, eviction_policy='evict_last')
    tmp339 = tl.load(in_ptr1 + (7232 + r1), None, eviction_policy='evict_last')
    tmp342 = tl.load(in_ptr1 + (7296 + r1), None, eviction_policy='evict_last')
    tmp345 = tl.load(in_ptr1 + (7360 + r1), None, eviction_policy='evict_last')
    tmp348 = tl.load(in_ptr1 + (7424 + r1), None, eviction_policy='evict_last')
    tmp351 = tl.load(in_ptr1 + (7488 + r1), None, eviction_policy='evict_last')
    tmp354 = tl.load(in_ptr1 + (7552 + r1), None, eviction_policy='evict_last')
    tmp357 = tl.load(in_ptr1 + (7616 + r1), None, eviction_policy='evict_last')
    tmp360 = tl.load(in_ptr1 + (7680 + r1), None, eviction_policy='evict_last')
    tmp363 = tl.load(in_ptr1 + (7744 + r1), None, eviction_policy='evict_last')
    tmp366 = tl.load(in_ptr1 + (7808 + r1), None, eviction_policy='evict_last')
    tmp369 = tl.load(in_ptr1 + (7872 + r1), None, eviction_policy='evict_last')
    tmp372 = tl.load(in_ptr1 + (7936 + r1), None, eviction_policy='evict_last')
    tmp375 = tl.load(in_ptr1 + (8000 + r1), None, eviction_policy='evict_last')
    tmp378 = tl.load(in_ptr1 + (8064 + r1), None, eviction_policy='evict_last')
    tmp381 = tl.load(in_ptr1 + (8128 + r1), None, eviction_policy='evict_last')
    tmp384 = tl.load(in_ptr1 + (8192 + r1), None, eviction_policy='evict_last')
    tmp387 = tl.load(in_ptr1 + (8256 + r1), None, eviction_policy='evict_last')
    tmp390 = tl.load(in_ptr1 + (8320 + r1), None, eviction_policy='evict_last')
    tmp393 = tl.load(in_ptr1 + (8384 + r1), None, eviction_policy='evict_last')
    tmp396 = tl.load(in_ptr1 + (8448 + r1), None, eviction_policy='evict_last')
    tmp399 = tl.load(in_ptr1 + (8512 + r1), None, eviction_policy='evict_last')
    tmp402 = tl.load(in_ptr1 + (8576 + r1), None, eviction_policy='evict_last')
    tmp405 = tl.load(in_ptr1 + (8640 + r1), None, eviction_policy='evict_last')
    tmp408 = tl.load(in_ptr1 + (8704 + r1), None, eviction_policy='evict_last')
    tmp411 = tl.load(in_ptr1 + (8768 + r1), None, eviction_policy='evict_last')
    tmp414 = tl.load(in_ptr1 + (8832 + r1), None, eviction_policy='evict_last')
    tmp417 = tl.load(in_ptr1 + (8896 + r1), None, eviction_policy='evict_last')
    tmp420 = tl.load(in_ptr1 + (8960 + r1), None, eviction_policy='evict_last')
    tmp423 = tl.load(in_ptr1 + (9024 + r1), None, eviction_policy='evict_last')
    tmp426 = tl.load(in_ptr1 + (9088 + r1), None, eviction_policy='evict_last')
    tmp429 = tl.load(in_ptr1 + (9152 + r1), None, eviction_policy='evict_last')
    tmp432 = tl.load(in_ptr1 + (9216 + r1), None, eviction_policy='evict_last')
    tmp435 = tl.load(in_ptr1 + (9280 + r1), None, eviction_policy='evict_last')
    tmp438 = tl.load(in_ptr1 + (9344 + r1), None, eviction_policy='evict_last')
    tmp441 = tl.load(in_ptr1 + (9408 + r1), None, eviction_policy='evict_last')
    tmp444 = tl.load(in_ptr1 + (9472 + r1), None, eviction_policy='evict_last')
    tmp447 = tl.load(in_ptr1 + (9536 + r1), None, eviction_policy='evict_last')
    tmp450 = tl.load(in_ptr1 + (9600 + r1), None, eviction_policy='evict_last')
    tmp453 = tl.load(in_ptr1 + (9664 + r1), None, eviction_policy='evict_last')
    tmp456 = tl.load(in_ptr1 + (9728 + r1), None, eviction_policy='evict_last')
    tmp459 = tl.load(in_ptr1 + (9792 + r1), None, eviction_policy='evict_last')
    tmp462 = tl.load(in_ptr1 + (9856 + r1), None, eviction_policy='evict_last')
    tmp465 = tl.load(in_ptr1 + (9920 + r1), None, eviction_policy='evict_last')
    tmp468 = tl.load(in_ptr1 + (9984 + r1), None, eviction_policy='evict_last')
    tmp471 = tl.load(in_ptr1 + (10048 + r1), None, eviction_policy='evict_last')
    tmp474 = tl.load(in_ptr1 + (10112 + r1), None, eviction_policy='evict_last'
        )
    tmp477 = tl.load(in_ptr1 + (10176 + r1), None, eviction_policy='evict_last'
        )
    tmp480 = tl.load(in_ptr1 + (10240 + r1), None, eviction_policy='evict_last'
        )
    tmp483 = tl.load(in_ptr1 + (10304 + r1), None, eviction_policy='evict_last'
        )
    tmp486 = tl.load(in_ptr1 + (10368 + r1), None, eviction_policy='evict_last'
        )
    tmp489 = tl.load(in_ptr1 + (10432 + r1), None, eviction_policy='evict_last'
        )
    tmp492 = tl.load(in_ptr1 + (10496 + r1), None, eviction_policy='evict_last'
        )
    tmp495 = tl.load(in_ptr1 + (10560 + r1), None, eviction_policy='evict_last'
        )
    tmp498 = tl.load(in_ptr1 + (10624 + r1), None, eviction_policy='evict_last'
        )
    tmp501 = tl.load(in_ptr1 + (10688 + r1), None, eviction_policy='evict_last'
        )
    tmp504 = tl.load(in_ptr1 + (10752 + r1), None, eviction_policy='evict_last'
        )
    tmp507 = tl.load(in_ptr1 + (10816 + r1), None, eviction_policy='evict_last'
        )
    tmp510 = tl.load(in_ptr1 + (10880 + r1), None, eviction_policy='evict_last'
        )
    tmp513 = tl.load(in_ptr1 + (10944 + r1), None, eviction_policy='evict_last'
        )
    tmp516 = tl.load(in_ptr1 + (11008 + r1), None, eviction_policy='evict_last'
        )
    tmp519 = tl.load(in_ptr1 + (11072 + r1), None, eviction_policy='evict_last'
        )
    tmp522 = tl.load(in_ptr1 + (11136 + r1), None, eviction_policy='evict_last'
        )
    tmp525 = tl.load(in_ptr1 + (11200 + r1), None, eviction_policy='evict_last'
        )
    tmp528 = tl.load(in_ptr1 + (11264 + r1), None, eviction_policy='evict_last'
        )
    tmp531 = tl.load(in_ptr1 + (11328 + r1), None, eviction_policy='evict_last'
        )
    tmp534 = tl.load(in_ptr1 + (11392 + r1), None, eviction_policy='evict_last'
        )
    tmp537 = tl.load(in_ptr1 + (11456 + r1), None, eviction_policy='evict_last'
        )
    tmp540 = tl.load(in_ptr1 + (11520 + r1), None, eviction_policy='evict_last'
        )
    tmp543 = tl.load(in_ptr1 + (11584 + r1), None, eviction_policy='evict_last'
        )
    tmp546 = tl.load(in_ptr1 + (11648 + r1), None, eviction_policy='evict_last'
        )
    tmp549 = tl.load(in_ptr1 + (11712 + r1), None, eviction_policy='evict_last'
        )
    tmp552 = tl.load(in_ptr1 + (11776 + r1), None, eviction_policy='evict_last'
        )
    tmp555 = tl.load(in_ptr1 + (11840 + r1), None, eviction_policy='evict_last'
        )
    tmp558 = tl.load(in_ptr1 + (11904 + r1), None, eviction_policy='evict_last'
        )
    tmp561 = tl.load(in_ptr1 + (11968 + r1), None, eviction_policy='evict_last'
        )
    tmp564 = tl.load(in_ptr1 + (12032 + r1), None, eviction_policy='evict_last'
        )
    tmp567 = tl.load(in_ptr1 + (12096 + r1), None, eviction_policy='evict_last'
        )
    tmp570 = tl.load(in_ptr1 + (12160 + r1), None, eviction_policy='evict_last'
        )
    tmp573 = tl.load(in_ptr1 + (12224 + r1), None, eviction_policy='evict_last'
        )
    tmp576 = tl.load(in_ptr1 + (12288 + r1), None, eviction_policy='evict_last'
        )
    tmp579 = tl.load(in_ptr1 + (12352 + r1), None, eviction_policy='evict_last'
        )
    tmp582 = tl.load(in_ptr1 + (12416 + r1), None, eviction_policy='evict_last'
        )
    tmp585 = tl.load(in_ptr1 + (12480 + r1), None, eviction_policy='evict_last'
        )
    tmp588 = tl.load(in_ptr1 + (12544 + r1), None, eviction_policy='evict_last'
        )
    tmp591 = tl.load(in_ptr1 + (12608 + r1), None, eviction_policy='evict_last'
        )
    tmp594 = tl.load(in_ptr1 + (12672 + r1), None, eviction_policy='evict_last'
        )
    tmp597 = tl.load(in_ptr1 + (12736 + r1), None, eviction_policy='evict_last'
        )
    tmp600 = tl.load(in_ptr1 + (12800 + r1), None, eviction_policy='evict_last'
        )
    tmp603 = tl.load(in_ptr1 + (12864 + r1), None, eviction_policy='evict_last'
        )
    tmp606 = tl.load(in_ptr1 + (12928 + r1), None, eviction_policy='evict_last'
        )
    tmp609 = tl.load(in_ptr1 + (12992 + r1), None, eviction_policy='evict_last'
        )
    tmp612 = tl.load(in_ptr1 + (13056 + r1), None, eviction_policy='evict_last'
        )
    tmp615 = tl.load(in_ptr1 + (13120 + r1), None, eviction_policy='evict_last'
        )
    tmp618 = tl.load(in_ptr1 + (13184 + r1), None, eviction_policy='evict_last'
        )
    tmp621 = tl.load(in_ptr1 + (13248 + r1), None, eviction_policy='evict_last'
        )
    tmp624 = tl.load(in_ptr1 + (13312 + r1), None, eviction_policy='evict_last'
        )
    tmp627 = tl.load(in_ptr1 + (13376 + r1), None, eviction_policy='evict_last'
        )
    tmp630 = tl.load(in_ptr1 + (13440 + r1), None, eviction_policy='evict_last'
        )
    tmp633 = tl.load(in_ptr1 + (13504 + r1), None, eviction_policy='evict_last'
        )
    tmp636 = tl.load(in_ptr1 + (13568 + r1), None, eviction_policy='evict_last'
        )
    tmp639 = tl.load(in_ptr1 + (13632 + r1), None, eviction_policy='evict_last'
        )
    tmp642 = tl.load(in_ptr1 + (13696 + r1), None, eviction_policy='evict_last'
        )
    tmp645 = tl.load(in_ptr1 + (13760 + r1), None, eviction_policy='evict_last'
        )
    tmp648 = tl.load(in_ptr1 + (13824 + r1), None, eviction_policy='evict_last'
        )
    tmp651 = tl.load(in_ptr1 + (13888 + r1), None, eviction_policy='evict_last'
        )
    tmp654 = tl.load(in_ptr1 + (13952 + r1), None, eviction_policy='evict_last'
        )
    tmp657 = tl.load(in_ptr1 + (14016 + r1), None, eviction_policy='evict_last'
        )
    tmp660 = tl.load(in_ptr1 + (14080 + r1), None, eviction_policy='evict_last'
        )
    tmp663 = tl.load(in_ptr1 + (14144 + r1), None, eviction_policy='evict_last'
        )
    tmp666 = tl.load(in_ptr1 + (14208 + r1), None, eviction_policy='evict_last'
        )
    tmp669 = tl.load(in_ptr1 + (14272 + r1), None, eviction_policy='evict_last'
        )
    tmp672 = tl.load(in_ptr1 + (14336 + r1), None, eviction_policy='evict_last'
        )
    tmp675 = tl.load(in_ptr1 + (14400 + r1), None, eviction_policy='evict_last'
        )
    tmp678 = tl.load(in_ptr1 + (14464 + r1), None, eviction_policy='evict_last'
        )
    tmp681 = tl.load(in_ptr1 + (14528 + r1), None, eviction_policy='evict_last'
        )
    tmp684 = tl.load(in_ptr1 + (14592 + r1), None, eviction_policy='evict_last'
        )
    tmp687 = tl.load(in_ptr1 + (14656 + r1), None, eviction_policy='evict_last'
        )
    tmp690 = tl.load(in_ptr1 + (14720 + r1), None, eviction_policy='evict_last'
        )
    tmp693 = tl.load(in_ptr1 + (14784 + r1), None, eviction_policy='evict_last'
        )
    tmp696 = tl.load(in_ptr1 + (14848 + r1), None, eviction_policy='evict_last'
        )
    tmp699 = tl.load(in_ptr1 + (14912 + r1), None, eviction_policy='evict_last'
        )
    tmp702 = tl.load(in_ptr1 + (14976 + r1), None, eviction_policy='evict_last'
        )
    tmp705 = tl.load(in_ptr1 + (15040 + r1), None, eviction_policy='evict_last'
        )
    tmp708 = tl.load(in_ptr1 + (15104 + r1), None, eviction_policy='evict_last'
        )
    tmp711 = tl.load(in_ptr1 + (15168 + r1), None, eviction_policy='evict_last'
        )
    tmp714 = tl.load(in_ptr1 + (15232 + r1), None, eviction_policy='evict_last'
        )
    tmp717 = tl.load(in_ptr1 + (15296 + r1), None, eviction_policy='evict_last'
        )
    tmp720 = tl.load(in_ptr1 + (15360 + r1), None, eviction_policy='evict_last'
        )
    tmp723 = tl.load(in_ptr1 + (15424 + r1), None, eviction_policy='evict_last'
        )
    tmp726 = tl.load(in_ptr1 + (15488 + r1), None, eviction_policy='evict_last'
        )
    tmp729 = tl.load(in_ptr1 + (15552 + r1), None, eviction_policy='evict_last'
        )
    tmp732 = tl.load(in_ptr1 + (15616 + r1), None, eviction_policy='evict_last'
        )
    tmp735 = tl.load(in_ptr1 + (15680 + r1), None, eviction_policy='evict_last'
        )
    tmp738 = tl.load(in_ptr1 + (15744 + r1), None, eviction_policy='evict_last'
        )
    tmp741 = tl.load(in_ptr1 + (15808 + r1), None, eviction_policy='evict_last'
        )
    tmp744 = tl.load(in_ptr1 + (15872 + r1), None, eviction_policy='evict_last'

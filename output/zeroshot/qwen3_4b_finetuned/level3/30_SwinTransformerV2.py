import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__log_10_mul__to_copy_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.log2(tmp2)
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = libdevice.log(tmp4)
    tmp7 = tmp3 / tmp6
    tmp8 = tmp7 * tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tmp8 + tmp9
    tmp11 = -16.0
    tmp12 = tmp7 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tmp13 / tmp4
    tmp15 = libdevice.sigmoid(tmp10)
    tmp16 = tmp15 * tmp14
    tmp17 = tmp16 + tmp5
    tl.store(out_ptr0 + x0, tmp17, xmask)


@triton.jit
def triton_per_fused__log_softmax_mul_1(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 147456
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 448
    x3 = xindex // 448
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (256 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (512 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp9 = tl.load(in_ptr0 + (768 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr0 + (1024 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr0 + (1280 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr0 + (1536 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr0 + (1792 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr0 + (2048 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr0 + (2304 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp30 = tl.load(in_ptr0 + (2560 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr0 + (2816 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr0 + (3072 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp39 = tl.load(in_ptr0 + (3328 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp42 = tl.load(in_ptr0 + (3584 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr0 + (3840 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp48 = tl.load(in_ptr0 + (4096 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp51 = tl.load(in_ptr0 + (4352 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp54 = tl.load(in_ptr0 + (4608 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp57 = tl.load(in_ptr0 + (4864 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp60 = tl.load(in_ptr0 + (5120 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp63 = tl.load(in_ptr0 + (5376 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp66 = tl.load(in_ptr0 + (5632 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp69 = tl.load(in_ptr0 + (5888 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp72 = tl.load(in_ptr0 + (6144 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp75 = tl.load(in_ptr0 + (6400 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp78 = tl.load(in_ptr0 + (6656 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp81 = tl.load(in_ptr0 + (6912 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp84 = tl.load(in_ptr0 + (7168 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr0 + (7424 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp90 = tl.load(in_ptr0 + (7680 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp93 = tl.load(in_ptr0 + (7936 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp96 = tl.load(in_ptr0 + (8192 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp99 = tl.load(in_ptr0 + (8448 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp102 = tl.load(in_ptr0 + (8704 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp105 = tl.load(in_ptr0 + (8960 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp108 = tl.load(in_ptr0 + (9216 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp111 = tl.load(in_ptr0 + (9472 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp114 = tl.load(in_ptr0 + (9728 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp117 = tl.load(in_ptr0 + (9984 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp120 = tl.load(in_ptr0 + (10240 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp123 = tl.load(in_ptr0 + (10496 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp126 = tl.load(in_ptr0 + (10752 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp129 = tl.load(in_ptr0 + (11008 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp132 = tl.load(in_ptr0 + (11264 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp135 = tl.load(in_ptr0 + (11520 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp138 = tl.load(in_ptr0 + (11776 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp141 = tl.load(in_ptr0 + (12032 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp144 = tl.load(in_ptr0 + (12288 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp147 = tl.load(in_ptr0 + (12544 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp150 = tl.load(in_ptr0 + (12800 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp153 = tl.load(in_ptr0 + (13056 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp156 = tl.load(in_ptr0 + (13312 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp159 = tl.load(in_ptr0 + (13568 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp162 = tl.load(in_ptr0 + (13824 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp165 = tl.load(in_ptr0 + (14080 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp168 = tl.load(in_ptr0 + (14336 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp171 = tl.load(in_ptr0 + (14592 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp174 = tl.load(in_ptr0 + (14848 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp177 = tl.load(in_ptr0 + (15104 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp180 = tl.load(in_ptr0 + (15360 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp183 = tl.load(in_ptr0 + (15616 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp186 = tl.load(in_ptr0 + (15872 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp189 = tl.load(in_ptr0 + (16128 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp192 = tl.load(in_ptr0 + (16384 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp195 = tl.load(in_ptr0 + (16640 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp198 = tl.load(in_ptr0 + (16896 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp201 = tl.load(in_ptr0 + (17152 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp204 = tl.load(in_ptr0 + (17408 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp207 = tl.load(in_ptr0 + (17664 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp210 = tl.load(in_ptr0 + (17920 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp213 = tl.load(in_ptr0 + (18176 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp216 = tl.load(in_ptr0 + (18432 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp219 = tl.load(in_ptr0 + (18688 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp222 = tl.load(in_ptr0 + (18944 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp225 = tl.load(in_ptr0 + (19200 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp228 = tl.load(in_ptr0 + (19456 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp231 = tl.load(in_ptr0 + (19712 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp234 = tl.load(in_ptr0 + (19968 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp237 = tl.load(in_ptr0 + (20224 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp240 = tl.load(in_ptr0 + (20480 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp243 = tl.load(in_ptr0 + (20736 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp246 = tl.load(in_ptr0 + (20992 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp249 = tl.load(in_ptr0 + (21248 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp252 = tl.load(in_ptr0 + (21504 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp255 = tl.load(in_ptr0 + (21760 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp258 = tl.load(in_ptr0 + (22016 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp261 = tl.load(in_ptr0 + (22272 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp264 = tl.load(in_ptr0 + (22528 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp267 = tl.load(in_ptr0 + (22784 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp270 = tl.load(in_ptr0 + (23040 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp273 = tl.load(in_ptr0 + (23296 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp276 = tl.load(in_ptr0 + (23552 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp279 = tl.load(in_ptr0 + (23808 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp282 = tl.load(in_ptr0 + (24064 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp285 = tl.load(in_ptr0 + (24320 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp288 = tl.load(in_ptr0 + (24576 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp291 = tl.load(in_ptr0 + (24832 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp294 = tl.load(in_ptr0 + (25088 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp297 = tl.load(in_ptr0 + (25344 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp300 = tl.load(in_ptr0 + (25600 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp303 = tl.load(in_ptr0 + (25856 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp306 = tl.load(in_ptr0 + (26112 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp309 = tl.load(in_ptr0 + (26368 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp312 = tl.load(in_ptr0 + (26624 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp315 = tl.load(in_ptr0 + (26880 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp318 = tl.load(in_ptr0 + (27136 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp321 = tl.load(in_ptr0 + (27392 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp324 = tl.load(in_ptr0 + (27648 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp327 = tl.load(in_ptr0 + (27904 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp330 = tl.load(in_ptr0 + (28160 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp333 = tl.load(in_ptr0 + (28416 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp336 = tl.load(in_ptr0 + (28672 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp339 = tl.load(in_ptr0 + (28928 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp342 = tl.load(in_ptr0 + (29184 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp345 = tl.load(in_ptr0 + (29440 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp348 = tl.load(in_ptr0 + (29696 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp351 = tl.load(in_ptr0 + (29952 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp354 = tl.load(in_ptr0 + (30208 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp357 = tl.load(in_ptr0 + (30464 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp360 = tl.load(in_ptr0 + (30720 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp363 = tl.load(in_ptr0 + (30976 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp366 = tl.load(in_ptr0 + (31232 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp369 = tl.load(in_ptr0 + (31488 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp372 = tl.load(in_ptr0 + (31744 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp375 = tl.load(in_ptr0 + (31999 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp378 = tl.load(in_ptr0 + (32256 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp381 = tl.load(in_ptr0 + (32512 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp384 = tl.load(in_ptr0 + (32768 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp387 = tl.load(in_ptr0 + (33024 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp390 = tl.load(in_ptr0 + (33280 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp393 = tl.load(in_ptr0 + (33536 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp396 = tl.load(in_ptr0 + (33792 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp399 = tl.load(in_ptr0 + (34048 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp402 = tl.load(in_ptr0 + (34304 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp405 = tl.load(in_ptr0 + (34560 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp408 = tl.load(in_ptr0 + (34816 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp411 = tl.load(in_ptr0 + (35072 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp414 = tl.load(in_ptr0 + (35328 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp417 = tl.load(in_ptr0 + (35584 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp420 = tl.load(in_ptr0 + (35840 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp423 = tl.load(in_ptr0 + (36096 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp426 = tl.load(in_ptr0 + (36352 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp429 = tl.load(in_ptr0 + (36608 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp432 = tl.load(in_ptr0 + (36864 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp435 = tl.load(in_ptr0 + (37120 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp438 = tl.load(in_ptr0 + (37376 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp441 = tl.load(in_ptr0 + (37632 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp444 = tl.load(in_ptr0 + (37888 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp447 = tl.load(in_ptr0 + (38144 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp450 = tl.load(in_ptr0 + (38400 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp453 = tl.load(in_ptr0 + (38656 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp456 = tl.load(in_ptr0 + (38912 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp459 = tl.load(in_ptr0 + (39168 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp462 = tl.load(in_ptr0 + (39424 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp465 = tl.load(in_ptr0 + (39680 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp468 = tl.load(in_ptr0 + (39936 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp471 = tl.load(in_ptr0 + (40192 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp474 = tl.load(in_ptr0 + (40448 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp477 = tl.load(in_ptr0 + (40704 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp480 = tl.load(in_ptr0 + (40960 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp483 = tl.load(in_ptr0 + (41216 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp486 = tl.load(in_ptr0 + (41472 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp489 = tl.load(in_ptr0 + (41728 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp492 = tl.load(in_ptr0 + (41984 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp495 = tl.load(in_ptr0 + (42240 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp498 = tl.load(in_ptr0 + (42496 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp501 = tl.load(in_ptr0 + (42752 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp504 = tl.load(in_ptr0 + (43008 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp507 = tl.load(in_ptr0 + (43264 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp510 = tl.load(in_ptr0 + (43520 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp513 = tl.load(in_ptr0 + (43776 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp516 = tl.load(in_ptr0 + (44032 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp519 = tl.load(in_ptr0 + (44288 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp522 = tl.load(in_ptr0 + (44544 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp525 = tl.load(in_ptr0 + (44800 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp528 = tl.load(in_ptr0 + (45056 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp531 = tl.load(in_ptr0 + (45312 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp534 = tl.load(in_ptr0 + (45568 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp537 = tl.load(in_ptr0 + (45824 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp540 = tl.load(in_ptr0 + (46080 + r1 + 256 * x0), rmask & xmask, other=0.0)
    tmp543 = tl.load(in_ptr0 + (46
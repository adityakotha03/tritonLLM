import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool1d_native_backward_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 25600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16 % 512
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (64 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (80 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (96 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (112 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp16 = tl.load(in_ptr0 + (128 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp18 = tl.load(in_ptr0 + (144 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp20 = tl.load(in_ptr0 + (160 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp22 = tl.load(in_ptr0 + (176 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp24 = tl.load(in_ptr0 + (192 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp26 = tl.load(in_ptr0 + (208 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp28 = tl.load(in_ptr0 + (224 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp30 = tl.load(in_ptr0 + (240 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp32 = tl.load(in_ptr0 + (256 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp34 = tl.load(in_ptr0 + (272 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp36 = tl.load(in_ptr0 + (288 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp38 = tl.load(in_ptr0 + (304 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp40 = tl.load(in_ptr0 + (320 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp42 = tl.load(in_ptr0 + (336 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp44 = tl.load(in_ptr0 + (352 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp46 = tl.load(in_ptr0 + (368 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp48 = tl.load(in_ptr0 + (384 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp50 = tl.load(in_ptr0 + (400 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp52 = tl.load(in_ptr0 + (416 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp54 = tl.load(in_ptr0 + (432 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp56 = tl.load(in_ptr0 + (448 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp58 = tl.load(in_ptr0 + (464 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp60 = tl.load(in_ptr0 + (480 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp62 = tl.load(in_ptr0 + (496 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp64 = tl.load(in_ptr0 + (512 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp66 = tl.load(in_ptr0 + (64 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp68 = tl.load(in_ptr0 + (80 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (96 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (112 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (128 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (144 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (160 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (176 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (192 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (208 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (224 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (240 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (256 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (272 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (288 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (304 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (320 + x1 + 512 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (336 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp102 = tl.load(in_ptr0 + (352 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp104 = tl.load(in_ptr0 + (368 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp106 = tl.load(in_ptr0 + (384 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp108 = tl.load(in_ptr0 + (400 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp110 = tl.load(in_ptr0 + (416 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp112 = tl.load(in_ptr0 + (432 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp114 = tl.load(in_ptr0 + (448 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp116 = tl.load(in_ptr0 + (464 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp118 = tl.load(in_ptr0 + (480 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp120 = tl.load(in_ptr0 + (496 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp122 = tl.load(in_ptr0 + (512 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp124 = tl.load(in_ptr0 + (544 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp126 = tl.load(in_ptr0 + (640 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp128 = tl.load(in_ptr0 + (704 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp130 = tl.load(in_ptr0 + (768 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp132 = tl.load(in_ptr0 + (832 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp134 = tl.load(in_ptr0 + (896 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp136 = tl.load(in_ptr0 + (960 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp138 = tl.load(in_ptr0 + (1024 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp140 = tl.load(in_ptr0 + (1088 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp142 = tl.load(in_ptr0 + (1152 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp144 = tl.load(in_ptr0 + (1216 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp146 = tl.load(in_ptr0 + (1280 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp148 = tl.load(in_ptr0 + (1344 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp150 = tl.load(in_ptr0 + (1408 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp152 = tl.load(in_ptr0 + (1472 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp154 = tl.load(in_ptr0 + (1536 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp156 = tl.load(in_ptr0 + (1600 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp158 = tl.load(in_ptr0 + (1664 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp160 = tl.load(in_ptr0 + (1728 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp162 = tl.load(in_ptr0 + (1792 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp164 = tl.load(in_ptr0 + (1856 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp166 = tl.load(in_ptr0 + (1920 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp168 = tl.load(in_ptr0 + (1984 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp170 = tl.load(in_ptr0 + (2048 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp172 = tl.load(in_ptr0 + (2112 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp174 = tl.load(in_ptr0 + (2176 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp176 = tl.load(in_ptr0 + (2240 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp178 = tl.load(in_ptr0 + (2304 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp180 = tl.load(in_ptr0 + (2368 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp182 = tl.load(in_ptr0 + (2432 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp184 = tl.load(in_ptr0 + (2496 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp186 = tl.load(in_ptr0 + (2560 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp188 = tl.load(in_ptr0 + (2624 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp190 = tl.load(in_ptr0 + (2688 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp192 = tl.load(in_ptr0 + (2752 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp194 = tl.load(in_ptr0 + (2816 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp196 = tl.load(in_ptr0 + (2880 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp198 = tl.load(in_ptr0 + (2944 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp200 = tl.load(in_ptr0 + (3008 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp202 = tl.load(in_ptr0 + (3072 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp204 = tl.load(in_ptr0 + (3136 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp206 = tl.load(in_ptr0 + (3200 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp208 = tl.load(in_ptr0 + (3264 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp210 = tl.load(in_ptr0 + (3328 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp212 = tl.load(in_ptr0 + (3392 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp214 = tl.load(in_ptr0 + (3456 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp216 = tl.load(in_ptr0 + (3520 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp218 = tl.load(in_ptr0 + (3584 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp220 = tl.load(in_ptr0 + (3648 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp222 = tl.load(in_ptr0 + (3712 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp224 = tl.load(in_ptr0 + (3776 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp226 = tl.load(in_ptr0 + (3840 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp228 = tl.load(in_ptr0 + (3904 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp230 = tl.load(in_ptr0 + (3968 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp232 = tl.load(in_ptr0 + (4032 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp234 = tl.load(in_ptr0 + (4096 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp236 = tl.load(in_ptr0 + (4160 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp238 = tl.load(in_ptr0 + (4224 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp240 = tl.load(in_ptr0 + (4288 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp242 = tl.load(in_ptr0 + (4352 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp244 = tl.load(in_ptr0 + (4416 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp246 = tl.load(in_ptr0 + (4480 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp248 = tl.load(in_ptr0 + (4544 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp250 = tl.load(in_ptr0 + (4608 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp252 = tl.load(in_ptr0 + (4672 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp254 = tl.load(in_ptr0 + (4736 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp256 = tl.load(in_ptr0 + (4800 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp258 = tl.load(in_ptr0 + (4864 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp260 = tl.load(in_ptr0 + (4928 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp262 = tl.load(in_ptr0 + (4992 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp264 = tl.load(in_ptr0 + (5056 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp266 = tl.load(in_ptr0 + (5120 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp268 = tl.load(in_ptr0 + (5184 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp270 = tl.load(in_ptr0 + (5248 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp272 = tl.load(in_ptr0 + (5312 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp274 = tl.load(in_ptr0 + (5376 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp276 = tl.load(in_ptr0 + (5440 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp278 = tl.load(in_ptr0 + (5504 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp280 = tl.load(in_ptr0 + (5568 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp282 = tl.load(in_ptr0 + (5632 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp284 = tl.load(in_ptr0 + (5696 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp286 = tl.load(in_ptr0 + (5760 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp288 = tl.load(in_ptr0 + (5824 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp290 = tl.load(in_ptr0 + (5888 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp292 = tl.load(in_ptr0 + (5952 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp294 = tl.load(in_ptr0 + (6016 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp296 = tl.load(in_ptr0 + (6080 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp298 = tl.load(in_ptr0 + (6144 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp300 = tl.load(in_ptr0 + (6208 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp302 = tl.load(in_ptr0 + (6272 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp304 = tl.load(in_ptr0 + (6336 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp306 = tl.load(in_ptr0 + (6400 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp308 = tl.load(in_ptr0 + (6464 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp310 = tl.load(in_ptr0 + (6528 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp312 = tl.load(in_ptr0 + (6592 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp314 = tl.load(in_ptr0 + (6656 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp316 = tl.load(in_ptr0 + (6720 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp318 = tl.load(in_ptr0 + (6784 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp320 = tl.load(in_ptr0 + (6848 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp322 = tl.load(in_ptr0 + (6912 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp324 = tl.load(in_ptr0 + (6976 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp326 = tl.load(in_ptr0 + (7040 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp328 = tl.load(in_ptr0 + (7104 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp330 = tl.load(in_ptr0 + (7168 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp332 = tl.load(in_ptr0 + (7232 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp334 = tl.load(in_ptr0 + (7296 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp336 = tl.load(in_ptr0 + (7360 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp338 = tl.load(in_ptr0 + (7424 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp340 = tl.load(in_ptr0 + (7488 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp342 = tl.load(in_ptr0 + (7552 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp344 = tl.load(in_ptr0 + (7616 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp346 = tl.load(in_ptr0 + (7680 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp348 = tl.load(in_ptr0 + (7744 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp350 = tl.load(in_ptr0 + (7808 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp352 = tl.load(in_ptr0 + (7872 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp354 = tl.load(in_ptr0 + (7936 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp356 = tl.load(in_ptr0 + (8000 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp358 = tl.load(in_ptr0 + (8064 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp360 = tl.load(in_ptr0 + (8128 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp362 = tl.load(in_ptr0 + (8192 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp364 = tl.load(in_ptr0 + (8256 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp366 = tl.load(in_ptr0 + (8320 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp368 = tl.load(in_ptr0 + (8384 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp370 = tl.load(in_ptr0 + (8448 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp372 = tl.load(in_ptr0 + (8512 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp374 = tl.load(in_ptr0 + (8576 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp376 = tl.load(in_ptr0 + (8640 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp378 = tl.load(in_ptr0 + (8704 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp380 = tl.load(in_ptr0 + (8768 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp382 = tl.load(in_ptr0 + (8832 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp384 = tl.load(in_ptr0 + (8896 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp386 = tl.load(in_ptr0 + (8960 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp388 = tl.load(in_ptr0 + (9024 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp390 = tl.load(in_ptr0 + (9088 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp392 = tl.load(in_ptr0 + (9152 + x1 + 512 * x0), xmask, eviction_policy
        ='evict_last')
    tmp394 = tl
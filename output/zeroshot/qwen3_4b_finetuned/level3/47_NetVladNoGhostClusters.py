import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (513 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (1024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (1536 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (2048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (2560 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (3072 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (3584 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (4096 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (4608 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (5120 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (5632 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (6144 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (6656 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (7168 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (7680 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (8192 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (8704 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (9216 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (9728 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (10240 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (10752 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (11264 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (11776 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (12288 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (12800 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (13312 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (13824 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (14336 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (14848 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (15360 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (15872 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (16384 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (16896 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (17408 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (17920 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (18432 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (18944 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (19456 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (19968 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (20480 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (20992 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (21504 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (22016 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (22528 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (23040 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (23552 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (24064 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (24576 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (25088 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (25600 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (26112 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (26624 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (27136 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (27648 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (28160 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (28672 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (29184 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (29696 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (30208 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (30720 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (31232 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (31744 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (32256 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (32768 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (33280 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (33792 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (34304 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (34816 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (35328 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (35840 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (36352 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (36864 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (37376 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (37888 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (38400 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (38912 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (39424 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (39936 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (40448 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (40960 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (41472 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (41984 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (42496 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (43008 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (43520 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (44032 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (44544 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (45056 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (45568 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (46080 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (46592 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (47104 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (47616 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (48128 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (48640 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (49152 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (49664 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (50176 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (50688 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (51200 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (51712 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (52224 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (52736 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (53248 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (53760 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (54272 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (54784 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (55296 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (55808 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (56320 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (56832 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (57344 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (57856 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (58368 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (58880 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (59392 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (59904 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (60416 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (60928 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (61440 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (61952 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (62464 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (62976 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (63488 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (63999 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (64512 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (65024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (65536 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (66048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (66560 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (67072 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (67584 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (68096 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (68608 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (69120 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (69632 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (70144 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (70656 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (71168 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (71680 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (72192 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (72704 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (73216 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (73728 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (74240 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (74752 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (75264 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (75776 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (76288 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (76800 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (77312 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (77824 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (78336 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (78848 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (79360 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (79872 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (80384 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (80896 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (81408 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (81920 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (82432 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (82944 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (83456 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (83968 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (84480 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (84992 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (85504 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (86016 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (86528 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (87040 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (87552 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (88064 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (88576 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (89088 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (89600 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (90112 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (90624 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (91136 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (91648 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (92160 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (92672 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp365 = tl.load(in_ptr0 + (93184 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp367 = tl.load(in_ptr0 + (93696 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (94208 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (94720 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp373 = tl.load(in_ptr0 + (95232 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (95744 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp377 = tl.load(in_ptr0 + (96256 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp379 = tl.load(in_ptr0 + (96768 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (97280 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (97792 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp385 = tl.load(in_ptr0 + (98304 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (98816 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp389 = tl.load(in_ptr0 + (99328 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp391 = tl.load(in_ptr0 + (99840 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (100352 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp395 = tl.load(in_ptr0 + (100864 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp397 = tl.load(in_ptr0 + (101376 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp399 = tl.load(in_ptr0 + (101888 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp401 = tl.load(in_ptr0 + (102400 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp403 = tl.load(in_ptr0 + (102912 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp405 = tl.load(in_ptr0 + (103424 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp407 = tl.load(in_ptr0 + (103936 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp409 = tl.load(in_ptr0 + (104448 + 512 * x1), xmask, eviction_policy
        ='evict_last')
    tmp411 = tl.load(in_ptr0 +
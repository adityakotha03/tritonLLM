import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
import torch as th
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 66560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (32 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (48 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (64 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (80 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (96 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (112 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (128 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (144 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (160 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (176 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (192 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (208 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (224 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (240 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (256 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (272 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (288 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (304 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (320 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (336 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (352 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (368 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (384 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (400 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (416 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (432 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (448 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (464 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (480 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (496 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (512 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (528 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (544 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (560 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (576 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (592 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (608 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (624 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (640 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (656 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (672 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (688 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (704 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (720 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (736 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (752 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (768 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (784 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (800 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (816 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (832 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (848 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (864 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (880 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (896 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (912 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (928 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (944 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (960 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (976 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (992 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (1008 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (1024 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (1040 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (1056 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (1072 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (1088 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (1104 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (1120 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (1136 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (1152 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (1168 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (1184 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (1200 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (1216 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (1232 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (1248 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (1264 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (1280 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (1296 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (1312 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (1328 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (1344 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (1360 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (1376 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (1392 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (1408 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (1424 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (1440 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (1456 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (1472 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (1488 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (1504 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (1520 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (1536 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (1552 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (1568 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (1584 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (1600 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (1616 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (1632 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (1648 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (1664 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (1680 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (1696 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (1712 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (1728 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (1744 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (1760 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (1776 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (1792 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (1808 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (1824 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (1840 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (1856 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (1872 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (1888 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (1904 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (1920 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (1936 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (1952 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (1968 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (1984 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (2000 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (2016 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (2032 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (2048 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (2064 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (2080 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (2096 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (2112 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (2128 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (2144 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (2160 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (2176 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (2192 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (2208 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (2224 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (2240 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (2256 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (2272 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (2288 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (2304 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (2320 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (2336 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (2352 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (2368 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (2384 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (2400 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (2416 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (2432 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (2448 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (2464 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (2480 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (2496 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (2512 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (2528 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (2544 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (2560 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (2576 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (2592 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (2608 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (2624 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (2640 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (2656 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (2672 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (2688 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (2704 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (2720 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (2736 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (2752 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (2768 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (2784 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (2800 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (2816 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (2832 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (2848 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (2864 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (2880 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (2896 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (2912 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (2928 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (2944 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (2960 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (2976 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (2992 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (3008 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (3024 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (3040 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (3056 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (3072 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (3088 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (3104 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (3120 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (3136 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (3152 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (3168 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (3184 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (3200 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (3216 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (3232 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2
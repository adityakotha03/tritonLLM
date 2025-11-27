import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tmp0 + tmp2
    tmp6 = tmp5 + tmp4
    tl.store(out_ptr0 + x0, tmp6, xmask)


@triton.jit
def triton_poi_fused_avg_pool3d_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (32 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (48 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp7 = tl.load(in_ptr0 + (64 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp9 = tl.load(in_ptr0 + (80 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp11 = tl.load(in_ptr0 + (96 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp13 = tl.load(in_ptr0 + (112 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp15 = tl.load(in_ptr0 + (128 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp17 = tl.load(in_ptr0 + (144 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp19 = tl.load(in_ptr0 + (160 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp21 = tl.load(in_ptr0 + (176 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp23 = tl.load(in_ptr0 + (192 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp25 = tl.load(in_ptr0 + (208 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp27 = tl.load(in_ptr0 + (224 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp29 = tl.load(in_ptr0 + (240 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp31 = tl.load(in_ptr0 + (256 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp33 = tl.load(in_ptr0 + (272 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp35 = tl.load(in_ptr0 + (288 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp37 = tl.load(in_ptr0 + (304 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp39 = tl.load(in_ptr0 + (320 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp41 = tl.load(in_ptr0 + (336 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp43 = tl.load(in_ptr0 + (352 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp45 = tl.load(in_ptr0 + (368 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp47 = tl.load(in_ptr0 + (384 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp49 = tl.load(in_ptr0 + (400 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp51 = tl.load(in_ptr0 + (416 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp53 = tl.load(in_ptr0 + (432 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp55 = tl.load(in_ptr0 + (448 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp57 = tl.load(in_ptr0 + (464 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp59 = tl.load(in_ptr0 + (480 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp61 = tl.load(in_ptr0 + (496 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp63 = tl.load(in_ptr0 + (512 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp65 = tl.load(in_ptr0 + (528 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp67 = tl.load(in_ptr0 + (544 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp69 = tl.load(in_ptr0 + (560 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp71 = tl.load(in_ptr0 + (576 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp73 = tl.load(in_ptr0 + (592 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp75 = tl.load(in_ptr0 + (608 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp77 = tl.load(in_ptr0 + (624 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp79 = tl.load(in_ptr0 + (640 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp81 = tl.load(in_ptr0 + (656 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp83 = tl.load(in_ptr0 + (672 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp85 = tl.load(in_ptr0 + (688 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp87 = tl.load(in_ptr0 + (704 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp89 = tl.load(in_ptr0 + (720 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp91 = tl.load(in_ptr0 + (736 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp93 = tl.load(in_ptr0 + (752 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp95 = tl.load(in_ptr0 + (768 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp97 = tl.load(in_ptr0 + (784 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp99 = tl.load(in_ptr0 + (800 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp101 = tl.load(in_ptr0 + (816 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp103 = tl.load(in_ptr0 + (832 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp105 = tl.load(in_ptr0 + (848 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp107 = tl.load(in_ptr0 + (864 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp109 = tl.load(in_ptr0 + (880 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp111 = tl.load(in_ptr0 + (896 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp113 = tl.load(in_ptr0 + (912 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp115 = tl.load(in_ptr0 + (928 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp117 = tl.load(in_ptr0 + (944 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp119 = tl.load(in_ptr0 + (960 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp121 = tl.load(in_ptr0 + (976 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp123 = tl.load(in_ptr0 + (992 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp125 = tl.load(in_ptr0 + (1008 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp127 = tl.load(in_ptr0 + (1024 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp129 = tl.load(in_ptr0 + (1040 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp131 = tl.load(in_ptr0 + (1056 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp133 = tl.load(in_ptr0 + (1072 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp135 = tl.load(in_ptr0 + (1088 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp137 = tl.load(in_ptr0 + (1104 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp139 = tl.load(in_ptr0 + (1120 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp141 = tl.load(in_ptr0 + (1136 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp143 = tl.load(in_ptr0 + (1152 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp145 = tl.load(in_ptr0 + (1168 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp147 = tl.load(in_ptr0 + (1184 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp149 = tl.load(in_ptr0 + (1200 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp151 = tl.load(in_ptr0 + (1216 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp153 = tl.load(in_ptr0 + (1232 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp155 = tl.load(in_ptr0 + (1248 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp157 = tl.load(in_ptr0 + (1264 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp159 = tl.load(in_ptr0 + (1280 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp161 = tl.load(in_ptr0 + (1296 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp163 = tl.load(in_ptr0 + (1312 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp165 = tl.load(in_ptr0 + (1328 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp167 = tl.load(in_ptr0 + (1344 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp169 = tl.load(in_ptr0 + (1360 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp171 = tl.load(in_ptr0 + (1376 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp173 = tl.load(in_ptr0 + (1392 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp175 = tl.load(in_ptr0 + (1408 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp177 = tl.load(in_ptr0 + (1424 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp179 = tl.load(in_ptr0 + (1440 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp181 = tl.load(in_ptr0 + (1456 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp183 = tl.load(in_ptr0 + (1472 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp185 = tl.load(in_ptr0 + (1488 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp187 = tl.load(in_ptr0 + (1504 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp189 = tl.load(in_ptr0 + (1520 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp191 = tl.load(in_ptr0 + (1536 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp193 = tl.load(in_ptr0 + (1552 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp195 = tl.load(in_ptr0 + (1568 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp197 = tl.load(in_ptr0 + (1584 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp199 = tl.load(in_ptr0 + (1600 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp201 = tl.load(in_ptr0 + (1616 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp203 = tl.load(in_ptr0 + (1632 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp205 = tl.load(in_ptr0 + (1648 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp207 = tl.load(in_ptr0 + (1664 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp209 = tl.load(in_ptr0 + (1680 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp211 = tl.load(in_ptr0 + (1696 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp213 = tl.load(in_ptr0 + (1712 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp215 = tl.load(in_ptr0 + (1728 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp217 = tl.load(in_ptr0 + (1744 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp219 = tl.load(in_ptr0 + (1760 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp221 = tl.load(in_ptr0 + (1776 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp223 = tl.load(in_ptr0 + (1792 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp225 = tl.load(in_ptr0 + (1808 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp227 = tl.load(in_ptr0 + (1824 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp229 = tl.load(in_ptr0 + (1840 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp231 = tl.load(in_ptr0 + (1856 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp233 = tl.load(in_ptr0 + (1872 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp235 = tl.load(in_ptr0 + (1888 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp237 = tl.load(in_ptr0 + (1904 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp239 = tl.load(in_ptr0 + (1920 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp241 = tl.load(in_ptr0 + (1936 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp243 = tl.load(in_ptr0 + (1952 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp245 = tl.load(in_ptr0 + (1968 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp247 = tl.load(in_ptr0 + (1984 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp249 = tl.load(in_ptr0 + (2000 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp251 = tl.load(in_ptr0 + (2016 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp253 = tl.load(in_ptr0 + (2032 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp255 = tl.load(in_ptr0 + (2048 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp257 = tl.load(in_ptr0 + (2064 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp259 = tl.load(in_ptr0 + (2080 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp261 = tl.load(in_ptr0 + (2096 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp263 = tl.load(in_ptr0 + (2112 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp265 = tl.load(in_ptr0 + (2128 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp267 = tl.load(in_ptr0 + (2144 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp269 = tl.load(in_ptr0 + (2160 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp271 = tl.load(in_ptr0 + (2176 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp273 = tl.load(in_ptr0 + (2192 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp275 = tl.load(in_ptr0 + (2208 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp277 = tl.load(in_ptr0 + (2224 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp279 = tl.load(in_ptr0 + (2240 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp281 = tl.load(in_ptr0 + (2256 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp283 = tl.load(in_ptr0 + (2272 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp285 = tl.load(in_ptr0 + (2288 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp287 = tl.load(in_ptr0 + (2304 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp289 = tl.load(in_ptr0 + (2320 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp291 = tl.load(in_ptr0 + (2336 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp293 = tl.load(in_ptr0 + (2352 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp295 = tl.load(in_ptr0 + (2368 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp297 = tl.load(in_ptr0 + (2384 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp299 = tl.load(in_ptr0 + (2400 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp301 = tl.load(in_ptr0 + (2416 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp303 = tl.load(in_ptr0 + (2432 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp305 = tl.load(in_ptr0 + (2448 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp307 = tl.load(in_ptr0 + (2464 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp309 = tl.load(in_ptr0 + (2480 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp311 = tl.load(in_ptr0 + (2496 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp313 = tl.load(in_ptr0 + (2512 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp315 = tl.load(in_ptr0 + (2528 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp317 = tl.load(in_ptr0 + (2544 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp319 = tl.load(in_ptr0 + (2560 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp321 = tl.load(in_ptr0 + (2576 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp323 = tl.load(in_ptr0 + (2592 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp325 = tl.load(in_ptr0 + (2608 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp327 = tl.load(in_ptr0 + (2624 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp329 = tl.load(in_ptr0 + (2640 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp331 = tl.load(in_ptr0 + (2656 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp333 = tl.load(in_ptr0 + (2672 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp335 = tl.load(in_ptr0 + (2688 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp337 = tl.load(in_ptr0 + (2704 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp339 = tl.load(in_ptr0 + (2720 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp341 = tl.load(in_ptr0 + (2736 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp343 = tl.load(in_ptr0 + (2752 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp345 = tl.load(in_ptr0 + (2768 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp347 = tl.load(in_ptr0 + (2784 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp349 = tl.load(in_ptr0 + (2800 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp351 = tl.load(in_ptr0 + (2816 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp353 = tl.load(in_ptr0 + (2832 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp355 = tl.load(in_ptr0 + (2848 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp357 = tl.load(in_ptr0 + (2864 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp359 = tl.load(in_ptr0 + (2880 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp361 = tl.load(in_ptr0 + (2896 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp363 = tl.load(in_ptr0 + (2912 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp365 = tl.load(in_ptr0 + (2928 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp367 = tl.load(in_ptr0 + (2944 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp369 = tl.load(in_ptr0 + (2960 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp371 = tl.load(in_ptr0 + (2976 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp373 = tl.load(in_ptr0 + (2992 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp375 = tl.load(in_ptr0 + (3008 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp377 = tl.load(in_ptr0 + (3024 + x0 + 16 * x1 + 256 * x2), xmask)
    tmp379 = tl.load(in_ptr0 + (3040 + x0 + 16 * x1 + 256 * x
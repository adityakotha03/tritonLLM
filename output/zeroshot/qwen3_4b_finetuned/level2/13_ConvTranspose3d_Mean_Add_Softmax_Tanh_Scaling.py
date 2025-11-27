import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (256 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (320 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (384 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (448 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (512 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (576 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (640 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (704 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (768 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (832 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (896 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (960 + x0 + 32 * x1), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (1024 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp33 = tl.load(in_ptr0 + (1088 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp35 = tl.load(in_ptr0 + (1152 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp37 = tl.load(in_ptr0 + (1216 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp39 = tl.load(in_ptr0 + (1280 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp41 = tl.load(in_ptr0 + (1344 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp43 = tl.load(in_ptr0 + (1408 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp45 = tl.load(in_ptr0 + (1472 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp47 = tl.load(in_ptr0 + (1536 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp49 = tl.load(in_ptr0 + (1600 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp51 = tl.load(in_ptr0 + (1664 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp53 = tl.load(in_ptr0 + (1728 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp55 = tl.load(in_ptr0 + (1792 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp57 = tl.load(in_ptr0 + (1856 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp59 = tl.load(in_ptr0 + (1920 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp61 = tl.load(in_ptr0 + (1984 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp63 = tl.load(in_ptr0 + (2048 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp65 = tl.load(in_ptr0 + (2112 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp67 = tl.load(in_ptr0 + (2176 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp69 = tl.load(in_ptr0 + (2240 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp71 = tl.load(in_ptr0 + (2304 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp73 = tl.load(in_ptr0 + (2368 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp75 = tl.load(in_ptr0 + (2432 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp77 = tl.load(in_ptr0 + (2496 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp79 = tl.load(in_ptr0 + (2560 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp81 = tl.load(in_ptr0 + (2624 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp83 = tl.load(in_ptr0 + (2688 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp85 = tl.load(in_ptr0 + (2752 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp87 = tl.load(in_ptr0 + (2816 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp89 = tl.load(in_ptr0 + (2880 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp91 = tl.load(in_ptr0 + (2944 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp93 = tl.load(in_ptr0 + (3008 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp95 = tl.load(in_ptr0 + (3072 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp97 = tl.load(in_ptr0 + (3136 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp99 = tl.load(in_ptr0 + (3200 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp101 = tl.load(in_ptr0 + (3264 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp103 = tl.load(in_ptr0 + (3328 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp105 = tl.load(in_ptr0 + (3392 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp107 = tl.load(in_ptr0 + (3456 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp109 = tl.load(in_ptr0 + (3520 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp111 = tl.load(in_ptr0 + (3584 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp113 = tl.load(in_ptr0 + (3648 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp115 = tl.load(in_ptr0 + (3712 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp117 = tl.load(in_ptr0 + (3776 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp119 = tl.load(in_ptr0 + (3840 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp121 = tl.load(in_ptr0 + (3904 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp123 = tl.load(in_ptr0 + (3968 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp125 = tl.load(in_ptr0 + (4032 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp127 = tl.load(in_ptr0 + (4096 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp129 = tl.load(in_ptr0 + (4160 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp131 = tl.load(in_ptr0 + (4224 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp133 = tl.load(in_ptr0 + (4288 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp135 = tl.load(in_ptr0 + (4352 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp137 = tl.load(in_ptr0 + (4416 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp139 = tl.load(in_ptr0 + (4480 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp141 = tl.load(in_ptr0 + (4544 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp143 = tl.load(in_ptr0 + (4608 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp145 = tl.load(in_ptr0 + (4672 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp147 = tl.load(in_ptr0 + (4736 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp149 = tl.load(in_ptr0 + (4800 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp151 = tl.load(in_ptr0 + (4864 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp153 = tl.load(in_ptr0 + (4928 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp155 = tl.load(in_ptr0 + (4992 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp157 = tl.load(in_ptr0 + (5056 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp159 = tl.load(in_ptr0 + (5120 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp161 = tl.load(in_ptr0 + (5184 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp163 = tl.load(in_ptr0 + (5248 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp165 = tl.load(in_ptr0 + (5312 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp167 = tl.load(in_ptr0 + (5376 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp169 = tl.load(in_ptr0 + (5440 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp171 = tl.load(in_ptr0 + (5504 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp173 = tl.load(in_ptr0 + (5568 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp175 = tl.load(in_ptr0 + (5632 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp177 = tl.load(in_ptr0 + (5696 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp179 = tl.load(in_ptr0 + (5760 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp181 = tl.load(in_ptr0 + (5824 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp183 = tl.load(in_ptr0 + (5888 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp185 = tl.load(in_ptr0 + (5952 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp187 = tl.load(in_ptr0 + (6016 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp189 = tl.load(in_ptr0 + (6080 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp191 = tl.load(in_ptr0 + (6144 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp193 = tl.load(in_ptr0 + (6208 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp195 = tl.load(in_ptr0 + (6272 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp197 = tl.load(in_ptr0 + (6336 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp199 = tl.load(in_ptr0 + (6400 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp201 = tl.load(in_ptr0 + (6464 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp203 = tl.load(in_ptr0 + (6528 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp205 = tl.load(in_ptr0 + (6592 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp207 = tl.load(in_ptr0 + (6656 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp209 = tl.load(in_ptr0 + (6720 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp211 = tl.load(in_ptr0 + (6784 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp213 = tl.load(in_ptr0 + (6848 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp215 = tl.load(in_ptr0 + (6912 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp217 = tl.load(in_ptr0 + (6976 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp219 = tl.load(in_ptr0 + (7040 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp221 = tl.load(in_ptr0 + (7104 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp223 = tl.load(in_ptr0 + (7168 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp225 = tl.load(in_ptr0 + (7232 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp227 = tl.load(in_ptr0 + (7296 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp229 = tl.load(in_ptr0 + (7360 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp231 = tl.load(in_ptr0 + (7424 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp233 = tl.load(in_ptr0 + (7488 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp235 = tl.load(in_ptr0 + (7552 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp237 = tl.load(in_ptr0 + (7616 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp239 = tl.load(in_ptr0 + (7680 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp241 = tl.load(in_ptr0 + (7744 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp243 = tl.load(in_ptr0 + (7808 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp245 = tl.load(in_ptr0 + (7872 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp247 = tl.load(in_ptr0 + (7936 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp249 = tl.load(in_ptr0 + (8000 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp251 = tl.load(in_ptr0 + (8064 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp253 = tl.load(in_ptr0 + (8128 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp255 = tl.load(in_ptr0 + (8192 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp257 = tl.load(in_ptr0 + (8256 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp259 = tl.load(in_ptr0 + (8320 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp261 = tl.load(in_ptr0 + (8384 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp263 = tl.load(in_ptr0 + (8448 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp265 = tl.load(in_ptr0 + (8512 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp267 = tl.load(in_ptr0 + (8576 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp269 = tl.load(in_ptr0 + (8640 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp271 = tl.load(in_ptr0 + (8704 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp273 = tl.load(in_ptr0 + (8768 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp275 = tl.load(in_ptr0 + (8832 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp277 = tl.load(in_ptr0 + (8896 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp279 = tl.load(in_ptr0 + (8960 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp281 = tl.load(in_ptr0 + (9024 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp283 = tl.load(in_ptr0 + (9088 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp285 = tl.load(in_ptr0 + (9152 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp287 = tl.load(in_ptr0 + (9216 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp289 = tl.load(in_ptr0 + (9280 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp291 = tl.load(in_ptr0 + (9344 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp293 = tl.load(in_ptr0 + (9408 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp295 = tl.load(in_ptr0 + (9472 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp297 = tl.load(in_ptr0 + (9536 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp299 = tl.load(in_ptr0 + (9600 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp301 = tl.load(in_ptr0 + (9664 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp303 = tl.load(in_ptr0 + (9728 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp305 = tl.load(in_ptr0 + (9792 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp307 = tl.load(in_ptr0 + (9856 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp309 = tl.load(in_ptr0 + (9920 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp311 = tl.load(in_ptr0 + (9984 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp313 = tl.load(in_ptr0 + (10048 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp315 = tl.load(in_ptr0 + (10112 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp317 = tl.load(in_ptr0 + (10176 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp319 = tl.load(in_ptr0 + (10240 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp321 = tl.load(in_ptr0 + (10304 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp323 = tl.load(in_ptr0 + (10368 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp325 = tl.load(in_ptr0 + (10432 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp327 = tl.load(in_ptr0 + (10496 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp329 = tl.load(in_ptr0 + (10560 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp331 = tl.load(in_ptr0 + (10624 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp333 = tl.load(in_ptr0 + (10688 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp335 = tl.load(in_ptr0 + (10752 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp337 = tl.load(in_ptr0 + (10816 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp339 = tl.load(in_ptr0 + (10880 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp341 = tl.load(in_ptr0 + (10944 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp343 = tl.load(in_ptr0 + (11008 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp345 = tl.load(in_ptr0 + (11072 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp347 = tl.load(in_ptr0 + (11136 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp349 = tl.load(in_ptr0 + (11200 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp351 = tl.load(in_ptr0 + (11264 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp353 = tl.load(in_ptr0 + (11328 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp355 = tl.load(in_ptr0 + (11392 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp357 = tl.load(in_ptr0 + (11456 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp359 = tl.load(in_ptr0 + (11520 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp361 = tl.load(in_ptr0 + (11584 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp363 = tl.load(in_ptr0 + (11648 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp365 = tl.load(in_ptr0 + (11712 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp367 = tl.load(in_ptr0 + (11776 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp369 = tl.load(in_ptr0 + (11840 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp371 = tl.load(in_ptr0 + (11904 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp373 = tl.load(in_ptr0 + (11968 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp375 = tl.load(in_ptr0 + (12032 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp377 = tl.load(in_ptr0 + (12096 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp379 = tl.load(in_ptr0 + (12160 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp381 = tl.load(in_ptr0 + (12224 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp383 = tl.load(in_ptr0 + (12288 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp385 = tl.load(in_ptr0 + (12352 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp387 = tl.load(in_ptr0 + (12416 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp389 = tl.load(in_ptr0 + (12480 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp391 = tl.load(in_ptr0 + (12544 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp393 = tl.load(in_ptr0 + (12608 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp395 = tl.load(in_ptr0 + (12672 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp397 = tl.load(in_ptr0 + (12736 + x0 + 32 * x1), xmask, eviction_policy
        ='evict_last')
    tmp399 = tl.load(in_ptr0 + (12800 + x0 + 32 * x
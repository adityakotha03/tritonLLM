import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cumsum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 800
    x4 = xindex % 800
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x4 + 8128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (x4 + 8096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (x4 + 8064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (x4 + 8032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (x4 + 7992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (x4 + 7960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (x4 + 7928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (x4 + 7896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (x4 + 7864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (x4 + 7832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (x4 + 7800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (x4 + 7768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (x4 + 7736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (x4 + 7704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (x4 + 7672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (x4 + 7640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (x4 + 7608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (x4 + 7576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (x4 + 7544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (x4 + 7512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (x4 + 7480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (x4 + 7448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (x4 + 7416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (x4 + 7384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (x4 + 7352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (x4 + 7320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (x4 + 7288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (x4 + 7256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (x4 + 7224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (x4 + 7192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (x4 + 7160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (x4 + 7128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (x4 + 7096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (x4 + 7064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (x4 + 7032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (x4 + 6992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (x4 + 6960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (x4 + 6928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (x4 + 6896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (x4 + 6864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (x4 + 6832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (x4 + 6800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (x4 + 6768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (x4 + 6736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (x4 + 6704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (x4 + 6672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (x4 + 6640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (x4 + 6608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (x4 + 6576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (x4 + 6544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (x4 + 6512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (x4 + 6480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (x4 + 6448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (x4 + 6416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (x4 + 6384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (x4 + 6352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (x4 + 6320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (x4 + 6288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (x4 + 6256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (x4 + 6224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (x4 + 6192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (x4 + 6160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (x4 + 6128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (x4 + 6096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (x4 + 6064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (x4 + 6032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (x4 + 5992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (x4 + 5960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (x4 + 5928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (x4 + 5896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (x4 + 5864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (x4 + 5832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (x4 + 5800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (x4 + 5768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (x4 + 5736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (x4 + 5704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (x4 + 5672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (x4 + 5640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (x4 + 5608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (x4 + 5576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (x4 + 5544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (x4 + 5512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (x4 + 5480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (x4 + 5448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (x4 + 5416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (x4 + 5384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (x4 + 5352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (x4 + 5320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (x4 + 5288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (x4 + 5256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (x4 + 5224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (x4 + 5192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (x4 + 5160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (x4 + 5128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (x4 + 5096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (x4 + 5064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (x4 + 5032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (x4 + 4992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (x4 + 4960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (x4 + 4928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (x4 + 4896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (x4 + 4864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (x4 + 4832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (x4 + 4800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (x4 + 4768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (x4 + 4736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (x4 + 4704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (x4 + 4672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (x4 + 4640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (x4 + 4608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (x4 + 4576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (x4 + 4544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (x4 + 4512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (x4 + 4480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (x4 + 4448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (x4 + 4416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (x4 + 4384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (x4 + 4352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (x4 + 4320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (x4 + 4288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (x4 + 4256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (x4 + 4224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (x4 + 4192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (x4 + 4160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (x4 + 4128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (x4 + 4096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (x4 + 4064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (x4 + 4032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (x4 + 3992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (x4 + 3960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (x4 + 3928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (x4 + 3896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (x4 + 3864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (x4 + 3832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (x4 + 3800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (x4 + 3768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (x4 + 3736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (x4 + 3704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (x4 + 3672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (x4 + 3640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (x4 + 3608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (x4 + 3576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (x4 + 3544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (x4 + 3512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (x4 + 3480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (x4 + 3448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (x4 + 3416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (x4 + 3384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (x4 + 3352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (x4 + 3320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (x4 + 3288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (x4 + 3256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (x4 + 3224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (x4 + 3192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (x4 + 3160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (x4 + 3128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (x4 + 3096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (x4 + 3064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (x4 + 3032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (x4 + 2992 * x3), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (x4 + 2960 * x3), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (x4 + 2928 * x3), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (x4 + 2896 * x3), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (x4 + 2864 * x3), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (x4 + 2832 * x3), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (x4 + 2800 * x3), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (x4 + 2768 * x3), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (x4 + 2736 * x3), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (x4 + 2704 * x3), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (x4 + 2672 * x3), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (x4 + 2640 * x3), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (x4 + 2608 * x3), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (x4 + 2576 * x3), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (x4 + 2544 * x3), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (x4 + 2512 * x3), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (x4 + 2480 * x3), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (x4 + 2448 * x3), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (x4 + 2416 * x3), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (x4 + 2384 * x3), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (x4 + 2352 * x3), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (x4 + 2320 * x3), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (x4 + 2288 * x3), xmask, eviction_policy=
        'evict_last')
    tmp363 = tl.load(in_ptr0 + (x4 + 2256 * x3), xmask, eviction_policy=
        'evict_last')
    tmp365 = tl.load(in_ptr0 + (x4 + 2224 * x3), xmask, eviction_policy=
        'evict_last')
    tmp367 = tl.load(in_ptr0 + (x4 + 2192 * x3), xmask, eviction_policy=
        'evict_last')
    tmp369 = tl.load(in_ptr0 + (x4 + 2160 * x3), xmask, eviction_policy=
        'evict_last')
    tmp371 = tl.load(in_ptr0 + (x4 + 2128 * x3), xmask, eviction_policy=
        'evict_last')
    tmp373 = tl.load(in_ptr0 + (x4 + 2096 * x3), xmask, eviction_policy=
        'evict_last')
    tmp375 = tl.load(in_ptr0 + (x4 + 2064 * x3), xmask, eviction_policy=
        'evict_last')
    tmp377 = tl.load(in_ptr0 + (x4 + 2032 * x3), xmask, eviction_policy=
        'evict_last')
    tmp379 = tl.load(in_ptr0 + (x4 + 2000 * x3), xmask, eviction_policy=
        'evict_last')
    tmp381 = tl.load(in_ptr0 + (x4 + 1968 * x3), xmask, eviction_policy=
        'evict_last')
    tmp383 = tl.load(in_ptr0 + (x4 + 1936 * x3), xmask, eviction_policy=
        'evict_last')
    tmp385 = tl.load(in_ptr0 + (x4 + 1904 * x3), xmask, eviction_policy=
        'evict_last')
    tmp387 = tl.load(in_ptr0 + (x4 + 1872 * x3), xmask, eviction_policy=
        'evict_last')
    tmp389 = tl.load(in_ptr0 + (x4 + 1840 * x3), xmask, eviction_policy=
        'evict_last')
    tmp391 = tl.load(in_ptr0 + (x4 + 1808 * x3), xmask, eviction_policy=
        'evict_last')
    tmp393 = tl.load(in_ptr0 + (x4 + 1776 * x3), xmask, eviction_policy=
        'evict_last')
    tmp395 = tl.load(in_ptr0 + (x4 + 1744 * x3), xmask, eviction_policy=
        'evict_last')
    tmp397 = tl.load(in_ptr0 + (x4 + 1712 * x3), xmask, eviction_policy=
        'evict_last')
    tmp399 = tl.load(in_ptr0 + (x4 + 1680 * x3), xmask, eviction_policy=
        'evict_last')
    tmp401 = tl.load(in_ptr0 + (x4 + 1648 * x3), xmask, eviction_policy=
        'evict_last')
    tmp403 = tl.load(in_ptr0 + (x4 + 1616 * x3), xmask, eviction_policy=
        'evict_last')
    tmp405 = tl.load(in_ptr0 + (x4 + 1584 * x3), xmask, eviction_policy=
        'evict_last')
    tmp407 = tl.load(in_ptr0 + (x4 + 1552 * x3), xmask, eviction_policy=
        'evict_last')
    tmp409 = tl.load(in_ptr0 + (x4 + 1520 * x3), xmask, eviction_policy=
        'evict_last')
    tmp411 = tl.load(in_ptr0 + (x4 + 1488 * x3), xmask, eviction_policy=
        'evict_last')
    tmp413 = tl.load(in_ptr0 + (x4 + 1456 * x3), xmask, eviction_policy=
        'evict_last')
    tmp415 = tl.load(in_ptr0 + (x4 + 1424 * x3), xmask, eviction_policy=
        'evict_last')
    tmp417 = tl.load(in_ptr0 + (x4 + 1392 * x3), xmask, eviction_policy=
        'evict_last')
    tmp419 = tl.load(in_ptr0 + (x4 + 1360 * x3), xmask, eviction_policy=
        'evict_last')
    tmp421 = tl.load(in_ptr0 + (x4 + 1328 * x3), xmask, eviction_policy=
        'evict_last')
    tmp423 = tl.load(in_ptr0 + (x4 + 1296 * x3), xmask, eviction_policy=
        'evict_last')
    tmp425 = tl.load(in_ptr0 + (x4 + 1264 * x3), xmask, eviction_policy=
        'evict_last')
    tmp427 = tl.load(in_ptr0 + (x4 + 1232 * x3), xmask, eviction_policy=
        'evict_last')
    tmp429 = tl.load(in_ptr0 + (x4 + 1200 * x3), xmask, eviction_policy=
        'evict_last')
    tmp431 = tl.load(in_ptr0 + (x4 + 1168 * x3), xmask, eviction_policy=
        'evict_last')
    tmp433 = tl.load(in_ptr0 + (x4 + 1136 * x
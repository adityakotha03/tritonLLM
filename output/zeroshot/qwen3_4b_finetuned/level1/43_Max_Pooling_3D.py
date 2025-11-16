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


@triton.jit
def triton_poi_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 36336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 129 % 42
    x4 = xindex // 5388
    x3 = xindex // 129 % 5388
    tmp0 = -2.0
    tmp1 = 0.0
    tmp2 = x2
    tmp3 = x1
    tmp4 = tmp3 + 1
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp5 + tmp5
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 < tmp6
    tmp9 = -1.0
    tmp10 = tl.where(tmp8, tmp4, tmp9)
    tmp11 = tmp4 < tmp7
    tmp12 = tl.where(tmp11, tmp4, tmp9)
    tmp13 = tmp2 < tmp7
    tmp14 = tl.where(tmp13, tmp2, tmp9)
    tmp15 = tmp14 < tmp6
    tmp16 = tl.where(tmp15, tmp14, tmp9)
    tmp17 = tmp12 < tmp6
    tmp18 = tl.where(tmp17, tmp12, tmp9)
    tmp19 = tmp0 < tmp5
    tmp20 = tl.where(tmp19, tmp0, tmp1)
    tmp21 = tmp19 | tmp20
    tmp22 = tl.where(tmp19, tmp18, tmp1)
    tmp23 = tmp22 < tmp5
    tmp24 = tl.where(tmp23, tmp22, tmp1)
    tmp25 = tmp24 < tmp6
    tmp26 = tl.where(tmp25, tmp24, tmp1)
    tmp27 = tmp26 < tmp7
    tmp28 = tl.where(tmp27, tmp26, tmp1)
    tmp29 = tl.full(tmp28.shape, 0, tmp28.dtype)
    tmp30 = tl.where(tmp28 != tmp29, tmp28, tmp29)
    tmp31 = tmp26 < tmp5
    tmp32 = tl.where(tmp31, tmp26, tmp1)
    tmp33 = tmp32 < tmp6
    tmp34 = tl.where(tmp33, tmp32, tmp1)
    tmp35 = tmp34 < tmp7
    tmp36 = tl.where(tmp35, tmp34, tmp1)
    tmp37 = tl.full(tmp36.shape, 0, tmp36.dtype)
    tmp38 = tl.where(tmp36 != tmp37, tmp36, tmp37)
    tmp39 = tmp36 < tmp5
    tmp40 = tl.where(tmp39, tmp36, tmp1)
    tmp41 = tmp40 < tmp6
    tmp42 = tl.where(tmp41, tmp40, tmp1)
    tmp43 = tmp42 < tmp7
    tmp44 = tl.where(tmp43, tmp42, tmp1)
    tmp45 = tl.full(tmp44.shape, 0, tmp44.dtype)
    tmp46 = tl.where(tmp44 != tmp45, tmp44, tmp45)
    tmp47 = tmp27 < tmp5
    tmp48 = tl.where(tmp47, tmp27, tmp1)
    tmp49 = tmp48 < tmp6
    tmp50 = tl.where(tmp49, tmp48, tmp1)
    tmp51 = tmp50 < tmp7
    tmp52 = tl.where(tmp51, tmp50, tmp1)
    tmp53 = tl.full(tmp52.shape, 0, tmp52.dtype)
    tmp54 = tl.where(tmp52 != tmp53, tmp52, tmp53)
    tmp55 = tmp52 < tmp5
    tmp56 = tl.where(tmp55, tmp52, tmp1)
    tmp57 = tmp56 < tmp6
    tmp58 = tl.where(tmp57, tmp56, tmp1)
    tmp59 = tmp58 < tmp7
    tmp60 = tl.where(tmp59, tmp58, tmp1)
    tmp61 = tl.full(tmp60.shape, 0, tmp60.dtype)
    tmp62 = tl.where(tmp60 != tmp61, tmp60, tmp61)
    tmp63 = tl.where(tmp60 < tmp5, tmp60, tmp1)
    tmp64 = tmp63 < tmp6
    tmp65 = tl.where(tmp64, tmp63, tmp1)
    tmp66 = tmp65 < tmp7
    tmp67 = tl.where(tmp66, tmp65, tmp1)
    tmp68 = tl.full(tmp67.shape, 0, tmp67.dtype)
    tmp69 = tl.where(tmp67 != tmp68, tmp67, tmp68)
    tmp70 = tmp67 < tmp5
    tmp71 = tl.where(tmp70, tmp67, tmp1)
    tmp72 = tmp71 < tmp6
    tmp73 = tl.where(tmp72, tmp71, tmp1)
    tmp74 = tmp73 < tmp7
    tmp75 = tl.where(tmp74, tmp73, tmp1)
    tmp76 = tl.full(tmp75.shape, 0, tmp75.dtype)
    tmp77 = tl.where(tmp75 != tmp76, tmp75, tmp76)
    tmp78 = tl.where(tmp75 < tmp5, tmp75, tmp1)
    tmp79 = tmp78 < tmp6
    tmp80 = tl.where(tmp79, tmp78, tmp1)
    tmp81 = tmp80 < tmp7
    tmp82 = tl.where(tmp81, tmp80, tmp1)
    tmp83 = tl.full(tmp82.shape, 0, tmp82.dtype)
    tmp84 = tl.where(tmp82 != tmp83, tmp82, tmp83)
    tmp85 = tl.where(tmp82 < tmp5, tmp82, tmp1)
    tmp86 = tmp85 < tmp6
    tmp87 = tl.where(tmp86, tmp85, tmp1)
    tmp88 = tmp87 < tmp7
    tmp89 = tl.where(tmp88, tmp87, tmp1)
    tmp90 = tl.full(tmp89.shape, 0, tmp89.dtype)
    tmp91 = tl.where(tmp89 != tmp90, tmp89, tmp90)
    tmp92 = tl.where(tmp89 < tmp5, tmp89, tmp1)
    tmp93 = tmp92 < tmp6
    tmp94 = tl.where(tmp93, tmp92, tmp1)
    tmp95 = tmp94 < tmp7
    tmp96 = tl.where(tmp95, tmp94, tmp1)
    tmp97 = tl.full(tmp96.shape, 0, tmp96.dtype)
    tmp98 = tl.where(tmp96 != tmp97, tmp96, tmp97)
    tmp99 = tmp96 < tmp5
    tmp100 = tl.where(tmp99, tmp96, tmp1)
    tmp101 = tmp100 < tmp6
    tmp102 = tl.where(tmp101, tmp100, tmp1)
    tmp103 = tmp102 < tmp7
    tmp104 = tl.where(tmp103, tmp102, tmp1)
    tmp105 = tl.full(tmp104.shape, 0, tmp104.dtype)
    tmp106 = tl.where(tmp104 != tmp105, tmp104, tmp105)
    tmp107 = tl.where(tmp104 < tmp5, tmp104, tmp1)
    tmp108 = tmp107 < tmp6
    tmp109 = tl.where(tmp108, tmp107, tmp1)
    tmp110 = tmp109 < tmp7
    tmp111 = tl.where(tmp110, tmp109, tmp1)
    tmp112 = tl.full(tmp111.shape, 0, tmp111.dtype)
    tmp113 = tl.where(tmp111 != tmp112, tmp111, tmp112)
    tmp114 = tl.where(tmp111 < tmp5, tmp111, tmp1)
    tmp115 = tmp114 < tmp6
    tmp116 = tl.where(tmp115, tmp114, tmp1)
    tmp117 = tmp116 < tmp7
    tmp118 = tl.where(tmp117, tmp116, tmp1)
    tmp119 = tl.full(tmp118.shape, 0, tmp118.dtype)
    tmp120 = tl.where(tmp118 != tmp119, tmp118, tmp119)
    tmp121 = tl.where(tmp118 < tmp5, tmp118, tmp1)
    tmp122 = tmp121 < tmp6
    tmp123 = tl.where(tmp122, tmp121, tmp1)
    tmp124 = tmp123 < tmp7
    tmp125 = tl.where(tmp124, tmp123, tmp1)
    tmp126 = tl.full(tmp125.shape, 0, tmp125.dtype)
    tmp127 = tl.where(tmp125 != tmp126, tmp125, tmp126)
    tmp128 = tl.where(tmp125 < tmp5, tmp125, tmp1)
    tmp129 = tmp128 < tmp6
    tmp130 = tl.where(tmp129, tmp128, tmp1)
    tmp131 = tmp130 < tmp7
    tmp132 = tl.where(tmp131, tmp130, tmp1)
    tmp133 = tl.full(tmp132.shape, 0, tmp132.dtype)
    tmp134 = tl.where(tmp132 != tmp133, tmp132, tmp133)
    tmp135 = tl.where(tmp132 < tmp5, tmp132, tmp1)
    tmp136 = tmp135 < tmp6
    tmp137 = tl.where(tmp136, tmp135, tmp1)
    tmp138 = tmp137 < tmp7
    tmp139 = tl.where(tmp138, tmp137, tmp1)
    tmp140 = tl.full(tmp139.shape, 0, tmp139.dtype)
    tmp141 = tl.where(tmp139 != tmp140, tmp139, tmp140)
    tmp142 = tl.where(tmp139 < tmp5, tmp139, tmp1)
    tmp143 = tmp142 < tmp6
    tmp144 = tl.where(tmp143, tmp142, tmp1)
    tmp145 = tmp144 < tmp7
    tmp146 = tl.where(tmp145, tmp144, tmp1)
    tmp147 = tl.full(tmp146.shape, 0, tmp146.dtype)
    tmp148 = tl.where(tmp146 != tmp147, tmp146, tmp147)
    tmp149 = tl.where(tmp146 < tmp5, tmp146, tmp1)
    tmp150 = tmp149 < tmp6
    tmp151 = tl.where(tmp150, tmp149, tmp1)
    tmp152 = tmp151 < tmp7
    tmp153 = tl.where(tmp152, tmp151, tmp1)
    tmp154 = tl.full(tmp153.shape, 0, tmp153.dtype)
    tmp155 = tl.where(tmp153 != tmp154, tmp153, tmp154)
    tmp156 = tl.where(tmp153 < tmp5, tmp153, tmp1)
    tmp157 = tmp156 < tmp6
    tmp158 = tl.where(tmp157, tmp156, tmp1)
    tmp159 = tmp158 < tmp7
    tmp160 = tl.where(tmp159, tmp158, tmp1)
    tmp161 = tl.full(tmp160.shape, 0, tmp160.dtype)
    tmp162 = tl.where(tmp160 != tmp161, tmp160, tmp161)
    tmp163 = tl.where(tmp160 < tmp5, tmp160, tmp1)
    tmp164 = tmp163 < tmp6
    tmp165 = tl.where(tmp164, tmp163, tmp1)
    tmp166 = tmp165 < tmp7
    tmp167 = tl.where(tmp166, tmp165, tmp1)
    tmp168 = tl.full(tmp167.shape, 0, tmp167.dtype)
    tmp169 = tl.where(tmp167 != tmp168, tmp167, tmp168)
    tmp170 = tl.where(tmp167 < tmp5, tmp167, tmp1)
    tmp171 = tmp170 < tmp6
    tmp172 = tl.where(tmp171, tmp170, tmp1)
    tmp173 = tmp172 < tmp7
    tmp174 = tl.where(tmp173, tmp172, tmp1)
    tmp175 = tl.full(tmp174.shape, 0, tmp174.dtype)
    tmp176 = tl.where(tmp174 != tmp175, tmp174, tmp175)
    tmp177 = tl.where(tmp174 < tmp5, tmp174, tmp1)
    tmp178 = tmp177 < tmp6
    tmp179 = tl.where(tmp178, tmp177, tmp1)
    tmp180 = tmp179 < tmp7
    tmp181 = tl.where(tmp180, tmp179, tmp1)
    tmp182 = tl.full(tmp181.shape, 0, tmp181.dtype)
    tmp183 = tl.where(tmp181 != tmp182, tmp181, tmp182)
    tmp184 = tl.where(tmp181 < tmp5, tmp181, tmp1)
    tmp185 = tmp184 < tmp6
    tmp186 = tl.where(tmp185, tmp184, tmp1)
    tmp187 = tmp186 < tmp7
    tmp188 = tl.where(tmp187, tmp186, tmp1)
    tmp189 = tl.full(tmp188.shape, 0, tmp188.dtype)
    tmp190 = tl.where(tmp188 != tmp189, tmp188, tmp189)
    tmp191 = tl.where(tmp188 < tmp5, tmp188, tmp1)
    tmp192 = tmp191 < tmp6
    tmp193 = tl.where(tmp192, tmp191, tmp1)
    tmp194 = tmp193 < tmp7
    tmp195 = tl.where(tmp194, tmp193, tmp1)
    tmp196 = tl.full(tmp195.shape, 0, tmp195.dtype)
    tmp197 = tl.where(tmp195 != tmp196, tmp195, tmp196)
    tmp198 = tl.where(tmp195 < tmp5, tmp195, tmp1)
    tmp199 = tmp198 < tmp6
    tmp200 = tl.where(tmp199, tmp198, tmp1)
    tmp201 = tmp200 < tmp7
    tmp202 = tl.where(tmp201, tmp200, tmp1)
    tmp203 = tl.full(tmp202.shape, 0, tmp202.dtype)
    tmp204 = tl.where(tmp202 != tmp203, tmp202, tmp203)
    tmp205 = tl.where(tmp202 < tmp5, tmp202, tmp1)
    tmp206 = tmp205 < tmp6
    tmp207 = tl.where(tmp206, tmp205, tmp1)
    tmp208 = tmp207 < tmp7
    tmp209 = tl.where(tmp208, tmp207, tmp1)
    tmp210 = tl.full(tmp209.shape, 0, tmp209.dtype)
    tmp211 = tl.where(tmp209 != tmp210, tmp209, tmp210)
    tmp212 = tl.where(tmp209 < tmp5, tmp209, tmp1)
    tmp213 = tmp212 < tmp6
    tmp214 = tl.where(tmp213, tmp212, tmp1)
    tmp215 = tmp214 < tmp7
    tmp216 = tl.where(tmp215, tmp214, tmp1)
    tmp217 = tl.full(tmp216.shape, 0, tmp216.dtype)
    tmp218 = tl.where(tmp216 != tmp217, tmp216, tmp217)
    tmp219 = tl.where(tmp216 < tmp5, tmp216, tmp1)
    tmp220 = tmp219 < tmp6
    tmp221 = tl.where(tmp220, tmp219, tmp1)
    tmp222 = tmp221 < tmp7
    tmp223 = tl.where(tmp222, tmp221, tmp1)
    tmp224 = tl.full(tmp223.shape, 0, tmp223.dtype)
    tmp225 = tl.where(tmp223 != tmp224, tmp223, tmp224)
    tmp226 = tl.where(tmp223 < tmp5, tmp223, tmp1)
    tmp227 = tmp226 < tmp6
    tmp228 = tl.where(tmp227, tmp226, tmp1)
    tmp229 = tmp228 < tmp7
    tmp230 = tl.where(tmp229, tmp228, tmp1)
    tmp231 = tl.full(tmp230.shape, 0, tmp230.dtype)
    tmp232 = tl.where(tmp230 != tmp231, tmp230, tmp231)
    tmp233 = tl.where(tmp230 < tmp5, tmp230, tmp1)
    tmp234 = tmp233 < tmp6
    tmp235 = tl.where(tmp234, tmp233, tmp1)
    tmp236 = tmp235 < tmp7
    tmp237 = tl.where(tmp236, tmp235, tmp1)
    tmp238 = tl.full(tmp237.shape, 0, tmp237.dtype)
    tmp239 = tl.where(tmp237 != tmp238, tmp237, tmp238)
    tmp240 = tl.where(tmp237 < tmp5, tmp237, tmp1)
    tmp241 = tmp240 < tmp6
    tmp242 = tl.where(tmp241, tmp240, tmp1)
    tmp243 = tmp242 < tmp7
    tmp244 = tl.where(tmp243, tmp242, tmp1)
    tmp245 = tl.full(tmp244.shape, 0, tmp244.dtype)
    tmp246 = tl.where(tmp244 != tmp245, tmp244, tmp245)
    tmp247 = tl.where(tmp244 < tmp5, tmp244, tmp1)
    tmp248 = tmp247 < tmp6
    tmp249 = tl.where(tmp248, tmp247, tmp1)
    tmp250 = tmp249 < tmp7
    tmp251 = tl.where(tmp250, tmp249, tmp1)
    tmp252 = tl.full(tmp251.shape, 0, tmp251.dtype)
    tmp253 = tl.where(tmp251 != tmp252, tmp251, tmp252)
    tmp254 = tl.where(tmp251 < tmp5, tmp251, tmp1)
    tmp255 = tmp254 < tmp6
    tmp256 = tl.where(tmp255, tmp254, tmp1)
    tmp257 = tmp256 < tmp7
    tmp258 = tl.where(tmp257, tmp256, tmp1)
    tmp259 = tl.full(tmp258.shape, 0, tmp258.dtype)
    tmp260 = tl.where(tmp258 != tmp259, tmp258, tmp259)
    tmp261 = tl.where(tmp258 < tmp5, tmp258, tmp1)
    tmp262 = tmp261 < tmp6
    tmp263 = tl.where(tmp262, tmp261, tmp1)
    tmp264 = tmp263 < tmp7
    tmp265 = tl.where(tmp264, tmp263, tmp1)
    tmp266 = tl.full(tmp265.shape, 0, tmp265.dtype)
    tmp267 = tl.where(tmp265 != tmp266, tmp265, tmp266)
    tmp268 = tl.where(tmp265 < tmp5, tmp265, tmp1)
    tmp269 = tmp268 < tmp6
    tmp270 = tl.where(tmp269, tmp268, tmp1)
    tmp271 = tmp270 < tmp7
    tmp272 = tl.where(tmp271, tmp270, tmp1)
    tmp273 = tl.full(tmp272.shape, 0, tmp272.dtype)
    tmp274 = tl.where(tmp272 != tmp273, tmp272, tmp273)
    tmp275 = tl.where(tmp272 < tmp5, tmp272, tmp1)
    tmp276 = tmp275 < tmp6
    tmp277 = tl.where(tmp276, tmp275, tmp1)
    tmp278 = tmp277 < tmp7
    tmp279 = tl.where(tmp278, tmp277, tmp1)
    tmp280 = tl.full(tmp279.shape, 0, tmp279.dtype)
    tmp281 = tl.where(tmp279 != tmp280, tmp279, tmp280)
    tmp282 = tl.where(tmp279 < tmp5, tmp279, tmp1)
    tmp283 = tmp282 < tmp6
    tmp284 = tl.where(tmp283, tmp282, tmp1)
    tmp285 = tmp284 < tmp7
    tmp286 = tl.where(tmp285, tmp284, tmp1)
    tmp287 = tl.full(tmp286.shape, 0, tmp286.dtype)
    tmp288 = tl.where(tmp286 != tmp287, tmp286, tmp287)
    tmp289 = tl.where(tmp286 < tmp5, tmp286, tmp1)
    tmp290 = tmp289 < tmp6
    tmp291 = tl.where(tmp290, tmp289, tmp1)
    tmp292 = tmp291 < tmp7
    tmp293 = tl.where(tmp292, tmp291, tmp1)
    tmp294 = tl.full(tmp293.shape, 0, tmp293.dtype)
    tmp295 = tl.where(tmp293 != tmp294, tmp293, tmp294)
    tmp296 = tl.where(tmp293 < tmp5, tmp293, tmp1)
    tmp297 = tmp296 < tmp6
    tmp298 = tl.where(tmp297, tmp296, tmp1)
    tmp299 = tmp298 < tmp7
    tmp300 = tl.where(tmp299, tmp298, tmp1)
    tmp301 = tl.full(tmp300.shape, 0, tmp300.dtype)
    tmp302 = tl.where(tmp300 != tmp301, tmp300, tmp301)
    tmp303 = tl.where(tmp300 < tmp5, tmp300, tmp1)
    tmp304 = tmp303 < tmp6
    tmp305 = tl.where(tmp304, tmp303, tmp1)
    tmp306 = tmp305 < tmp7
    tmp307 = tl.where(tmp306, tmp305, tmp1)
    tmp308 = tl.full(tmp307.shape, 0, tmp307.dtype)
    tmp309 = tl.where(tmp307 != tmp308, tmp307, tmp308)
    tmp310 = tl.where(tmp307 < tmp5, tmp307, tmp1)
    tmp311 = tmp310 < tmp6
    tmp312 = tl.where(tmp311, tmp310, tmp1)
    tmp313 = tmp312 < tmp7
    tmp314 = tl.where(tmp313, tmp312, tmp1)
    tmp315 = tl.full(tmp314.shape, 0, tmp314.dtype)
    tmp316 = tl.where(tmp314 != tmp315, tmp314, tmp315)
    tmp317 = tl.where(tmp314 < tmp5, tmp314, tmp1)
    tmp318 = tmp317 < tmp6
    tmp319 = tl.where(tmp318, tmp317, tmp1)
    tmp320 = tmp319 < tmp7
    tmp321 = tl.where(tmp320, tmp319, tmp1)
    tmp322 = tl.full(tmp321.shape, 0, tmp321.dtype)
    tmp323 = tl.where(tmp321 != tmp322, tmp321, tmp322)
    tmp324 = tl.where(tmp321 < tmp5, tmp321, tmp1)
    tmp325 = tmp324 < tmp6
    tmp326 = tl.where(tmp325, tmp324, tmp1)
    tmp327 = tmp326 < tmp7
    tmp328 = tl.where(tmp327, tmp326, tmp1)
    tmp329 = tl.full(tmp328.shape, 0, tmp328.dtype)
    tmp330 = tl.where(tmp328 != tmp329, tmp328, tmp329)
    tmp331 = tl.where(tmp328 < tmp5, tmp328, tmp1)
    tmp332 = tmp331 < tmp6
    tmp333 = tl.where(tmp332, tmp331, tmp1)
    tmp334 = tmp333 < tmp7
    tmp335 = tl.where(tmp334, tmp333, tmp1)
    tmp336 = tl.full(tmp335.shape, 0, tmp335.dtype)
    tmp337 = tl.where(tmp335 != tmp336, tmp335, tmp336)
    tmp338 = tl.where(tmp335 < tmp5, tmp335, tmp1)
    tmp339 = tmp338 < tmp6
    tmp340 = tl.where(tmp339, tmp338, tmp1)
    tmp341 = tmp340 < tmp7
    tmp342 = tl.where(tmp341, tmp340, tmp1)
    tmp343 = tl.full(tmp342.shape, 0, tmp342.dtype)
    tmp344 = tl.where(tmp342 != tmp343, tmp342, tmp343)
    tmp345 = tl.where(tmp342 < tmp5, tmp342, tmp1)
    tmp346 = tmp345 < tmp6
    tmp347 = tl.where(tmp346, tmp345, tmp1)
    tmp348 = tmp347 < tmp7
    tmp349 = tl.where(tmp348, tmp347, tmp1)
    tmp350 = tl.full(tmp349.shape, 0, tmp349.dtype)
    tmp351 = tl.where(tmp349 != tmp350, tmp349, tmp350)
    tmp352 = tl.where(tmp349 < tmp5, tmp349, tmp1)
    tmp353 = tmp352 < tmp6
    tmp354 = tl.where(tmp353, tmp352, tmp1)
    tmp355 = tmp354 < tmp7
    tmp356 = tl.where(tmp355, tmp354, tmp1)
    tmp357 = tl.full(tmp356.shape, 0, tmp356.dtype)
    tmp358 = tl.where(tmp356 != tmp357, tmp356, tmp357)
    tmp359 = tl.where(tmp356 < tmp5, tmp356, tmp1)
    tmp360 = tmp359 < tmp6
    tmp361 = tl.where(tmp360, tmp359, tmp1)
    tmp362 = tmp361 < tmp7
    tmp363 = tl.where(tmp362, tmp361, tmp1)
    tmp364 = tl.full(tmp363.shape, 0, tmp363.dtype)
    tmp365 = tl.where(tmp363 != tmp364, tmp363, tmp364)
    tmp366 = tl.where(tmp363 < tmp5, tmp363, tmp1)
    tmp367 = tmp366 < tmp6
    tmp368 = tl.where(tmp367, tmp366, tmp1)
    tmp369 = tmp368 < tmp7
    tmp370 = tl.where(tmp369, tmp368, tmp1)
    tmp371 = tl.full(tmp370.shape, 0, tmp370.dtype)
    tmp372 = tl.where(tmp370 != tmp371, tmp370, tmp371)
    tmp373 = tl.where(tmp370 < tmp5, tmp370, tmp1)
    tmp374 = tmp373 < tmp6
    tmp375 = tl.where(tmp374, tmp373, tmp1)
    tmp376 = tmp375 < tmp7
    tmp377 = tl.where(tmp376, tmp375, tmp1)
    tmp378 = tl.full(tmp377.shape, 0, tmp377.dtype)
    tmp379 = tl.where(tmp377 != tmp378, tmp377, tmp378)
    tmp380 = tl.where(tmp377 < tmp5, tmp377, tmp1)
    tmp381 = tmp380 < tmp6
    tmp382 = tl.where(tmp381, tmp380, tmp1)
    tmp383 = tmp382 < tmp7
    tmp384 = tl.where(tmp383, tmp382, tmp1)
    tmp385 = tl.full(tmp384.shape, 0, tmp384.dtype)
    tmp386 = tl.where(tmp384 != tmp385, tmp384, tmp385)
    tmp387 = tl.where(tmp384 < tmp5, tmp384, tmp1)
    tmp388 = tmp387 < tmp6
    tmp389 = tl.where(tmp388, tmp387, tmp1)
    tmp390 = tmp389 < tmp7
    tmp391 = tl.where(tmp390, tmp389, tmp1)
    tmp392 = tl.full(tmp391.shape, 0, tmp391.dtype)
    tmp393 = tl.where(tmp391 != tmp392, tmp391, tmp392)
    tmp394 = tl.where(tmp391 < tmp5, tmp391, tmp1)
    tmp395 = tmp394 < tmp6
    tmp396 = tl.where(tmp395, tmp394, tmp1)
    tmp397 = tmp396 < tmp7
    tmp398 = tl.where(tmp397, tmp396, tmp1)
    tmp399 = tl.full(tmp398.shape, 0, tmp398.dtype)
    tmp400 = tl.where(tmp398 != tmp399, tmp398, tmp399)
    tmp401 = tl.where(tmp398 < tmp5, tmp398, tmp1)
    tmp402 = tmp401 < tmp6
    tmp403 = tl.where(tmp402, tmp401, tmp1)
    tmp404 = tmp403 < tmp7
    tmp405 = tl.where(tmp404, tmp403, tmp1)
    tmp40
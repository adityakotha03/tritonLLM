import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4,
    in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 222059520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15648
    x1 = xindex // 15648
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 10, tl.int64)
    tmp4 = tmp0 < 32
    tmp5 = tl.full([1], 32, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp4 | tmp6
    tmp8 = tmp0 < 64
    tmp9 = tmp0 >= 32
    tmp10 = tmp8 & tmp9
    tmp11 = tmp7 | tmp10
    tmp12 = tmp0 < 192
    tmp13 = tmp0 >= 64
    tmp14 = tmp12 & tmp13
    tmp15 = tmp11 | tmp14
    tmp16 = tmp0 < 480
    tmp17 = tmp0 >= 192
    tmp18 = tmp16 & tmp17
    tmp19 = tmp15 | tmp18
    tmp20 = tl.load(in_ptr0 + x2, xmask & tmp19, eviction_policy='evict_last'
        )
    tmp21 = tmp0 >= 480
    tmp22 = tmp0 < 576
    tmp23 = tmp21 & tmp22
    tmp24 = tl.full([1], 576, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 | tmp25
    tmp27 = tmp0 >= 576
    tmp28 = tmp0 < 800
    tmp29 = tmp27 & tmp28
    tmp30 = tmp26 | tmp29
    tmp31 = tmp0 >= 800
    tmp32 = tmp0 < 15648
    tmp33 = tmp31 & tmp32
    tmp34 = tmp30 | tmp33
    tmp35 = tl.load(in_ptr1 + x2, xmask & tmp34, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + x2, xmask & tmp34, eviction_policy='evict_last')
    tmp37 = tmp26 & tmp29
    tmp38 = tl.full([1], 15648, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tmp37 | tmp39
    tmp41 = tmp36 < 208
    tmp42 = tmp36 >= 0
    tmp43 = tmp41 & tmp42
    tmp44 = tmp40 & tmp43
    tmp45 = tl.load(in_ptr3 + (tmp36 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp44,
        eviction_policy='evict_last')
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp34 & tmp40
    tmp48 = tmp35 < 208
    tmp49 = tmp35 >= 0
    tmp50 = tmp48 & tmp49
    tmp51 = tmp47 & tmp50
    tmp52 = tl.load(in_ptr4 + (tmp35 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp51,
        eviction_policy='evict_last')
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp22 & tmp29
    tmp55 = tmp29 | tmp54
    tmp56 = tl.load(in_ptr5 + (tmp36 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp55,
        eviction_policy='evict_last')
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp40 & tmp51
    tmp59 = tmp50 | tmp58
    tmp60 = tl.load(in_ptr6 + (tmp35 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp59,
        eviction_policy='evict_last')
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp25 & tmp40
    tmp63 = tmp40 | tmp62
    tmp64 = tmp39 & tmp59
    tmp65 = tmp58 | tmp64
    tmp66 = tl.load(in_ptr7 + (tmp35 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp65,
        eviction_policy='evict_last')
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp29 & tmp40
    tmp69 = tmp40 | tmp68
    tmp70 = tmp25 & tmp59
    tmp71 = tmp59 | tmp70
    tmp72 = tl.load(in_ptr8 + (tmp35 + 208 * (x1 + 15648 * x0) + 15648 * 576 *
        tmp25 + 15648 * 192 * tmp12 + 192 * 192 * tmp8 + 192 * 64 * tmp13 + 
        64 * 64 * tmp10 + 32 * 32 * tmp4 + 0 * tmp11 + 192 * 32 * tmp10 + 
        32 * 64 * tmp13 + 64 * 192 * tmp28 + 192 * 192 * tmp17), tmp71,
        eviction_policy='evict_last')
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tl.where(tmp61 < tmp67, tmp67, tmp61)
    tmp75 = tl.where(tmp53 < tmp57, tmp57, tmp53)
    tmp76 = tl.where(tmp47 < tmp51, tmp51, tmp47)
    tmp77 = tl.where(tmp45 < tmp49, tmp49, tmp45)
    tmp78 = tl.where(tmp36 < tmp40, tmp40, tmp36)
    tmp79 = tl.where(tmp35 < tmp39, tmp39, tmp35)
    tmp80 = tl.where(tmp29 < tmp33, tmp33, tmp29)
    tmp81 = tl.where(tmp25 < tmp28, tmp28, tmp25)
    tmp82 = tl.where(tmp13 < tmp16, tmp16, tmp13)
    tmp83 = tl.where(tmp8 < tmp11, tmp11, tmp8)
    tmp84 = tl.where(tmp4 < tmp7, tmp7, tmp4)
    tmp85 = tl.where(tmp0 < tmp3, tmp3, tmp0)
    tmp86 = tl.where(tmp34 < tmp38, tmp38, tmp34)
    tmp87 = tl.where(tmp30 < tmp35, tmp35, tmp30)
    tmp88 = tl.where(tmp26 < tmp31, tmp31, tmp26)
    tmp89 = tl.where(tmp22 < tmp27, tmp27, tmp22)
    tmp90 = tl.where(tmp18 < tmp21, tmp21, tmp18)
    tmp91 = tl.where(tmp14 < tmp17, tmp17, tmp14)
    tmp92 = tl.where(tmp10 < tmp13, tmp13, tmp10)
    tmp93 = tl.where(tmp6 < tmp9, tmp9, tmp6)
    tmp94 = tl.where(tmp3 < tmp6, tmp6, tmp3)
    tmp95 = tl.where(tmp0 < tmp3, tmp3, tmp0)
    tmp96 = tmp24
    tmp97 = tmp23
    tmp98 = tl.where(tmp69, tmp67, tmp61)
    tmp99 = tl.where(tmp75, tmp57, tmp53)
    tmp100 = tl.where(tmp79, tmp51, tmp47)
    tmp101 = tl.where(tmp77, tmp49, tmp45)
    tmp102 = tl.where(tmp85, tmp39, tmp35)
    tmp103 = tl.where(tmp89, tmp33, tmp29)
    tmp104 = tl.where(tmp88, tmp31, tmp27)
    tmp105 = tl.where(tmp87, tmp30, tmp26)
    tmp106 = tl.where(tmp86, tmp34, tmp30)
    tmp107 = tl.where(tmp84, tmp32, tmp28)
    tmp108 = tl.where(tmp83, tmp30, tmp24)
    tmp109 = tl.where(tmp82, tmp32, tmp20)
    tmp110 = tl.where(tmp81, tmp34, tmp16)
    tmp111 = tl.where(tmp80, tmp36, tmp12)
    tmp112 = tl.where(tmp76, tmp38, tmp10)
    tmp113 = tl.where(tmp74, tmp40, tmp6)
    tmp114 = tl.where(tmp72, tmp42, tmp4)
    tmp115 = tl.where(tmp70, tmp44, tmp2)
    tmp116 = tl.where(tmp68, tmp46, tmp0)
    tmp117 = tmp88
    tmp118 = tl.where(tmp117, tmp31, tmp27)
    tmp119 = tl.where(tmp118, tmp30, tmp26)
    tmp120 = tl.where(tmp119, tmp29, tmp25)
    tmp121 = tl.where(tmp120, tmp28, tmp24)
    tmp122 = tl.where(tmp121, tmp27, tmp23)
    tmp123 = tl.where(tmp122, tmp26, tmp22)
    tmp124 = tl.where(tmp123, tmp25, tmp21)
    tmp125 = tl.where(tmp124, tmp24, tmp19)
    tmp126 = tl.where(tmp125, tmp23, tmp18)
    tmp127 = tl.where(tmp126, tmp22, tmp17)
    tmp128 = tl.where(tmp127, tmp21, tmp16)
    tmp129 = tl.where(tmp128, tmp20, tmp15)
    tmp130 = tl.where(tmp129, tmp19, tmp14)
    tmp131 = tl.where(tmp130, tmp18, tmp13)
    tmp132 = tl.where(tmp131, tmp17, tmp12)
    tmp133 = tl.where(tmp132, tmp16, tmp11)
    tmp134 = tl.where(tmp133, tmp15, tmp10)
    tmp135 = tl.where(tmp134, tmp14, tmp9)
    tmp136 = tl.where(tmp135, tmp13, tmp8)
    tmp137 = tl.where(tmp136, tmp12, tmp7)
    tmp138 = tl.where(tmp137, tmp11, tmp6)
    tmp139 = tl.where(tmp138, tmp10, tmp5)
    tmp140 = tl.where(tmp139, tmp9, tmp4)
    tmp141 = tl.where(tmp140, tmp8, tmp3)
    tmp142 = tl.where(tmp141, tmp7, tmp2)
    tmp143 = tl.where(tmp142, tmp6, tmp1)
    tmp144 = tl.where(tmp143, tmp5, tmp0)
    tmp145 = tl.where(tmp74, tmp72, tmp70)
    tmp146 = tl.where(tmp107, tmp105, tmp103)
    tmp147 = tl.where(tmp145, tmp144, tmp142)
    tmp148 = tl.where(tmp101, tmp98, tmp96)
    tmp149 = tl.where(tmp147, tmp146, tmp144)
    tmp150 = tl.where(tmp105, tmp94, tmp92)
    tmp151 = tl.where(tmp149, tmp148, tmp146)
    tmp152 = tl.where(tmp99, tmp97, tmp95)
    tmp153 = tl.where(tmp151, tmp150, tmp148)
    tmp154 = tl.where(tmp93, tmp91, tmp89)
    tmp155 = tl.where(tmp153, tmp152, tmp150)
    tmp156 = tl.where(tmp87, tmp85, tmp83)
    tmp157 = tl.where(tmp155, tmp154, tmp152)
    tmp158 = tl.where(tmp81, tmp79, tmp77)
    tmp159 = tl.where(tmp157, tmp156, tmp154)
    tmp160 = tl.where(tmp75, tmp73, tmp71)
    tmp161 = tl.where(tmp159, tmp158, tmp156)
    tmp162 = tl.where(tmp71, tmp69, tmp67)
    tmp163 = tl.where(tmp161, tmp160, tmp158)
    tmp164 = tl.where(tmp69, tmp68, tmp66)
    tmp165 = tl.where(tmp163, tmp162, tmp160)
    tmp166 = tl.where(tmp67, tmp65, tmp63)
    tmp167 = tl.where(tmp165, tmp164, tmp162)
    tmp168 = tl.where(tmp65, tmp64, tmp62)
    tmp169 = tl.where(tmp167, tmp166, tmp164)
    tmp170 = tl.where(tmp63, tmp61, tmp59)
    tmp171 = tl.where(tmp169, tmp168, tmp166)
    tmp172 = tl.where(tmp59, tmp57, tmp55)
    tmp173 = tl.where(tmp171, tmp170, tmp168)
    tmp174 = tl.where(tmp57, tmp53, tmp51)
    tmp175 = tl.where(tmp173, tmp172, tmp170)
    tmp176 = tl.where(tmp53, tmp49, tmp47)
    tmp177 = tl.where(tmp175, tmp174, tmp172)
    tmp178 = tl.where(tmp49, tmp45, tmp43)
    tmp179 = tl.where(tmp177, tmp176, tmp174)
    tmp180 = tl.where(tmp45, tmp41, tmp39)
    tmp181 = tl.where(tmp179, tmp178, tmp176)
    tmp182 = tl.where(tmp41, tmp37, tmp35)
    tmp183 = tl.where(tmp181, tmp180, tmp178)
    tmp184 = tl.where(tmp37, tmp33, tmp31)
    tmp185 = tl.where(tmp183, tmp182, tmp180)
    tmp186 = tl.where(tmp33, tmp29, tmp27)
    tmp187 = tl.where(tmp185, tmp184, tmp182)
    tmp188 = tl.where(tmp29, tmp25, tmp23)
    tmp189 = tl.where(tmp187, tmp186, tmp184)
    tmp190 = tl.where(tmp25, tmp21, tmp19)
    tmp191 = tl.where(tmp189, tmp188, tmp186)
    tmp192 = tl.where(tmp21, tmp17, tmp15)
    tmp193 = tl.where(tmp191, tmp190, tmp188)
    tmp194 = tl.where(tmp17, tmp13, tmp11)
    tmp195 = tl.where(tmp193, tmp192, tmp190)
    tmp196 = tl.where(tmp13, tmp9, tmp7)
    tmp197 = tl.where(tmp195, tmp194, tmp192)
    tmp198 = tl.where(tmp9, tmp5, tmp3)
    tmp199 = tl.where(tmp197, tmp196, tmp194)
    tmp200 = tl.where(tmp5, tmp1, tmp0)
    tmp201 = tl.where(tmp199, tmp198, tmp196)
    tmp202 = tl.where(tmp1, tmp0, tmp1)
    tmp203 = tl.where(tmp201, tmp202, tmp199)
    tmp204 = tl.where(tmp0, tmp1, tmp0)
    tmp205 = tl.where(tmp203, tmp204, tmp202)
    tmp206 = tl.where(tmp1, tmp0, tmp0)
    tmp207 = tl.where(tmp205, tmp206, tmp204)
    tmp208 = tl.where(tmp0, tmp1, tmp1)
    tmp209 = tl.where(tmp207, tmp208, tmp206)
    tmp210 = tl.where(tmp1, tmp0, tmp1)
    tmp211 = tl.where(tmp209, tmp210, tmp208)
    tmp212 = tl.where(tmp0, tmp1, tmp1)
    tmp213 = tl.where(tmp211, tmp212, tmp210)
    tmp214 = tl.where(tmp1, tmp0, tmp1)
    tmp215 = tl.where(tmp213, tmp214, tmp212)
    tmp216 = tl.where(tmp0, tmp1, tmp1)
    tmp217 = tl.where(tmp215, tmp216, tmp214)
    tmp218 = tl.where(tmp1, tmp0, tmp1)
    tmp219 = tl.where(tmp217, tmp218, tmp216)
    tmp220 = tl.where(tmp0, tmp1, tmp1)
    tmp221 = tl.where(tmp219, tmp220, tmp218)
    tmp222 = tl.where(tmp1, tmp0, tmp1)
    tmp223 = tl.where(tmp221, tmp222, tmp220)
    tmp224 = tl.where(tmp0, tmp1, tmp1)
    tmp225 = tl.where(tmp223, tmp224, tmp222)
    tmp226 = tl.where(tmp1, tmp0, tmp1)
    tmp227 = tl.where(tmp225, tmp226, tmp224)
    tmp228 = tl.where(tmp0, tmp1, tmp1)
    tmp229 = tl.where(tmp227, tmp228, tmp226)
    tmp230 = tl.where(tmp1, tmp0, tmp1)
    tmp231 = tl.where(tmp229, tmp230, tmp228)
    tmp232 = tl.where(tmp0, tmp1, tmp1)
    tmp233 = tl.where(tmp231, tmp232, tmp230)
    tmp234 = tl.where(tmp1, tmp0, tmp1)
    tmp235 = tl.where(tmp233, tmp234, tmp232)
    tmp236 = tl.where(tmp0, tmp1, tmp1)
    tmp237 = tl.where(tmp235, tmp236, tmp234)
    tmp238 = tl.where(tmp1, tmp0, tmp1)
    tmp239 = tl.where(tmp237, tmp240, tmp236)
    tmp240 = tmp239
    tmp241 = tl.where(tmp0, tmp1, tmp1)
    tmp242 = tl.where(tmp239, tmp241, tmp239)
    tmp243 = tl.where(tmp1, tmp0, tmp1)
    tmp244 = tl.where(tmp242, tmp243, tmp241)
    tmp245 = tl.where(tmp0, tmp1, tmp1)
    tmp246 = tl.where(tmp244, tmp245, tmp243)
    tmp247 = tl.where(tmp1, tmp0, tmp1)
    tmp248 = tl.where(tmp246, tmp247, tmp245)
    tmp249 = tl.where(tmp0, tmp1, tmp1)
    tmp250 = tl.where(tmp248, tmp249, tmp247)
    tmp251 = tl.where(tmp1, tmp0, tmp1)
    tmp252 = tl.where(tmp250, tmp251, tmp249)
    tmp253 = tl.where(tmp0, tmp1, tmp1)
    tmp254 = tl.where(tmp252, tmp253, tmp251)
    tmp255 = tl.where(tmp1, tmp0, tmp1)
    tmp256 = tl.where(tmp254, tmp255, tmp253)
    tmp257 = tl.where(tmp0, tmp1, tmp1)
    tmp258 = tl.where(tmp256, tmp257, tmp255)
    tmp259 = tl.where(tmp1, tmp0, tmp1)
    tmp260 = tl.where(tmp258, tmp259, tmp257)
    tmp261 = tl.where(tmp0, tmp1, tmp1)
    tmp262 = tl.where(tmp260, tmp261, tmp259)
    tmp263 = tl.where(tmp1, tmp0, tmp1)
    tmp264 = tl.where(tmp262, tmp263, tmp261)
    tmp265 = tl.where(tmp0, tmp1, tmp1)
    tmp266 = tl.where(tmp264, tmp265, tmp263)
    tmp267 = tl.where(tmp1, tmp0, tmp1)
    tmp268 = tl.where(tmp266, tmp267, tmp265)
    tmp269 = tl.where(tmp0, tmp1, tmp1)
    tmp270 = tl.where(tmp268, tmp269, tmp267)
    tmp271 = tl.where(tmp1, tmp0, tmp1)
    tmp272 = tl.where(tmp270, tmp271, tmp269)
    tmp273 = tl.where(tmp0, tmp1, tmp1)
    tmp274 = tl.where(tmp272, tmp273, tmp271)
    tmp275 = tl.where(tmp1, tmp0, tmp1)
    tmp276 = tl.where(tmp274, tmp275, tmp273)
    tmp277 = tl.where(tmp0, tmp1, tmp1)
    tmp278 = tl.where(tmp276, tmp277, tmp275)
    tmp279 = tl.where(tmp1, tmp0, tmp1)
    tmp280 = tl.where(tmp278, tmp279, tmp277)
    tmp281 = tl.where(tmp0, tmp1, tmp1)
    tmp282 = tl.where(tmp280, tmp281, tmp279)
    tmp283 = tl.where(tmp1, tmp0, tmp1)
    tmp284 = tl.where(tmp282, tmp283, tmp281)
    tmp285 = tl.where(tmp0, tmp1, tmp1)
    tmp286 = tl.where(tmp284, tmp285, tmp283)
    tmp287 = tl.where(tmp1, tmp0, tmp1)
    tmp288 = tl.where(tmp286, tmp287, tmp285)
    tmp289 = tl.where(tmp0, tmp1, tmp1)
    tmp290 = tl.where(tmp288, tmp289, tmp287)
    tmp291 = tl.where(tmp1, tmp0, tmp1)
    tmp292 = tl.where(tmp290, tmp291, tmp289)
    tmp293 = tl.where(tmp0, tmp1, tmp1)
    tmp294 = tl.where(tmp292, tmp293, tmp291)
    tmp295 = tl.where(tmp1, tmp0, tmp1)
    tmp296 = tl.where(tmp294, tmp295, tmp293)
    tmp297 = tl.where(tmp0, tmp1, tmp1)
    tmp298 = tl.where(tmp296, tmp297, tmp295)
    tmp299 = tl.where(tmp1, tmp0, tmp1)
    tmp300 = tl.where(tmp298, tmp299, tmp297)
    tmp301 = tl.where(tmp0, tmp1, tmp1)
    tmp302 = tl.where(tmp300, tmp301, tmp299)
    tmp303 = tl.where(tmp1, tmp0, tmp1)
    tmp304 = tl.where(tmp302, tmp303, tmp301)
    tmp305 = tl.where(tmp0, tmp1, tmp1)
    tmp306 = tl.where(tmp304, tmp305, tmp303)
    tmp307 = tl.where(tmp1, tmp0, tmp1)
    tmp308 = tl.where(tmp306, tmp307, tmp305)
    tmp309 = tl.where(tmp0, tmp1, tmp1)
    tmp310 = tl.where(tmp308, tmp309, tmp307)
    tmp311 = tl.where(tmp1, tmp0, tmp1)
    tmp312 = tl.where(tmp310, tmp311, tmp309)
    tmp313 = tl.where(tmp0, tmp1, tmp1)
    tmp314 = tl.where(tmp312, tmp313, tmp311)
    tmp315 = tl.where(tmp1, tmp0, tmp1)
    tmp316 = tl.where(tmp314, tmp315, tmp313)
    tmp317 = tl.where(tmp0, tmp1, tmp1)
    tmp318 = tl.where(tmp316, tmp317, tmp315)
    tmp319 = tl.where(tmp1, tmp0, tmp1)
    tmp320 = tl.where(tmp318, tmp319, tmp317)
    tmp321 = tl.where(tmp0, tmp1, tmp1)
    tmp322 = tl.where(tmp320, tmp321, tmp319)
    tmp323 = tl.where(tmp1, tmp0, tmp1)
    tmp324 = tl.where(tmp322, tmp323, tmp321)
    tmp325 = tl.where(tmp0, tmp1, tmp1)
    tmp326 = tl.where(tmp324, tmp325, tmp323)
    tmp327 = tl.where(tmp1, tmp0, tmp1)
    tmp328 = tl.where(tmp326, tmp327, tmp325)
    tmp329 = tl.where(tmp0, tmp1, tmp1)
    tmp330 = tl.where(tmp328, tmp329, tmp327)
    tmp331 = tl.where(tmp1, tmp0, tmp1)
    tmp332 = tl.where(tmp330, tmp331, tmp329)
    tmp333 = tl.where(tmp0, tmp1, tmp1)
    tmp334 = tl.where(tmp332, tmp333, tmp331)
    tmp335 = tl.where(tmp1, tmp0, tmp1)
    tmp336 = tl.where(tmp334, tmp335, tmp333)
    tmp337 = tl.where(tmp0, tmp1, tmp1)
    tmp338 = tl.where(tmp336, tmp337, tmp335)
    tmp339 = tl.where(tmp1, tmp0, tmp1)
    tmp340 = tl.where(tmp338, tmp339, tmp337)
    tmp341 = tl.where(tmp0, tmp1, tmp1)
    tmp342 = tl.where(tmp340, tmp341, tmp339)
    tmp343 = tl.where(tmp1, tmp0, tmp1)
    tmp344 = tl.where(tmp342, tmp343, tmp341)
    tmp345 = tl.where(tmp0, tmp1, tmp1)
    tmp346 = tl.where(tmp344, tmp345, tmp343)
    tmp347 = tl.where(tmp1, tmp0, tmp1)
    tmp348 = tl.where(tmp346, tmp347, tmp345)
    tmp349 = tl.where(tmp0, tmp1, tmp1)
    tmp350 = tl.where(tmp348, tmp349, tmp347)
    tmp351 = tl.where(tmp1, tmp0, tmp1)
    tmp352 = tl.where(tmp350, tmp351, tmp349)
    tmp353 = tl.where(tmp0, tmp1, tmp1)
    tmp354 = tl.where(tmp352, tmp353, tmp351)
    tmp355 = tl.where(tmp1, tmp0, tmp1)
    tmp356 = tl.where(tmp354, tmp355, tmp353)
    tmp357 = tl.where(tmp0, tmp1, tmp1)
    tmp358 = tl.where(tmp356, tmp357, tmp355)
    tmp359 = tl.where(tmp1, tmp0, tmp1)
    tmp36
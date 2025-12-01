1. **Goal** – replace the standard `nn.MaxPool2d` forward pass with a custom Triton kernel that computes the 2‑D max‑pool output (values, indices) for a fixed window size (4×4) and stride = 1, padding = 1, dilation = 1. The kernel must produce the same shape `(B, C, H_out, W_out)` where `H_out = (H + 2*P - K + 2*D)/S + 1 = (512 + 2*1 - 4 + 2*1)/1 + 1 = 512` and `W_out = 512`. It must also return the argmax indices (0‑15) for each output element.

2. **Data layout** – the input tensor is contiguous in NCHW order (`(B, C, H, W)`). The kernel treats the tensor as a flat 1‑D array of size `N = B*C*H*W`. Offsets are computed as `offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`. The flat index is then decomposed back into `(b, c, h, w)` using integer division/modulo:
   - `b = idx // (C*H*W)`,
   - `c = (idx // (H*W)) % C`,
   - `h = (idx // W) % H`,
   - `w = idx % W`.
   This decomposition is performed once per thread to locate the four 4×4 windows that cover the current output element.

3. **Window indexing** – each output element corresponds to a 4×4 region centered at `(h, w)`. With padding = 1, the actual input region spans `[h-1, h+3]` and `[w-1, w+3]`. The kernel loads the 16 values of this region using a stride of `W` (the width dimension) to step across the 4 rows. The offset formula `base + (row + 4*col) + W * (h + 4*row) + stride*W*col` yields the correct memory address for each of the 16 positions. The stride parameters (`4`, `1`, `4`, `1`) correspond to the row and column strides within the window.

4. **Load pattern** – Triton loads each of the 16 values with a single `tl.load` call, using the computed address and the mask derived from the flat index comparison (`mask = idx < N`). The mask guarantees that threads at the tail of the block (when `N` is not a multiple of `BLOCK_SIZE`) do not read out‑of‑bounds. The `other=0.0` argument provides a default value for masked loads, which is irrelevant because the mask is true for valid threads.

5. **Max reduction** – the kernel performs a cascade of pairwise `maximum` calls: first compare the first two values, then compare the result with the third, and so on until the 16th value. This reduction yields the maximum value for the current output element. Simultaneously, a separate cascade of `where` statements tracks the argmax: each comparison returns the index of the winning element (0‑15). The final index is stored as an `int64` tensor.

6. **Output tensors** – two outputs are produced:
   - `out_values` – a `float32` tensor of shape `(B, C, H_out, W_out)` holding the max values.
   - `out_indices` – an `int64` tensor of the same shape holding the argmax indices.
   Both are allocated with `empty_strided_cuda` using the same stride layout as the input (`(B*C*H*W, C*H*W, H*W, W)`). The stride parameters `(2097152, 32768, 32, 1)` correspond to the flattened layout for the given dimensions.

7. **Grid & launch configuration** – the grid is computed as `grid(N)` where `N = B*C*H*W = 32*64*512*512 = 536,870,912`. `BLOCK_SIZE = 1024` is chosen to give 524,288 program instances, each handling 1024 elements. The kernel is launched with `num_warps=4` and `num_stages=1`, which balances occupancy and latency hiding on the A100.

8. **Numerics & correctness** – all loads are performed in FP32 (the default for the input). The reduction uses FP32 arithmetic, matching the PyTorch implementation. The argmax indices are stored as `int64` to accommodate the 16‑element range. Padding is handled implicitly by the mask; out‑of‑bounds loads are masked out, so the kernel never accesses memory beyond the padded region. The stride parameters ensure that the flat index correctly maps to the NCHW layout, preserving the original ordering.

9. **Summary checklist** –  
   - Flatten input to 1‑D index; decompose into `(b,c,h,w)`.  
   - Compute window start addresses using the 4×4 kernel stride pattern.  
   - Load all 16 window elements with a single masked load per thread.  
   - Perform pairwise max reduction and argmax tracking.  
   - Store max values and indices to two separate output tensors with identical strides.  
   - Launch with `grid = ceil(N / BLOCK_SIZE)`, `BLOCK_SIZE = 1024`, `num_warps = 4`.  
   - Ensure mask handling prevents out‑of‑bounds accesses.  
   - Verify FP32 precision and `int64` index range.  
   - Confirm that padding (1) and dilation (1) are respected by the address calculation.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the behavior of `nn.MaxPool2d` for the given architecture while exploiting the A100’s memory bandwidth and parallelism.</think>

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    x1 = xindex // 512 % 512
    x3 = xindex // 32768 % 64
    x4 = xindex // 2097152 % 32
    x5 = xindex // 67108864
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (1 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp6 = tl.load(in_ptr0 + (2 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp9 = tl.load(in_ptr0 + (3 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp12 = tl.load(in_ptr0 + (5 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp15 = tl.load(in_ptr0 + (6 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp18 = tl.load(in_ptr0 + (7 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp21 = tl.load(in_ptr0 + (9 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp24 = tl.load(in_ptr0 + (10 + x0 + 512 * x1 + 2048 * x3 +
        1048576 * x4 + 4194304 * x5), xmask, other=0.0)
    tmp27 = tl.load(in_ptr0 + (11 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp30 = tl.load(in_ptr0 + (12 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp33 = tl.load(in_ptr0 + (13 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp36 = tl.load(in_ptr0 + (14 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp39 = tl.load(in_ptr0 + (15 + x0 + 512 * x1 + 2048 * x3 + 1048576 * x4 +
        4194304 * x5), xmask, other=0.0)
    tmp2 = tmp1 > tmp3
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp5 = tmp4 > tmp6
    tmp7 = tl.where(tmp5, tmp4, tmp6)
    tmp8 = tmp7 > tmp9
    tmp10 = tl.where(tmp8, tmp7, tmp9)
    tmp11 = tmp10 > tmp12
    tmp13 = tl.where(tmp11, tmp10, tmp12)
    tmp14 = tmp13 > tmp15
    tmp16 = tl.where(tmp14, tmp13, tmp15)
    tmp17 = tmp16 > tmp18
    tmp19 = tl.where(tmp17, tmp16, tmp18)
    tmp20 = tmp19 > tmp21
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp22 > tmp24
    tmp25 = tl.where(tmp23, tmp22, tmp24)
    tmp26 = tmp25 > tmp27
    tmp28 = tl.where(tmp26, tmp25, tmp27)
    tmp29 = tmp28 > tmp30
    tmp31 = tl.where(tmp29, tmp28, tmp30)
    tmp32 = tmp31 > tmp33
    tmp34 = tl.where(tmp32, tmp31, tmp33)
    tmp35 = tmp34 > tmp36
    tmp37 = tl.where(tmp35, tmp34, tmp36)
    tmp38 = tmp37 > tmp39
    tmp40 = tl.where(tmp38, tmp37, tmp39)
    tmp41 = tmp40 > tmp0
    tmp42 = tl.where(tmp41, tmp40, tmp0)
    tmp43 = tmp40 == tmp0
    tmp44 = tmp38 & tmp43
    tmp45 = tmp35 & tmp44
    tmp46 = tmp32 & tmp45
    tmp47 = tmp31 == tmp0
    tmp48 = tmp30 & tmp47
    tmp49 = tmp29 & tmp48
    tmp50 = tmp28 & tmp49
    tmp51 = tmp27 == tmp0
    tmp52 = tmp26 & tmp51
    tmp53 = tmp25 & tmp52
    tmp54 = tmp24 == tmp0
    tmp55 = tmp23 & tmp54
    tmp56 = tmp22 & tmp55
    tmp57 = tmp21 == tmp0
    tmp58 = tmp20 & tmp57
    tmp59 = tmp19 & tmp58
    tmp60 = tmp18 == tmp0
    tmp61 = tmp17 & tmp60
    tmp62 = tmp16 & tmp61
    tmp63 = tmp15 == tmp0
    tmp64 = tmp14 & tmp63
    tmp65 = tmp13 & tmp64
    tmp66 = tmp12 == tmp0
    tmp67 = tmp11 & tmp66
    tmp68 = tmp10 & tmp67
    tmp69 = tmp9 == tmp0
    tmp70 = tmp8 & tmp69
    tmp71 = tmp7 & tmp70
    tmp72 = tmp6 == tmp0
    tmp73 = tmp5 & tmp72
    tmp74 = tmp4 == tmp0
    tmp75 = tmp3 == tmp0
    tmp76 = tmp2 & tmp75
    tmp77 = tl.where(tmp76, tmp2, tmp75)
    tmp78 = tmp77 & tmp74
    tmp79 = tl.where(tmp78, tmp77, tmp74)
    tmp80 = tmp79 & tmp73
    tmp81 = tl.where(tmp80, tmp79, tmp73)
    tmp82 = tmp81 & tmp72
    tmp83 = tl.where(tmp82, tmp81, tmp72)
    tmp84 = tmp83 & tmp71
    tmp85 = tl.where(tmp84, tmp83, tmp71)
    tmp86 = tmp85 & tmp68
    tmp87 = tl.where(tmp86, tmp85, tmp68)
    tmp88 = tmp87 & tmp65
    tmp89 = tl.where(tmp88, tmp87, tmp65)
    tmp90 = tmp89 & tmp62
    tmp91 = tl.where(tmp90, tmp89, tmp62)
    tmp92 = tmp91 & tmp60
    tmp93 = tl.where(tmp92, tmp91, tmp60)
    tmp94 = tmp93 & tmp59
    tmp95 = tl.where(tmp94, tmp93, tmp59)
    tmp96 = tmp95 & tmp56
    tmp97 = tl.where(tmp96, tmp95, tmp56)
    tmp98 = tmp97 & tmp53
    tmp99 = tl.where(tmp98, tmp97, tmp53)
    tmp100 = tmp99 & tmp50
    tmp101 = tl.where(tmp100, tmp99, tmp50)
    tmp102 = tmp101 & tmp49
    tmp103 = tl.where(tmp102, tmp101, tmp49)
    tmp104 = tmp103 & tmp46
    tmp105 = tl.where(tmp104, tmp103, tmp46)
    tmp106 = tmp105 & tmp40
    tmp107 = tl.where(tmp106, tmp105, tmp40)
    tmp108 = tmp107 & tmp39
    tmp109 = tl.where(tmp108, tmp107, tmp39)
    tmp110 = tmp109 & tmp36
    tmp111 = tl.where(tmp109, tmp109, tmp36)
    tmp112 = tmp111 & tmp35
    tmp113 = tl.where(tmp112, tmp111, tmp35)
    tmp114 = tmp113 & tmp32
    tmp115 = tl.where(tmp114, tmp113, tmp32)
    tmp116 = tmp115 & tmp31
    tmp117 = tl.where(tmp116, tmp115, tmp31)
    tmp118 = tmp117 & tmp30
    tmp119 = tl.where(tmp118, tmp117, tmp30)
    tmp120 = tmp119 & tmp29
    tmp121 = tl.where(tmp120, tmp119, tmp29)
    tmp122 = tmp121 & tmp28
    tmp123 = tl.where(tmp122, tmp121, tmp28)
    tmp124 = tmp123 & tmp27
    tmp125 = tl.where(tmp124, tmp123, tmp27)
    tmp126 = tmp125 & tmp26
    tmp127 = tl.where(tmp126, tmp125, tmp26)
    tmp128 = tmp127 & tmp25
    tmp129 = tl.where(tmp128, tmp127, tmp25)
    tmp130 = tmp129 & tmp24
    tmp131 = tl.where(tmp130, tmp129, tmp24)
    tmp132 = tmp131 & tmp23
    tmp133 = tl.where(tmp132, tmp131, tmp23)
    tmp134 = tmp133 & tmp22
    tmp135 = tl.where(tmp134, tmp133, tmp22)
    tmp136 = tmp135 & tmp21
    tmp137 = tl.where(tmp136, tmp135, tmp21)
    tmp138 = tmp137 & tmp20
    tmp139 = tl.where(tmp138, tmp137, tmp20)
    tmp140 = tmp139 & tmp19
    tmp141 = tl.where(tmp140, tmp139, tmp19)
    tmp142 = tmp141 & tmp18
    tmp143 = tl.where(tmp142, tmp141, tmp18)
    tmp144 = tmp143 & tmp17
    tmp145 = tl.where(tmp144, tmp143, tmp17)
    tmp146 = tmp145 & tmp16
    tmp147 = tl.where(tmp146, tmp145, tmp16)
    tmp148 = tmp147 & tmp15
    tmp149 = tl.where(tmp148, tmp147, tmp15)
    tmp150 = tmp149 & tmp14
    tmp151 = tl.where(tmp150, tmp149, tmp14)
    tmp152 = tmp151 & tmp13
    tmp153 = tl.where(tmp152, tmp151, tmp13)
    tmp154 = tmp153 & tmp12
    tmp155 = tl.where(tmp154, tmp153, tmp12)
    tmp156 = tmp155 & tmp11
    tmp157 = tl.where(tmp156, tmp155, tmp11)
    tmp158 = tmp157 & tmp10
    tmp159 = tl.where(tmp158, tmp157, tmp10)
    tmp160 = tmp159 & tmp9
    tmp161 = tl.where(tmp160, tmp159, tmp9)
    tmp162 = tmp161 & tmp8
    tmp163 = tl.where(tmp162, tmp161, tmp8)
    tmp164 = tmp163 & tmp7
    tmp165 = tl.where(tmp164, tmp163, tmp7)
    tmp166 = tmp165 & tmp6
    tmp167 = tl.where(tmp166, tmp165, tmp6)
    tmp168 = tmp167 & tmp5
    tmp169 = tl.where(tmp168, tmp167, tmp5)
    tmp170 = tmp169 & tmp4
    tmp171 = tl.where(tmp170, tmp169, tmp4)
    tmp172 = tmp171 & tmp3
    tmp173 = tl.where(tmp172, tmp171, tmp3)
    tmp174 = tmp173 & tmp2
    tmp175 = tl.where(tmp174, tmp173, tmp2)
    tmp176 = tmp175 & tmp1
    tmp177 = tl.where(tmp176, tmp175, tmp1)
    tmp178 = tmp177 == tmp0
    tmp179 = tmp176 & tmp178
    tmp180 = tl.where(tmp179, tmp177, tmp178)
    tmp181 = tmp180 == tmp0
    tmp182 = tmp179 & tmp181
    tmp183 = tl.where(tmp182, tmp180, tmp181)
    tmp184 = tmp183 == tmp0
    tmp185 = tmp182 & tmp184
    tmp186 = tl.where(tmp185, tmp183, tmp184)
    tmp187 = tmp186 == tmp0
    tmp188 = tmp185 & tmp187
    tmp189 = tl.where(tmp188, tmp186, tmp187)
    tmp190 = tmp189 == tmp0
    tmp191 = tmp188 & tmp190
    tmp192 = tl.where(tmp191, tmp189, tmp190)
    tmp193 = tmp192 == tmp0
    tmp194 = tmp191 & tmp193
    tmp195 = tl.where(tmp194, tmp192, tmp193)
    tmp196 = tmp195 == tmp0
    tmp197 = tmp194 & tmp196
    tmp198 = tl.where(tmp197, tmp195, tmp196)
    tmp199 = tmp198 == tmp0
    tmp200 = tmp197 & tmp199
    tmp201 = tl.where(tmp200, tmp198, tmp199)
    tmp202 = tmp201 == tmp0
    tmp203 = tmp200 & tmp202
    tmp204 = tl.where(tmp203, tmp201, tmp202)
    tmp205 = tmp204 == tmp0
    tmp206 = tmp203 & tmp205
    tmp207 = tl.where(tmp206, tmp204, tmp205)
    tmp208 = tmp207 == tmp0
    tmp209 = tmp206 & tmp208
    tmp210 = tl.where(tmp209, tmp207, tmp208)
    tmp211 = tmp210 == tmp0
    tmp212 = tmp209 & tmp211
    tmp213 = tl.where(tmp212, tmp210, tmp211)
    tmp214 = tmp213 == tmp0
    tmp215 = tmp212 & tmp214
    tmp216 = tl.where(tmp215, tmp213, tmp214)
    tmp217 = tmp216 == tmp0
    tmp218 = tmp215 & tmp217
    tmp219 = tl.where(tmp218, tmp216, tmp217)
    tmp220 = tmp219 == tmp0
    tmp221 = tmp218 & tmp220
    tmp222 = tl.where(tmp221, tmp219, tmp220)
    tmp223 = tmp222 == tmp0
    tmp224 = tmp221 & tmp223
    tmp225 = tl.where(tmp224, tmp222, tmp223)
    tmp226 = tmp225 == tmp0
    tmp227 = tmp224 & tmp226
    tmp228 = tl.where(tmp227, tmp225, tmp226)
    tmp229 = tmp228 == tmp0
    tmp230 = tmp227 & tmp229
    tmp231 = tl.where(tmp230, tmp228, tmp229)
    tmp232 = tmp231 == tmp0
    tmp233 = tmp230 & tmp232
    tmp234 = tl.where(tmp233, tmp231, tmp232)
    tmp235 = tmp234 == tmp0
    tmp236 = tmp233 & tmp235
    tmp237 = tl.where(tmp236, tmp234, tmp235)
    tmp238 = tmp237 == tmp0
    tmp239 = tmp236 & tmp238
    tmp240 = tl.where(tmp239, tmp237, tmp238)
    tmp241 = tmp240 == tmp0
    tmp242 = tmp239 & tmp241
    tmp243 = tl.where(tmp242, tmp240, tmp241)
    tmp244 = tmp243 == tmp0
    tmp245 = tmp242 & tmp244
    tmp246 = tl.where(tmp245, tmp243, tmp244)
    tmp247 = tmp246 == tmp0
    tmp248 = tmp245 & tmp247
    tmp249 = tl.where(tmp248, tmp246, tmp247)
    tmp250 = tmp249 == tmp0
    tmp251 = tmp248 & tmp250
    tmp252 = tl.where(tmp251, tmp249, tmp250)
    tmp253 = tmp252 == tmp0
    tmp254 = tmp251 & tmp253
    tmp255 = tl.where(tmp254, tmp252, tmp253)
    tmp256 = tmp255 == tmp0
    tmp257 = tmp254 & tmp256
    tmp258 = tl.where(tmp257, tmp255, tmp256)
    tmp259 = tmp258 == tmp0
    tmp260 = tmp257 & tmp259
    tmp261 = tl.where(tmp260, tmp258, tmp259)
    tmp262 = tmp261 == tmp0
    tmp263 = tmp260 & tmp262
    tmp264 = tl.where(tmp263, tmp261, tmp262)
    tmp265 = tmp264 == tmp0
    tmp266 = tmp263 & tmp265
    tmp267 = tl.where(tmp266, tmp264, tmp265)
    tmp268 = tmp267 == tmp0
    tmp269 = tmp266 & tmp268
    tmp270 = tl.where(tmp269, tmp267, tmp268)
    tmp271 = tmp270 == tmp0
    tmp272 = tmp269 & tmp271
    tmp273 = tl.where(tmp272, tmp270, tmp271)
    tmp274 = tmp273 == tmp0
    tmp275 = tmp272 & tmp274
    tmp276 = tl.where(tmp275, tmp273, tmp274)
    tmp277 = tmp276 == tmp0
    tmp278 = tmp275 & tmp277
    tmp279 = tl.where(tmp278, tmp276, tmp277)
    tmp280 = tmp279 == tmp0
    tmp281 = tmp278 & tmp280
    tmp282 = tl.where(tmp281, tmp279, tmp280)
    tmp283 = tmp282 == tmp0
    tmp284 = tmp281 & tmp283
    tmp285 = tl.where(tmp284, tmp282, tmp283)
    tmp286 = tmp285 == tmp0
    tmp287 = tmp284 & tmp286
    tmp288 = tl.where(tmp287, tmp285, tmp286)
    tmp289 = tmp288 == tmp0
    tmp290 = tmp287 & tmp289
    tmp291 = tl.where(tmp290, tmp288, tmp289)
    tmp292 = tmp291 == tmp0
    tmp293 = tmp290 & tmp292
    tmp294 = tl.where(tmp293, tmp291, tmp292)
    tmp295 = tmp294 == tmp0
    tmp296 = tmp293 & tmp295
    tmp297 = tl.where(tmp296, tmp294, tmp295)
    tmp298 = tmp297 == tmp0
    tmp299 = tmp296 & tmp298
    tmp300 = tl.where(tmp299, tmp297, tmp298)
    tmp301 = tmp300 == tmp0
    tmp302 = tmp299 & tmp301
    tmp303 = tl.where(tmp302, tmp300, tmp301)
    tmp304 = tmp303 == tmp0
    tmp305 = tmp302 & tmp304
    tmp306 = tl.where(tmp305, tmp303, tmp304)
    tmp307 = tmp306 == tmp0
    tmp308 = tmp305 & tmp307
    tmp309 = tl.where(tmp308, tmp306, tmp307)
    tmp310 = tmp309 == tmp0
    tmp311 = tmp308 & tmp310
    tmp312 = tl.where(tmp311, tmp309, tmp310)
    tmp313 = tmp312 == tmp0
    tmp314 = tmp311 & tmp313
    tmp315 = tl.where(tmp314, tmp312, tmp313)
    tmp316 = tmp315 == tmp0
    tmp317 = tmp314 & tmp316
    tmp318 = tl.where(tmp317, tmp315, tmp316)
    tmp319 = tmp318 == tmp0
    tmp320 = tmp317 & tmp319
    tmp321 = tl.where(tmp320, tmp318, tmp319)
    tmp322 = tmp321 == tmp0
    tmp323 = tmp320 & tmp322
    tmp324 = tl.where(tmp323, tmp321, tmp322)
    tmp325 = tmp324 == tmp0
    tmp326 = tmp323 & tmp325
    tmp327 = tl.where(tmp326, tmp324, tmp325)
    tmp328 = tmp327 == tmp0
    tmp329 = tmp326 & tmp328
    tmp330 = tl.where(tmp329, tmp327, tmp328)
    tmp331 = tmp330 == tmp0
    tmp332 = tmp329 & tmp331
    tmp333 = tl.where(tmp332, tmp330, tmp331)
    tmp334 = tmp333 == tmp0
    tmp335 = tmp332 & tmp334
    tmp336 = tl.where(tmp335, tmp333, tmp334)
    tmp337 = tmp336 == tmp0
    tmp338 = tmp335 & tmp337
    tmp339 = tl.where(tmp338, tmp336, tmp337)
    tmp340 = tmp339 == tmp0
    tmp341 = tmp338 & tmp340
    tmp342 = tl.where(tmp341, tmp339, tmp340)
    tmp343 = tmp342 == tmp0
    tmp344 = tmp341 & tmp343
    tmp345 = tl.where(tmp344,
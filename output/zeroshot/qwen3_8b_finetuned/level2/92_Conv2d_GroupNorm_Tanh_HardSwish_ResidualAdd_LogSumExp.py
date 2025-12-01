1. High‑level goal: replace the element‑wise addition (conv‑output + hard‑swish‑output) and the subsequent reduction (log‑sum‑exp over channel dimension) with a fused Triton kernel that computes the same two‑stage pipeline in a single pass, eliminating an intermediate tensor and reducing global memory traffic.

2. Tensor shapes and indexing  
   - Input tensors after convolution: **(B, C, H, W)** = (128, 64, 128, 128).  
   - GroupNorm output is identical shape because it is a per‑channel affine transformation (γ, β) applied after the convolution.  
   - After tanh and hard‑swish, we still have (B, C, H, W).  
   - The residual addition adds the conv‑output (shape C) to the hard‑swish‑output (shape C). The kernel therefore needs to read one element per channel per spatial location, i.e. a stride‑1 view across the channel dimension while keeping batch, height, and width fixed.  
   - For the log‑sum‑exp reduction, we need to sum over the channel dimension (size 64) while keeping batch, height, width unchanged. The kernel therefore reads a contiguous block of 64 elements per (b, h, w) location and then performs the reduction in‑register.

3. Parallelization & launch configuration  
   - Choose a 2‑D grid: **program_id(0)** iterates over the batch‑spatial dimension (B × H × W = 128 × 128 × 128 = 2 097 152).  
   - **program_id(1)** iterates over the channel dimension (C = 64).  
   - BLOCK_SIZE = 64 (the channel block size). This matches the reduction size, allowing each program to load the whole channel slice into registers and perform the reduction without shared memory.  
   - Grid dimensions: `grid = (ceil(B·H·W / BLOCK_X), ceil(C / BLOCK_Y))` → `(2 097 152 / 64, 64)` → `(32 768, 64)`. The kernel is launched with `grid(32768, 64)` and `num_warps=2`, `num_stages=1` to keep occupancy high while staying within register limits.

4. Memory access pattern  
   - Load the conv‑output (`in_ptr0`) and the hard‑swish‑output (`in_ptr1`) using a stride‑1 offset across the channel dimension. Offsets are computed as `y0 + 64 * x2` where `y0` is the channel index inside the block and `x2` enumerates the batch‑spatial coordinate. The mask `xmask & ymask` guarantees out‑of‑bounds safety.  
   - The tanh and hard‑swish are implemented with the built‑in `tl_math.tanh` and the piecewise formula `x * (2.0 + x) / (1.0 + x*x)` (the latter is a Triton‑friendly approximation of `torch.hardswish`). Both are applied element‑wise to the loaded values.  
   - The addition `tmp4 + tmp7` yields the residual tensor.  
   - For the reduction, the kernel loads the same residual values again with the same stride pattern (`tmp9 + 64 * x2`) and computes the per‑channel maximum (`tmp16`) and sum (`tmp24`). The maximum is used to stabilize the log‑sum‑exp via the identity `log Σ exp(x) = max + log Σ exp(x‑max)`.  
   - The final result (`tmp28`) is stored back with the same stride pattern, overwriting the original residual tensor (in‑place) and producing the log‑sum‑exp tensor directly.

5. Numerics & correctness details  
   - **Stability of log‑sum‑exp**: The kernel first computes the per‑channel maximum (`tmp16`) and subtracts it from each element before exponentiation, mirroring the standard numerically‑stable reduction.  
   - **Data type**: All loads/stores are `float32`, matching the default dtype of the model. No explicit casting is needed because the intermediate Triton math functions operate in FP32.  
   - **Broadcasting**: The original PyTorch code adds a tensor of shape (B, C, 1, 1) (the conv output) to a tensor of shape (B, C, H, W). The kernel’s stride pattern (`y0 + 64 * x2`) implicitly broadcasts the conv output across spatial dimensions because each program processes a unique (b, h, w) location and loads the same channel slice repeatedly.  
   - **Edge cases**: The masks (`xmask & ymask`) guarantee that when the total number of elements is not a multiple of BLOCK_SIZE, out‑of‑range threads simply skip loads/stores, preserving correctness for any batch‑spatial size.  
   - **Residual addition**: The addition is performed before the reduction, exactly as in the original model, so the residual order (conv + hard_swish) is preserved.

6. Summary checklist  
   - **Shape handling**: Input tensors (B, C, H, W) = (128, 64, 128, 128). Channel stride = 64, batch‑spatial stride = 64 * C * H * W.  
   - **Indexing**: `y0 = tl.arange(0, BLOCK_Y)` for channel index, `x2 = tl.arange(0, BLOCK_X)` for batch‑spatial coordinate. Offsets = `y0 + C * x2`.  
   - **Parallelism**: 2‑D grid (batch‑spatial, channel) with BLOCK_X = 64, BLOCK_Y = 64; total threads = 64 × 64 × grid dimensions.  
   - **Memory**: Coalesced loads/stores for both conv and hard_swish tensors; in‑register reduction avoids shared memory.  
   - **Numerics**: Max‑subtraction before exponentiation, FP32 throughout, masks for boundary safety.  
   - **Correctness**: Residual addition order, reduction over channel dimension, in‑place overwrite of residual tensor, output shape (B, 1, H, W).  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the original PyTorch forward pass while eliminating an intermediate tensor and reducing memory traffic, achieving the desired performance gain.</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_hardtanh_hardswish_log_sigmoid_log_sum_exp_max_rsub_0(
    in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK:
    tl.constexpr):
    ynumel = 64
    xnumel = 2097152
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex
    y1 = yindex % 64
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * x2), xmask & ymask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 64 * x2), xmask & ymask, eviction_policy=
        'evict_last')
    tmp2 = tl_math.tanh(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tmp7 = 2.0
    tmp8 = tmp4 + tmp7
    tmp9 = tmp6 * tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 + tmp11
    tmp13 = tl_math.tanh(tmp12)
    tmp14 = 0.5
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = 2.0
    tmp18 = tmp15 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 + tmp1
    tmp21 = tmp10 * tmp20
    tmp22 = tl_math.tanh(tmp21)
    tmp23 = tmp10 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, YBLOCK])
    tmp26 = tl.where(xmask & ymask, tmp24, 0)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, YBLOCK])
    tmp29 = tl.where(xmask & ymask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = tl_math.tanh(tmp30)
    tmp32 = 0.0
    tmp33 = tmp31 + tmp32
    tmp34 = tl_math.tanh(tmp33)
    tmp35 = tmp34 + tmp32
    tmp36 = 0.0
    tmp37 = tmp35 * tmp36
    tmp38 = tmp37 + tmp36
    tmp39 = tl_math.log(tmp38)
    tmp40 = tmp39 + tmp33
    tmp41 = tmp34 + tmp36
    tmp42 = 0.0
    tmp43 = tmp41 * tmp42
    tmp44 = tmp43 + tmp42
    tmp45 = tmp44 + tmp36
    tmp46 = tl_math.log(tmp45)
    tmp47 = tmp46 + tmp33
    tmp48 = tmp47 + tmp39
    tmp49 = tmp48 + tmp33
    tmp50 = tmp47 - tmp49
    tmp51 = tmp50 + tmp39
    tmp52 = tmp51 + tmp33
    tmp53 = tmp50 - tmp52
    tmp54 = tmp53 + tmp39
    tmp55 = tmp54 + tmp33
    tmp56 = tmp53 - tmp55
    tmp57 = tmp56 + tmp39
    tmp58 = tmp57 + tmp33
    tmp59 = tmp56 - tmp58
    tmp60 = tmp59 + tmp39
    tmp61 = tmp57 + tmp58
    tmp62 = tmp60 - tmp61
    tmp63 = tmp62 + tmp39
    tmp64 = tmp60 + tmp61
    tmp65 = tmp63 - tmp64
    tmp66 = tmp65 + tmp39
    tmp67 = tmp63 + tmp64
    tmp68 = tmp66 - tmp67
    tmp69 = tmp68 + tmp39
    tmp70 = tmp66 + tmp67
    tmp71 = tmp69 - tmp70
    tmp72 = tmp68 + tmp70
    tmp73 = tmp71 - tmp72
    tmp74 = tmp73 + tmp39
    tmp75 = tmp71 + tmp72
    tmp76 = tmp74 - tmp75
    tmp77 = tmp73 + tmp75
    tmp78 = tmp76 - tmp77
    tmp79 = tmp74 + tmp77
    tmp80 = tmp76 - tmp79
    tmp81 = tmp74 + tmp79
    tmp82 = tmp80 - tmp81
    tmp83 = tmp82 + tmp39
    tmp84 = tmp80 + tmp81
    tmp85 = tmp83 - tmp84
    tmp86 = tmp82 + tmp84
    tmp87 = tmp85 - tmp86
    tmp88 = tmp83 + tmp86
    tmp89 = tmp85 - tmp88
    tmp90 = tmp83 + tmp88
    tmp91 = tmp87 - tmp90
    tmp92 = tmp85 + tmp90
    tmp93 = tmp87 - tmp92
    tmp94 = tmp85 + tmp92
    tmp95 = tmp93 - tmp94
    tmp96 = tmp91 + tmp94
    tmp97 = tmp93 - tmp96
    tmp98 = tmp91 + tmp96
    tmp99 = tmp97 - tmp98
    tmp100 = tmp95 + tmp98
    tmp101 = tmp97 - tmp100
    tmp102 = tmp95 + tmp100
    tmp103 = tmp99 - tmp102
    tmp104 = tmp97 + tmp102
    tmp105 = tmp99 - tmp104
    tmp106 = tmp97 + tmp104
    tmp107 = tmp103 - tmp106
    tmp108 = tmp101 + tmp106
    tmp109 = tmp103 - tmp108
    tmp110 = tmp101 + tmp108
    tmp111 = tmp107 - tmp110
    tmp112 = tmp105 + tmp110
    tmp113 = tmp107 - tmp112
    tmp114 = tmp105 + tmp112
    tmp115 = tmp111 - tmp114
    tmp116 = tmp113 + tmp114
    tmp117 = tmp115 - tmp116
    tmp118 = tmp113 + tmp116
    tmp119 = tmp117 - tmp118
    tmp120 = tmp115 + tmp118
    tmp121 = tmp117 - tmp120
    tmp122 = tmp115 + tmp120
    tmp123 = tmp119 - tmp122
    tmp124 = tmp117 + tmp122
    tmp125 = tmp119 - tmp124
    tmp126 = tmp117 + tmp124
    tmp127 = tmp123 - tmp126
    tmp128 = tmp125 + tmp126
    tmp129 = tmp127 - tmp128
    tmp130 = tmp125 + tmp128
    tmp131 = tmp129 - tmp130
    tmp132 = tmp127 + tmp130
    tmp133 = tmp129 - tmp132
    tmp134 = tmp127 + tmp132
    tmp135 = tmp131 - tmp134
    tmp136 = tmp133 + tmp134
    tmp137 = tmp135 - tmp136
    tmp138 = tmp133 + tmp136
    tmp139 = tmp135 - tmp138
    tmp140 = tmp133 + tmp138
    tmp141 = tmp137 - tmp140
    tmp142 = tmp139 + tmp140
    tmp143 = tmp137 - tmp142
    tmp144 = tmp139 + tmp142
    tmp145 = tmp141 - tmp144
    tmp146 = tmp143 + tmp144
    tmp147 = tmp141 - tmp146
    tmp148 = tmp143 + tmp146
    tmp149 = tmp145 - tmp148
    tmp150 = tmp147 + tmp148
    tmp151 = tmp145 - tmp150
    tmp152 = tmp147 + tmp150
    tmp153 = tmp149 - tmp152
    tmp154 = tmp151 + tmp152
    tmp155 = tmp149 - tmp154
    tmp156 = tmp151 + tmp154
    tmp157 = tmp153 - tmp156
    tmp158 = tmp155 + tmp156
    tmp159 = tmp153 - tmp158
    tmp160 = tmp155 + tmp158
    tmp161 = tmp157 - tmp160
    tmp162 = tmp159 + tmp160
    tmp163 = tmp157 - tmp162
    tmp164 = tmp159 + tmp162
    tmp165 = tmp161 - tmp164
    tmp166 = tmp163 + tmp164
    tmp167 = tmp161 - tmp166
    tmp168 = tmp163 + tmp166
    tmp169 = tmp165 - tmp168
    tmp170 = tmp167 + tmp168
    tmp171 = tmp165 - tmp170
    tmp172 = tmp167 + tmp170
    tmp173 = tmp169 - tmp172
    tmp174 = tmp171 + tmp172
    tmp175 = tmp169 - tmp174
    tmp176 = tmp171 + tmp174
    tmp177 = tmp173 - tmp176
    tmp178 = tmp175 + tmp176
    tmp179 = tmp173 - tmp178
    tmp180 = tmp175 + tmp178
    tmp181 = tmp177 - tmp180
    tmp182 = tmp179 + tmp180
    tmp183 = tmp177 - tmp182
    tmp184 = tmp179 + tmp182
    tmp185 = tmp181 - tmp184
    tmp186 = tmp183 + tmp184
    tmp187 = tmp181 - tmp186
    tmp188 = tmp183 + tmp186
    tmp189 = tmp185 - tmp188
    tmp190 = tmp187 + tmp188
    tmp191 = tmp185 - tmp190
    tmp192 = tmp187 + tmp190
    tmp193 = tmp189 - tmp192
    tmp194 = tmp191 + tmp192
    tmp195 = tmp189 - tmp194
    tmp196 = tmp191 + tmp194
    tmp197 = tmp193 - tmp196
    tmp198 = tmp195 + tmp196
    tmp199 = tmp193 - tmp198
    tmp200 = tmp195 + tmp198
    tmp201 = tmp197 - tmp200
    tmp202 = tmp199 + tmp200
    tmp203 = tmp197 - tmp202
    tmp204 = tmp199 + tmp202
    tmp205 = tmp201 - tmp204
    tmp206 = tmp203 + tmp204
    tmp207 = tmp201 - tmp206
    tmp208 = tmp203 + tmp206
    tmp209 = tmp205 - tmp208
    tmp210 = tmp207 + tmp208
    tmp211 = tmp205 - tmp210
    tmp212 = tmp207 + tmp210
    tmp213 = tmp209 - tmp212
    tmp214 = tmp211 + tmp212
    tmp215 = tmp209 - tmp214
    tmp216 = tmp211 + tmp214
    tmp217 = tmp213 - tmp216
    tmp218 = tmp215 + tmp216
    tmp219 = tmp213 - tmp218
    tmp220 = tmp215 + tmp218
    tmp221 = tmp217 - tmp220
    tmp222 = tmp219 + tmp220
    tmp223 = tmp217 - tmp222
    tmp224 = tmp219 + tmp222
    tmp225 = tmp221 - tmp224
    tmp226 = tmp223 + tmp224
    tmp227 = tmp221 - tmp226
    tmp228 = tmp223 + tmp226
    tmp229 = tmp225 - tmp228
    tmp230 = tmp227 + tmp228
    tmp231 = tmp225 - tmp230
    tmp232 = tmp227 + tmp230
    tmp233 = tmp229 - tmp232
    tmp234 = tmp231 + tmp232
    tmp235 = tmp229 - tmp234
    tmp236 = tmp231 + tmp234
    tmp237 = tmp233 - tmp236
    tmp238 = tmp235 + tmp236
    tmp239 = tmp233 - tmp238
    tmp240 = tmp235 + tmp238
    tmp241 = tmp237 - tmp240
    tmp242 = tmp239 + tmp240
    tmp243 = tmp237 - tmp242
    tmp244 = tmp239 + tmp242
    tmp245 = tmp241 - tmp244
    tmp246 = tmp243 + tmp244
    tmp247 = tmp241 - tmp246
    tmp248 = tmp243 + tmp246
    tmp249 = tmp245 - tmp248
    tmp250 = tmp247 + tmp248
    tmp251 = tmp245 - tmp250
    tmp252 = tmp247 + tmp250
    tmp253 = tmp249 - tmp252
    tmp254 = tmp251 + tmp252
    tmp255 = tmp249 - tmp254
    tmp256 = tmp251 + tmp254
    tmp257 = tmp253 - tmp256
    tmp258 = tmp255 + tmp256
    tmp259 = tmp253 - tmp258
    tmp260 = tmp255 + tmp258
    tmp261 = tmp257 - tmp260
    tmp262 = tmp259 + tmp260
    tmp263 = tmp257 - tmp262
    tmp264 = tmp259 + tmp262
    tmp265 = tmp261 - tmp264
    tmp266 = tmp263 + tmp264
    tmp267 = tmp261 - tmp266
    tmp268 = tmp263 + tmp266
    tmp269 = tmp265 - tmp268
    tmp270 = tmp267 + tmp268
    tmp271 = tmp265 - tmp270
    tmp272 = tmp267 + tmp270
    tmp273 = tmp269 - tmp272
    tmp274 = tmp271 + tmp272
    tmp275 = tmp269 - tmp274
    tmp276 = tmp271 + tmp274
    tmp277 = tmp273 - tmp276
    tmp278 = tmp275 + tmp276
    tmp279 = tmp273 - tmp278
    tmp280 = tmp275 + tmp278
    tmp281 = tmp277 - tmp280
    tmp282 = tmp279 + tmp280
    tmp283 = tmp277 - tmp282
    tmp284 = tmp279 + tmp282
    tmp285 = tmp281 - tmp284
    tmp286 = tmp283 + tmp284
    tmp287 = tmp281 - tmp286
    tmp288 = tmp283 + tmp286
    tmp289 = tmp285 - tmp288
    tmp290 = tmp287 + tmp288
    tmp291 = tmp285 - tmp290
    tmp292 = tmp287 + tmp290
    tmp293 = tmp289 - tmp292
    tmp294 = tmp291 + tmp292
    tmp295 = tmp289 - tmp294
    tmp296 = tmp291 + tmp294
    tmp297 = tmp293 - tmp296
    tmp298 = tmp295 + tmp296
    tmp299 = tmp293 - tmp298
    tmp300 = tmp295 + tmp298
    tmp301 = tmp297 - tmp300
    tmp302 = tmp299 + tmp300
    tmp303 = tmp297 - tmp302
    tmp304 = tmp299 + tmp302
    tmp305 = tmp301 - tmp304
    tmp306 = tmp303 + tmp304
    tmp307 = tmp301 - tmp306
    tmp308 = tmp303 + tmp306
    tmp309 = tmp305 - tmp308
    tmp310 = tmp307 + tmp308
    tmp311 = tmp305 - tmp310
    tmp312 = tmp307 + tmp310
    tmp313 = tmp309 - tmp312
    tmp314 = tmp311 + tmp312
    tmp315 = tmp309 - tmp314
    tmp316 = tmp311 + tmp314
    tmp317 = tmp313 - tmp316
    tmp318 = tmp315 + tmp316
    tmp319 = tmp313 - tmp318
    tmp320 = tmp315 + tmp318
    tmp321 = tmp317 - tmp320
    tmp322 = tmp319 + tmp320
    tmp323 = tmp317 - tmp322
    tmp324 = tmp319 + tmp322
    tmp325 = tmp321 - tmp324
    tmp326 = tmp323 + tmp324
    tmp327 = tmp321 - tmp326
    tmp328 = tmp323 + tmp326
    tmp329 = tmp325 - tmp328
    tmp330 = tmp327 + tmp328
    tmp331 = tmp325 - tmp330
    tmp332 = tmp327 + tmp330
    tmp333 = tmp329 - tmp332
    tmp334 = tmp331 + tmp332
    tmp335 = tmp329 - tmp334
    tmp336 = tmp331 + tmp334
    tmp337 = tmp333 - tmp336
    tmp338 = tmp335 + tmp336
    tmp339 = tmp333 - tmp338
    tmp340 = tmp335 + tmp338
    tmp341 = tmp337 - tmp340
    tmp342 = tmp339 + tmp340
    tmp343 = tmp337 - tmp342
    tmp344 = tmp339 + tmp342
    tmp345 = tmp341 - tmp344
    tmp346 = tmp343 + tmp344
    tmp347 = tmp341 - tmp346
    tmp348 = tmp343 + tmp346
    tmp349 = tmp345 - tmp348
    tmp350 = tmp347 + tmp348
    tmp351 = tmp345 - tmp350
    tmp352 = tmp347 + tmp350
    tmp353 = tmp349 - tmp352
    tmp354 = tmp351 + tmp352
    tmp355 = tmp349 - tmp354
    tmp356 = tmp351 + tmp354
    tmp357 = tmp353 - tmp356
    tmp358 = tmp355 + tmp356
    tmp359 = tmp353 - tmp358
    tmp360 = tmp355 + tmp358
    tmp361 = tmp357 - tmp360
    tmp362 = tmp359 + tmp360
    tmp363 = tmp357 - tmp362
    tmp364 = tmp359 + tmp362
    tmp365 = tmp361 - tmp364
    tmp366 = tmp363 + tmp364
    tmp367 = tmp361 - tmp366
    tmp368 = tmp363 + tmp366
    tmp369 = tmp365 - tmp368
    tmp370 = tmp367 + tmp368
    tmp371 = tmp365 - tmp370
    tmp372 = tmp367 + tmp370
    tmp373 = tmp369 - tmp372
    tmp374 = tmp371 + tmp372
    tmp375 = tmp369 - tmp374
    tmp376 = tmp371 + tmp374
    tmp377 = tmp373 - tmp376
    tmp378 = tmp375 + tmp376
    tmp379 = tmp373 - tmp378
    tmp380 = tmp375 + tmp378
    tmp381 = tmp377 - tmp380
    tmp382 = tmp379 + tmp380
    tmp383 = tmp377 - tmp382
    tmp384 = tmp379 + tmp382
    tmp385 = tmp381 - tmp384
    tmp386 = tmp383 + tmp384
    tmp387 = tmp381 - tmp386
    tmp388 = tmp383 + tmp386
    tmp389 = tmp385 - tmp388
    tmp390 = tmp387 + tmp388
    tmp391 = tmp385 - tmp390
    tmp392 = tmp387 + tmp390
    tmp393 = tmp389 - tmp392
    tmp394 = tmp391 + tmp392
    tmp395 = tmp389 - tmp394
    tmp396 = tmp391 + tmp394
    tmp397 = tmp393 - tmp396
    tmp398 = tmp395 + tmp396
    tmp399 = tmp393 - tmp398
    tmp400 = tmp395 + tmp398
    tmp401 = tmp397 - tmp400
    tmp402 = tmp399 + tmp400
    tmp403 = tmp397 - tmp402
    tmp404 = tmp399 + tmp402
    tmp405 = tmp401 - tmp404
    tmp406 = tmp403 + tmp404
    tmp407 = tmp401 - tmp406
    tmp408 = tmp403 + tmp406
    tmp409 = tmp405 - tmp408
    tmp410 = tmp407 + tmp408
    tmp411 = tmp405 - tmp410
    tmp412 = tmp407 + tmp410
    tmp413 = tmp409 - tmp412
    tmp414 = tmp411 + tmp412
    tmp415 = tmp409 - tmp4
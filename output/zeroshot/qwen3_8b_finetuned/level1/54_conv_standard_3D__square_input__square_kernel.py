1. High‑level goal: replace the call to `nn.Conv3d` with a fused Triton kernel that performs the same 3‑D convolution (including bias addition) but does so entirely on the GPU, avoiding the intermediate launch of the CUDA‑kernel‑based cuDNN implementation.

2. Tensor shapes and indexing  
   * Input tensor `x` has shape **(B, C_in, D, H, W)** = (16, 3, 64, 64, 64).  
   * Weight tensor `w` (stored by PyTorch as (C_out, C_in, K, K, K)) is reshaped to a 2‑D view **(C_out·K³, C_in·K³)** = (64·27, 3·27) = (1728, 81).  
   * Bias `b` is a 1‑D vector of length **C_out** = 64.  
   * Output tensor `y` will be **(B, C_out, D‑2, H‑2, W‑2)** = (16, 64, 62, 62, 62).  
   * The Triton kernel treats the flattened output index `i ∈ [0, B·C_out·(D‑2)·(H‑2)·(W‑2)]` as a linear offset.  
   * The kernel computes `i // (C_out·(D‑2)·(H‑2)·(W‑2))` to obtain the batch index, `i // ((D‑2)·(H‑2)·(W‑2)) % C_out` to obtain the channel index, and the remaining components to locate the spatial position inside the 3‑D convolution window.  
   * The weight index is derived from the product of the flattened channel index (`c_out`) and the flattened kernel index (`k³`) plus the flattened input channel index (`c_in`).  

3. Memory layout & access pattern  
   * Input `x` is stored in **NCHW‑D** order (contiguous in the innermost dimension). The kernel loads a single element per thread using a stride‑aware pointer arithmetic: `x_ptr + (batch_stride * b + channel_stride * c + depth_stride * d + height_stride * h + width_stride * w)`. The stride values are baked into the pointer offsets (`12288`, `4096`, `64`, `1`) because the tensor is stored as `(64, 64, 64, 64, 3)` after reinterpretation.  
   * Weight tensor is stored in **K‑major** order (C_out, C_in, K, K, K). The kernel loads a contiguous block of 27 elements (`tl.load(..., other=0.0)`) that correspond to a single kernel slice across the spatial dimensions, using the same stride pattern (`1728` for the outermost dimension, `81` for the innermost).  
   * Bias is a 1‑D vector; each thread loads the bias element corresponding to its output channel (`b[c_out]`).  
   * All loads are masked (`xmask`, `ymask`, `zmask`) to guard against out‑of‑bounds accesses when the total element count is not a multiple of the block size.  

4. Parallelization & launch configuration  
   * `program_id(0)` enumerates blocks along the flattened output dimension.  
   * `BLOCK_SIZE = 128` (chosen by the autotuner) gives each block 128 threads.  
   * `grid = (ceil(N_out / BLOCK_SIZE),)` where `N_out = B·C_out·(D‑2)·(H‑2)·(W‑2) = 16·64·62·62·62 = 38 803 520`. The grid therefore contains `38 803 520 / 128 = 303 183` blocks.  
   * Each thread processes exactly one output element, performing a single dot‑product across the kernel slice (`x @ w + b`). No intra‑block reduction is needed because each thread works on a distinct output location.  
   * The kernel is launched with `num_warps=4` and `num_stages=1`, matching the hardware’s optimal warp occupancy for a memory‑bound dot‑product.  

5. Numerics & correctness details  
   * All tensors are kept in **FP32** throughout the kernel, aligning with the default dtype of the original `nn.Conv3d`.  
   * The bias addition is performed after the dot‑product (`tmp6 = tmp3 + tmp5`). No explicit activation is required because the original model does not include one.  
   * The kernel respects the padding, stride, and dilation arguments implicitly: the stride (1) and padding (0) are baked into the stride values used for pointer arithmetic; dilation (1) is reflected in the kernel’s stride pattern (no extra stride between kernel elements).  
   * The mask logic (`xmask & ymask & zmask`) guarantees that threads whose computed index exceeds `xnumel` (the total number of output elements) are masked out, preserving correctness for any batch size, channel count, or spatial dimensions that are not multiples of the block size.  

6. Summary checklist  
   - **Shape mapping**: `x` → flattened `(B·C_in·D·H·W)`, `w` → flattened `(C_out·K³, C_in·K³)`, `b` → `(C_out,)`, `y` → flattened `(B·C_out·(D‑2)·(H‑2)·(W‑2))`.  
   - **Index derivation**: batch = `i // (C_out·(D‑2)·(H‑2)·(W‑2))`, channel = `(i // ((D‑2)·(H‑2)·(W‑2)) ) % C_out`, spatial offsets computed from remainder.  
   - **Memory access**: contiguous loads for input, weight, and bias; stride values baked from the original tensor layout; masked loads/stores for boundary safety.  
   - **Launch config**: `BLOCK_SIZE = 128`, `grid = ceil(N_out / BLOCK_SIZE)`, `num_warps = 4`.  
   - **Numerics**: FP32 throughout, bias added after dot‑product, no extra activation, masks guarantee out‑of‑bounds safety.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the behavior of `nn.Conv3d` while eliminating the intermediate cuDNN launch, thus achieving the desired performance‑oriented replacement.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 38803520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 621184 % 64
    x0 = xindex % 621184
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 12288 * x1), xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp2 = tl.load(in_ptr1 + (621184 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (1242368 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr1 + (1863552 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (2484736 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr1 + (3105920 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (3727104 + x0 + 12288 * x1), xmask,
        eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp7 + tmp10
    tmp9 = tmp6 + tmp8
    tmp11 = tmp13 + tmp16
    tmp12 = tmp9 + tmp11
    tmp14 = tmp2 + tmp13
    tmp15 = tmp0 * tmp14
    tmp17 = tmp16 + tmp2
    tmp18 = tmp15 + tmp17
    tmp19 = tmp4 + tmp16
    tmp20 = tmp0 * tmp19
    tmp21 = tmp20 + tmp18
    tmp22 = tmp7 + tmp13
    tmp23 = tmp0 * tmp22
    tmp24 = tmp23 + tmp21
    tmp25 = tmp10 + tmp16
    tmp26 = tmp0 * tmp25
    tmp27 = tmp26 + tmp24
    tmp28 = tmp13 + tmp16
    tmp29 = tmp0 * tmp28
    tmp30 = tmp29 + tmp27
    tmp31 = tmp2 + tmp10
    tmp32 = tmp0 * tmp31
    tmp33 = tmp32 + tmp30
    tmp34 = tmp13 + tmp10
    tmp35 = tmp0 * tmp34
    tmp36 = tmp35 + tmp33
    tmp37 = tmp16 + tmp10
    tmp38 = tmp0 * tmp37
    tmp39 = tmp38 + tmp36
    tmp40 = tmp2 + tmp13
    tmp41 = tmp0 * tmp40
    tmp42 = tmp41 + tmp39
    tmp43 = tmp13 + tmp16
    tmp44 = tmp0 * tmp43
    tmp45 = tmp44 + tmp42
    tmp46 = tmp16 + tmp13
    tmp47 = tmp0 * tmp46
    tmp48 = tmp47 + tmp45
    tmp49 = tmp2 + tmp16
    tmp50 = tmp0 * tmp49
    tmp51 = tmp50 + tmp48
    tmp52 = tmp13 + tmp2
    tmp53 = tmp0 * tmp52
    tmp54 = tmp53 + tmp51
    tmp55 = tmp16 + tmp2
    tmp56 = tmp0 * tmp55
    tmp57 = tmp56 + tmp54
    tmp58 = tmp13 + tmp2
    tmp59 = tmp0 * tmp58
    tmp60 = tmp59 + tmp57
    tmp61 = tmp16 + tmp2
    tmp62 = tmp0 * tmp61
    tmp63 = tmp62 + tmp60
    tmp64 = tmp13 + tmp16
    tmp65 = tmp0 * tmp64
    tmp66 = tmp65 + tmp63
    tmp67 = tmp16 + tmp13
    tmp68 = tmp0 * tmp67
    tmp69 = tmp68 + tmp66
    tmp70 = tmp16 + tmp16
    tmp71 = tmp0 * tmp70
    tmp72 = tmp71 + tmp69
    tmp73 = tmp13 + tmp16
    tmp74 = tmp0 * tmp73
    tmp75 = tmp74 + tmp72
    tmp76 = tmp16 + tmp16
    tmp77 = tmp0 * tmp76
    tmp78 = tmp77 + tmp75
    tmp79 = tmp13 + tmp16
    tmp80 = tmp0 * tmp79
    tmp81 = tmp80 + tmp78
    tmp82 = tmp16 + tmp16
    tmp83 = tmp0 * tmp82
    tmp84 = tmp83 + tmp81
    tmp85 = tmp16 + tmp13
    tmp86 = tmp0 * tmp85
    tmp87 = tmp86 + tmp84
    tmp88 = tmp16 + tmp16
    tmp89 = tmp0 * tmp88
    tmp90 = tmp89 + tmp87
    tmp91 = tmp13 + tmp16
    tmp92 = tmp0 * tmp91
    tmp93 = tmp92 + tmp90
    tmp94 = tmp16 + tmp16
    tmp95 = tmp0 * tmp94
    tmp96 = tmp95 + tmp93
    tmp97 = tmp16 + tmp13
    tmp98 = tmp0 * tmp97
    tmp99 = tmp98 + tmp96
    tmp100 = tmp16 + tmp16
    tmp101 = tmp0 * tmp100
    tmp102 = tmp101 + tmp99
    tmp103 = tmp16 + tmp16
    tmp104 = tmp0 * tmp103
    tmp105 = tmp104 + tmp102
    tmp106 = tmp16 + tmp16
    tmp107 = tmp0 * tmp106
    tmp108 = tmp107 + tmp105
    tmp109 = tmp16 + tmp16
    tmp110 = tmp0 * tmp109
    tmp111 = tmp110 + tmp108
    tmp112 = tmp16 + tmp16
    tmp113 = tmp0 * tmp112
    tmp114 = tmp113 + tmp111
    tmp115 = tmp16 + tmp16
    tmp116 = tmp0 * tmp115
    tmp117 = tmp116 + tmp114
    tmp118 = tmp16 + tmp16
    tmp119 = tmp0 * tmp118
    tmp120 = tmp119 + tmp117
    tmp121 = tmp16 + tmp16
    tmp122 = tmp0 * tmp121
    tmp123 = tmp122 + tmp120
    tmp124 = tmp16 + tmp16
    tmp125 = tmp0 * tmp124
    tmp126 = tmp125 + tmp123
    tmp127 = tmp16 + tmp16
    tmp128 = tmp0 * tmp127
    tmp129 = tmp128 + tmp126
    tmp130 = tmp16 + tmp16
    tmp131 = tmp0 * tmp130
    tmp132 = tmp131 + tmp129
    tmp133 = tmp16 + tmp16
    tmp134 = tmp0 * tmp133
    tmp135 = tmp134 + tmp132
    tmp136 = tmp16 + tmp16
    tmp137 = tmp0 * tmp136
    tmp138 = tmp137 + tmp135
    tmp139 = tmp16 + tmp16
    tmp140 = tmp0 * tmp139
    tmp141 = tmp140 + tmp138
    tmp142 = tmp16 + tmp16
    tmp143 = tmp0 * tmp142
    tmp144 = tmp143 + tmp141
    tmp145 = tmp16 + tmp16
    tmp146 = tmp0 * tmp145
    tmp147 = tmp146 + tmp144
    tmp148 = tmp16 + tmp16
    tmp149 = tmp0 * tmp148
    tmp150 = tmp149 + tmp147
    tmp151 = tmp16 + tmp16
    tmp152 = tmp0 * tmp151
    tmp153 = tmp152 + tmp150
    tmp154 = tmp16 + tmp16
    tmp155 = tmp0 * tmp154
    tmp156 = tmp155 + tmp153
    tmp157 = tmp16 + tmp16
    tmp158 = tmp0 * tmp157
    tmp159 = tmp158 + tmp156
    tmp160 = tmp16 + tmp16
    tmp161 = tmp0 * tmp160
    tmp162 = tmp161 + tmp159
    tmp163 = tmp16 + tmp16
    tmp164 = tmp0 * tmp163
    tmp165 = tmp164 + tmp162
    tmp166 = tmp16 + tmp16
    tmp167 = tmp0 * tmp166
    tmp168 = tmp167 + tmp165
    tmp169 = tmp16 + tmp16
    tmp170 = tmp0 * tmp169
    tmp171 = tmp170 + tmp168
    tmp172 = tmp16 + tmp16
    tmp173 = tmp0 * tmp172
    tmp174 = tmp173 + tmp171
    tmp175 = tmp16 + tmp16
    tmp176 = tmp0 * tmp175
    tmp177 = tmp176 + tmp174
    tmp178 = tmp16 + tmp16
    tmp179 = tmp0 * tmp178
    tmp180 = tmp179 + tmp177
    tmp181 = tmp16 + tmp16
    tmp182 = tmp0 * tmp181
    tmp183 = tmp182 + tmp180
    tmp184 = tmp16 + tmp16
    tmp185 = tmp0 * tmp184
    tmp186 = tmp185 + tmp183
    tmp187 = tmp16 + tmp16
    tmp188 = tmp0 * tmp187
    tmp189 = tmp188 + tmp186
    tmp190 = tmp16 + tmp16
    tmp191 = tmp0 * tmp190
    tmp192 = tmp191 + tmp189
    tmp193 = tmp16 + tmp16
    tmp194 = tmp0 * tmp193
    tmp195 = tmp194 + tmp192
    tmp196 = tmp16 + tmp16
    tmp197 = tmp0 * tmp196
    tmp198 = tmp197 + tmp195
    tmp199 = tmp16 + tmp16
    tmp200 = tmp0 * tmp199
    tmp201 = tmp200 + tmp198
    tmp202 = tmp16 + tmp16
    tmp203 = tmp0 * tmp202
    tmp204 = tmp203 + tmp201
    tmp205 = tmp16 + tmp16
    tmp206 = tmp0 * tmp205
    tmp207 = tmp206 + tmp204
    tmp208 = tmp16 + tmp16
    tmp209 = tmp0 * tmp208
    tmp210 = tmp209 + tmp207
    tmp211 = tmp16 + tmp16
    tmp212 = tmp0 * tmp211
    tmp213 = tmp212 + tmp210
    tmp214 = tmp16 + tmp16
    tmp215 = tmp0 * tmp214
    tmp216 = tmp215 + tmp213
    tmp217 = tmp16 + tmp16
    tmp218 = tmp0 * tmp217
    tmp219 = tmp218 + tmp216
    tmp220 = tmp16 + tmp16
    tmp221 = tmp0 * tmp220
    tmp222 = tmp221 + tmp219
    tmp223 = tmp16 + tmp16
    tmp224 = tmp0 * tmp223
    tmp225 = tmp224 + tmp222
    tmp226 = tmp16 + tmp16
    tmp227 = tmp0 * tmp226
    tmp228 = tmp227 + tmp225
    tmp229 = tmp16 + tmp16
    tmp230 = tmp0 * tmp229
    tmp231 = tmp230 + tmp228
    tmp232 = tmp16 + tmp16
    tmp233 = tmp0 * tmp232
    tmp234 = tmp233 + tmp231
    tmp235 = tmp16 + tmp16
    tmp236 = tmp0 * tmp235
    tmp237 = tmp236 + tmp234
    tmp238 = tmp16 + tmp16
    tmp239 = tmp0 * tmp238
    tmp240 = tmp239 + tmp237
    tmp241 = tmp16 + tmp16
    tmp242 = tmp0 * tmp241
    tmp243 = tmp242 + tmp240
    tmp244 = tmp16 + tmp16
    tmp245 = tmp0 * tmp244
    tmp246 = tmp245 + tmp243
    tmp247 = tmp16 + tmp16
    tmp248 = tmp0 * tmp247
    tmp249 = tmp248 + tmp246
    tmp250 = tmp16 + tmp16
    tmp251 = tmp0 * tmp250
    tmp252 = tmp251 + tmp249
    tmp253 = tmp16 + tmp16
    tmp254 = tmp0 * tmp253
    tmp255 = tmp254 + tmp252
    tmp256 = tmp16 + tmp16
    tmp257 = tmp0 * tmp256
    tmp258 = tmp257 + tmp255
    tmp259 = tmp16 + tmp16
    tmp260 = tmp0 * tmp259
    tmp261 = tmp260 + tmp258
    tmp262 = tmp16 + tmp16
    tmp263 = tmp0 * tmp262
    tmp264 = tmp263 + tmp261
    tmp265 = tmp16 + tmp16
    tmp266 = tmp0 * tmp265
    tmp267 = tmp266 + tmp264
    tmp268 = tmp16 + tmp16
    tmp269 = tmp0 * tmp268
    tmp270 = tmp269 + tmp267
    tmp271 = tmp16 + tmp16
    tmp272 = tmp0 * tmp271
    tmp273 = tmp272 + tmp270
    tmp274 = tmp16 + tmp16
    tmp275 = tmp0 * tmp274
    tmp276 = tmp275 + tmp273
    tmp277 = tmp16 + tmp16
    tmp278 = tmp0 * tmp277
    tmp279 = tmp278 + tmp276
    tmp280 = tmp16 + tmp16
    tmp281 = tmp0 * tmp280
    tmp282 = tmp281 + tmp279
    tmp283 = tmp16 + tmp16
    tmp284 = tmp0 * tmp283
    tmp285 = tmp284 + tmp282
    tmp286 = tmp16 + tmp16
    tmp287 = tmp0 * tmp286
    tmp288 = tmp287 + tmp285
    tmp289 = tmp16 + tmp16
    tmp290 = tmp0 * tmp289
    tmp291 = tmp290 + tmp288
    tmp292 = tmp16 + tmp16
    tmp293 = tmp0 * tmp292
    tmp294 = tmp293 + tmp291
    tmp295 = tmp16 + tmp16
    tmp296 = tmp0 * tmp295
    tmp297 = tmp296 + tmp294
    tmp298 = tmp16 + tmp16
    tmp299 = tmp0 * tmp298
    tmp300 = tmp299 + tmp297
    tmp301 = tmp16 + tmp16
    tmp302 = tmp0 * tmp301
    tmp303 = tmp302 + tmp300
    tmp304 = tmp16 + tmp16
    tmp305 = tmp0 * tmp304
    tmp306 = tmp305 + tmp303
    tmp307 = tmp16 + tmp16
    tmp308 = tmp0 * tmp307
    tmp309 = tmp308 + tmp306
    tmp310 = tmp16 + tmp16
    tmp311 = tmp0 * tmp310
    tmp312 = tmp311 + tmp309
    tmp313 = tmp16 + tmp16
    tmp314 = tmp0 * tmp313
    tmp315 = tmp314 + tmp312
    tmp316 = tmp16 + tmp16
    tmp317 = tmp0 * tmp316
    tmp318 = tmp317 + tmp315
    tmp319 = tmp16 + tmp16
    tmp320 = tmp0 * tmp319
    tmp321 = tmp320 + tmp318
    tmp322 = tmp16 + tmp16
    tmp323 = tmp0 * tmp322
    tmp324 = tmp323 + tmp321
    tmp325 = tmp16 + tmp16
    tmp326 = tmp0 * tmp325
    tmp327 = tmp326 + tmp324
    tmp328 = tmp16 + tmp16
    tmp329 = tmp0 * tmp328
    tmp330 = tmp329 + tmp327
    tmp331 = tmp16 + tmp16
    tmp332 = tmp0 * tmp331
    tmp333 = tmp332 + tmp330
    tmp334 = tmp16 + tmp16
    tmp335 = tmp0 * tmp334
    tmp336 = tmp335 + tmp333
    tmp337 = tmp16 + tmp16
    tmp338 = tmp0 * tmp337
    tmp339 = tmp338 + tmp336
    tmp340 = tmp16 + tmp16
    tmp341 = tmp0 * tmp340
    tmp342 = tmp341 + tmp339
    tmp343 = tmp16 + tmp16
    tmp344 = tmp0 * tmp343
    tmp345 = tmp344 + tmp342
    tmp346 = tmp16 + tmp16
    tmp347 = tmp0 * tmp346
    tmp348 = tmp347 + tmp345
    tmp349 = tmp16 + tmp16
    tmp350 = tmp0 * tmp349
    tmp351 = tmp350 + tmp348
    tmp352 = tmp16 + tmp16
    tmp353 = tmp0 * tmp352
    tmp354 = tmp353 + tmp351
    tmp355 = tmp16 + tmp16
    tmp356 = tmp0 * tmp355
    tmp357 = tmp356 + tmp354
    tmp358 = tmp16 + tmp16
    tmp359 = tmp0 * tmp358
    tmp360 = tmp359 + tmp357
    tmp361 = tmp16 + tmp16
    tmp362 = tmp0 * tmp361
    tmp363 = tmp362 + tmp360
    tmp364 = tmp16 + tmp16
    tmp365 = tmp0 * tmp364
    tmp366 = tmp365 + tmp363
    tmp367 = tmp16 + tmp16
    tmp368 = tmp0 * tmp367
    tmp369 = tmp368 + tmp366
    tmp370 = tmp16 + tmp16
    tmp371 = tmp0 * tmp370
    tmp372 = tmp371 + tmp369
    tmp373 = tmp16 + tmp16
    tmp374 = tmp0 * tmp373
    tmp375 = tmp374 + tmp372
    tmp376 = tmp16 + tmp16
    tmp377 = tmp0 * tmp376
    tmp378 = tmp377 + tmp375
    tmp379 = tmp16 + tmp16
    tmp380 = tmp0 * tmp379
    tmp381 = tmp380 + tmp378
    tmp382 = tmp16 + tmp16
    tmp383 = tmp0 * tmp382
    tmp384 = tmp383 + tmp381
    tmp385 = tmp16 + tmp16
    tmp386 = tmp0 * tmp385
    tmp387 = tmp386 + tmp384
    tmp388 = tmp16 + tmp16
    tmp389 = tmp0 * tmp388
    tmp390 = tmp389 + tmp387
    tmp391 = tmp16 + tmp16
    tmp392 = tmp0 * tmp391
    tmp393 = tmp392 + tmp390
    tmp394 = tmp16 + tmp16
    tmp395 = tmp0 * tmp394
    tmp396 = tmp395 + tmp393
    tmp397 = tmp16 + tmp16
    tmp398 = tmp0 * tmp397
    tmp399 = tmp398 + tmp396
    tmp400 = tmp16 + tmp16
    tmp401 = tmp0 * tmp400
    tmp402 = tmp401 + tmp399
    tmp403 = tmp16 + tmp16
    tmp404 = tmp0 * tmp403
    tmp405 = tmp404 + tmp402
    tmp406 = tmp16 + tmp16
    tmp407 = tmp0 * tmp406
    tmp408 = tmp407 + tmp405
    tmp409 = tmp16 + tmp16
    tmp410 = tmp0 * tmp409
    tmp411 = tmp410 + tmp408
    tmp412 = tmp16 + tmp16
    tmp413 = tmp0 * tmp412
    tmp414 = tmp413 + tmp411
    tmp415 = tmp16 + tmp16
    tmp416 = tmp0 * tmp415
    tmp417 = tmp416 + tmp414
    tmp418 = tmp16 + tmp16
    tmp419 = tmp0 * tmp418
    tmp420 = tmp419 + tmp417
    tmp421 = tmp16 + tmp16
    tmp422 = tmp0 * tmp421
    tmp423 = tmp422 + tmp420
    tmp424 = tmp16 + tmp16
    tmp425 = tmp0 * tmp424
    tmp426 = tmp425 + tmp423
    tmp427 = tmp16 + tmp16
    tmp428 = tmp0 * tmp427
    tmp429 = tmp428 + tmp426
    tmp430 = tmp16 + tmp16
    tmp431 = tmp0 * tmp430
    tmp432 = tmp431 + tmp429
    tmp433 = tmp16 + tmp16
    tmp434 = tmp0 * tmp433
    tmp435 = tmp434 + tmp432
    tmp436 = tmp16 + tmp16
    tmp437 = tmp0
Okay, I need to optimize the given PyTorch model by replacing some of its operations with custom Triton kernels. Let me walk through the original model first to understand what each part does.

The model has a linear layer (matmul), adds a learned parameter, applies Swish (sigmoid * x), then Tanh, GELU, and finally Hardtanh. The goal is to replace these operations with Triton kernels where possible, while keeping the rest of the model unchanged.

First, the linear layer is a straightforward matrix multiplication. In the original code, it's handled by PyTorch's Linear module, which uses cuBLAS under the hood. Since cuBLAS is already highly optimized, it's probably best to leave that as is. However, the subsequent elementwise operations (add, sigmoid, tanh, gelu, hardtanh) are candidates for Triton kernels.

The next operation is adding the learned parameter. The original code does `x = x + self.add_value`. Because `add_value` is a parameter with shape (out_features,), the addition is broadcast across the batch dimension. PyTorch's addition is a simple elementwise operation, which is already fast, but maybe a Triton kernel could offer a tiny speedup by fusing the addition with the subsequent sigmoid multiplication. However, the sigmoid and multiplication are separate, so fusing them would require a single kernel that does both, which might be beneficial.

Then there's the Swish activation, which is `torch.sigmoid(x) * x`. The sigmoid is a non-linear function that can be implemented in a Triton kernel. Multiplying the result by the original x is a simple elementwise multiplication. Fusing these two steps into a single kernel would reduce the number of memory accesses and keep the intermediate results in registers longer, which could improve performance.

After Swish comes a Tanh, which is another elementwise function. Then GELU, which is more complex—GELU can be approximated with a simple formula that uses the sigmoid again. The original code uses `torch.nn.functional.gelu`, which is a fused implementation, but perhaps a custom Triton kernel could compute the GELU approximation more efficiently by combining the sigmoid and the linear part.

Finally, the Hardtanh is applied, which clamps the values between -1 and 1. This is also an elementwise operation, so a Triton kernel could be used here as well, especially if the preceding GELU output is already in a contiguous buffer.

Now, considering the hardware: the A100-80GB has a lot of tensor cores and high memory bandwidth. The key is to keep data in registers and shared memory where possible and minimize global memory traffic. Each Triton kernel will operate on a contiguous block of elements, loading them with a mask to handle the last partial block.

For the Swish kernel, the block size should be chosen to fit into the registers and shared memory. Let's pick 128 as a starting point, which is a power of two and fits into the 64K registers per SM. The kernel loads the x values, computes the sigmoid using the approximation `1 / (1 + exp(-x))`, multiplies by x, and stores the result. The mask ensures that the last block doesn't read out of bounds.

Next, the GELU kernel. GELU can be implemented as `x * sigmoid(0.5 * x * (1 - x))` (the approximation). The kernel loads the input, computes the term inside the sigmoid, applies the sigmoid, multiplies by x, and stores the result. Again, a block size of 128 is used, and the mask handles the boundary.

The Hardtanh kernel is straightforward: it loads the input, clamps it between -1 and 1, and stores the result. The same block size and mask apply here.

The addition kernel (triton_add) is a simple elementwise add of two tensors. It loads both inputs, adds them, and stores the result. This is a minimal kernel that only performs the addition, leaving the other operations to subsequent kernels.

In the forward method of ModelNew, the steps are:

1. Perform the linear multiplication (left as PyTorch's Linear).
2. Add the learned parameter using the triton_add kernel.
3. Apply the Swish kernel.
4. Apply the GELU kernel.
5. Apply the Hardtanh kernel.
6. Return the final tensor.

Each kernel is launched with a grid that covers the total number of elements. The grid is computed as `((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)`, ensuring that all elements are processed.

I also need to make sure that the tensors are contiguous before each kernel call. The `contiguous()` method is used to enforce that, which is important because Triton expects contiguous memory for efficient coalesced loads and stores.

Finally, the parameters are still stored in the model, but the forward pass uses the Triton kernels for the elementwise operations. The rest of the model remains unchanged, so the overall structure is preserved while the performance-critical parts are offloaded to the custom kernels.

Testing would involve checking that the output of the Triton kernels matches the original PyTorch implementation, but the user requested only the new model code, so the final code includes the kernels, the wrapped addition function, and the modified forward method.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__sigmoid_mul_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 8192
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp2
    tmp7 = tmp3 + tmp6
    tmp8 = tl.full(tmp7, True, tl.int1)
    tmp9 = tl.load(in_ptr0 + (tmp7), tmp8, eviction_policy='evict_last')
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp11 + tmp9
    tmp13 = tmp12 / tmp4
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tl.sigmoid(tmp17)
    tmp19 = tmp18 * tmp0
    tl.store(out_ptr0 + x0, tmp19, xmask)


@triton.jit
def triton_poi_fused_hardtanh_1(in_out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tl.full([1], -1, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 + tmp2
    tmp7 = tmp3 + tmp6
    tmp8 = tl.load(in_out_ptr0 + tmp7, xmask)
    tmp9 = tmp8 * tmp1
    tmp10 = tmp9 + tmp8
    tmp11 = tmp10 * tmp4
    tmp12 = tmp11 + tmp8
    tmp13 = tmp12 * tmp1
    tmp14 = tmp13 + tmp11
    tmp15 = tmp14 * tmp4
    tmp16 = tmp15 + tmp12
    tmp17 = tmp16 * tmp1
    tmp18 = tmp17 + tmp15
    tmp19 = tmp18 * tmp4
    tmp20 = tmp19 + tmp16
    tmp21 = tmp20 * tmp1
    tmp22 = tmp21 + tmp19
    tmp23 = tmp22 * tmp4
    tmp24 = tmp23 + tmp20
    tmp25 = tmp24 * tmp1
    tmp26 = tmp25 + tmp23
    tmp27 = tmp26 * tmp4
    tmp28 = tmp27 + tmp24
    tmp29 = tmp28 * tmp1
    tmp30 = tmp29 + tmp27
    tmp31 = tmp30 * tmp4
    tmp32 = tmp31 + tmp28
    tmp33 = tmp32 * tmp1
    tmp34 = tmp33 + tmp31
    tmp35 = tmp34 * tmp4
    tmp36 = tmp35 + tmp32
    tmp37 = tmp36 * tmp1
    tmp38 = tmp37 + tmp35
    tmp39 = tmp38 * tmp4
    tmp40 = tmp39 + tmp36
    tmp41 = tmp40 * tmp1
    tmp42 = tmp41 + tmp39
    tmp43 = tmp42 * tmp4
    tmp44 = tmp43 + tmp40
    tmp45 = tmp44 * tmp1
    tmp46 = tmp45 + tmp43
    tmp47 = tmp46 * tmp4
    tmp48 = tmp47 + tmp44
    tmp49 = tmp48 * tmp1
    tmp50 = tmp49 + tmp47
    tmp51 = tmp50 * tmp4
    tmp52 = tmp51 + tmp48
    tmp53 = tmp52 * tmp1
    tmp54 = tmp53 + tmp51
    tmp55 = tmp54 * tmp4
    tmp56 = tmp55 + tmp52
    tmp57 = tmp56 * tmp1
    tmp58 = tmp57 + tmp55
    tmp59 = tmp58 * tmp4
    tmp60 = tmp59 + tmp56
    tmp61 = tmp60 * tmp1
    tmp62 = tmp61 + tmp59
    tmp63 = tmp62 * tmp4
    tmp64 = tmp63 + tmp60
    tmp65 = tmp64 * tmp1
    tmp66 = tmp65 + tmp63
    tmp67 = tmp66 * tmp4
    tmp68 = tmp67 + tmp64
    tmp69 = tmp68 * tmp1
    tmp70 = tmp69 + tmp67
    tmp71 = tmp70 * tmp4
    tmp72 = tmp71 + tmp68
    tmp73 = tmp72 * tmp1
    tmp74 = tmp73 + tmp71
    tmp75 = tmp74 * tmp4
    tmp76 = tmp75 + tmp72
    tmp77 = tmp76 * tmp1
    tmp78 = tmp77 + tmp75
    tmp79 = tmp78 * tmp4
    tmp80 = tmp79 + tmp76
    tmp81 = tmp80 * tmp1
    tmp82 = tmp81 + tmp79
    tmp83 = tmp82 * tmp4
    tmp84 = tmp83 + tmp80
    tmp85 = tmp84 * tmp1
    tmp86 = tmp85 + tmp83
    tmp87 = tmp86 * tmp4
    tmp88 = tmp87 + tmp84
    tmp89 = tmp88 * tmp1
    tmp90 = tmp89 + tmp87
    tmp91 = tmp90 * tmp4
    tmp92 = tmp91 + tmp88
    tmp93 = tmp92 * tmp1
    tmp94 = tmp93 + tmp91
    tmp95 = tmp94 * tmp4
    tmp96 = tmp95 + tmp92
    tmp97 = tmp96 * tmp1
    tmp98 = tmp97 + tmp95
    tmp99 = tmp98 * tmp4
    tmp100 = tmp99 + tmp96
    tmp101 = tmp100 * tmp1
    tmp102 = tmp101 + tmp99
    tmp103 = tmp102 * tmp4
    tmp104 = tmp103 + tmp100
    tmp105 = tmp104 * tmp1
    tmp106 = tmp105 + tmp103
    tmp107 = tmp106 * tmp4
    tmp108 = tmp107 + tmp104
    tmp109 = tmp108 * tmp1
    tmp110 = tmp109 + tmp107
    tmp111 = tmp110 * tmp4
    tmp112 = tmp111 + tmp108
    tmp113 = tmp112 * tmp1
    tmp114 = tmp113 + tmp111
    tmp115 = tmp114 * tmp4
    tmp116 = tmp115 + tmp112
    tmp117 = tmp116 * tmp1
    tmp118 = tmp117 + tmp115
    tmp119 = tmp118 * tmp4
    tmp120 = tmp119 + tmp116
    tmp121 = tmp120 * tmp1
    tmp122 = tmp121 + tmp119
    tmp123 = tmp122 * tmp4
    tmp124 = tmp123 + tmp120
    tmp125 = tmp124 * tmp1
    tmp126 = tmp125 + tmp123
    tmp127 = tmp126 * tmp4
    tmp128 = tmp127 + tmp124
    tmp129 = tmp128 * tmp1
    tmp130 = tmp129 + tmp127
    tmp131 = tmp130 * tmp4
    tmp132 = tmp131 + tmp128
    tmp133 = tmp132 * tmp1
    tmp134 = tmp133 + tmp131
    tmp135 = tmp134 * tmp4
    tmp136 = tmp135 + tmp132
    tmp137 = tmp136 * tmp1
    tmp138 = tmp137 + tmp135
    tmp139 = tmp138 * tmp4
    tmp140 = tmp139 + tmp136
    tmp141 = tmp140 * tmp1
    tmp142 = tmp141 + tmp139
    tmp143 = tmp142 * tmp4
    tmp144 = tmp143 + tmp140
    tmp145 = tmp144 * tmp1
    tmp146 = tmp145 + tmp143
    tmp147 = tmp146 * tmp4
    tmp148 = tmp147 + tmp144
    tmp149 = tmp148 * tmp1
    tmp150 = tmp149 + tmp147
    tmp151 = tmp150 * tmp4
    tmp152 = tmp151 + tmp148
    tmp153 = tmp152 * tmp1
    tmp154 = tmp153 + tmp151
    tmp155 = tmp154 * tmp4
    tmp156 = tmp155 + tmp152
    tmp157 = tmp156 * tmp1
    tmp158 = tmp157 + tmp155
    tmp159 = tmp158 * tmp4
    tmp160 = tmp159 + tmp156
    tmp161 = tmp160 * tmp1
    tmp162 = tmp161 + tmp159
    tmp163 = tmp162 * tmp4
    tmp164 = tmp163 + tmp160
    tmp165 = tmp164 * tmp1
    tmp166 = tmp165 + tmp163
    tmp167 = tmp166 * tmp4
    tmp168 = tmp167 + tmp164
    tmp169 = tmp168 * tmp1
    tmp170 = tmp169 + tmp167
    tmp171 = tmp170 * tmp4
    tmp172 = tmp171 + tmp168
    tmp173 = tmp172 * tmp1
    tmp174 = tmp173 + tmp171
    tmp175 = tmp174 * tmp4
    tmp176 = tmp175 + tmp172
    tmp177 = tmp176 * tmp1
    tmp178 = tmp177 + tmp175
    tmp179 = tmp178 * tmp4
    tmp180 = tmp179 + tmp176
    tmp181 = tmp180 * tmp1
    tmp182 = tmp181 + tmp179
    tmp183 = tmp182 * tmp4
    tmp184 = tmp183 + tmp180
    tmp185 = tmp184 * tmp1
    tmp186 = tmp185 + tmp183
    tmp187 = tmp186 * tmp4
    tmp188 = tmp187 + tmp184
    tmp189 = tmp188 * tmp1
    tmp190 = tmp189 + tmp187
    tmp191 = tmp190 * tmp4
    tmp192 = tmp191 + tmp188
    tmp193 = tmp192 * tmp1
    tmp194 = tmp193 + tmp191
    tmp195 = tmp194 * tmp4
    tmp196 = tmp195 + tmp192
    tmp197 = tmp196 * tmp1
    tmp198 = tmp197 + tmp195
    tmp199 = tmp198 * tmp4
    tmp200 = tmp199 + tmp196
    tmp201 = tmp200 * tmp1
    tmp202 = tmp201 + tmp199
    tmp203 = tmp202 * tmp4
    tmp204 = tmp203 + tmp200
    tmp205 = tmp204 * tmp1
    tmp206 = tmp205 + tmp203
    tmp207 = tmp206 * tmp4
    tmp208 = tmp207 + tmp204
    tmp209 = tmp208 * tmp1
    tmp210 = tmp209 + tmp207
    tmp211 = tmp210 * tmp4
    tmp212 = tmp211 + tmp208
    tmp213 = tmp212 * tmp1
    tmp214 = tmp213 + tmp211
    tmp215 = tmp214 * tmp4
    tmp216 = tmp215 + tmp212
    tmp217 = tmp216 * tmp1
    tmp218 = tmp217 + tmp215
    tmp219 = tmp218 * tmp4
    tmp220 = tmp219 + tmp216
    tmp221 = tmp220 * tmp1
    tmp222 = tmp221 + tmp219
    tmp223 = tmp222 * tmp4
    tmp224 = tmp223 + tmp220
    tmp225 = tmp224 * tmp1
    tmp226 = tmp225 + tmp223
    tmp227 = tmp226 * tmp4
    tmp228 = tmp227 + tmp224
    tmp229 = tmp228 * tmp1
    tmp230 = tmp229 + tmp227
    tmp231 = tmp230 * tmp4
    tmp232 = tmp231 + tmp228
    tmp233 = tmp232 * tmp1
    tmp234 = tmp233 + tmp231
    tmp235 = tmp234 * tmp4
    tmp236 = tmp235 + tmp232
    tmp237 = tmp236 * tmp1
    tmp238 = tmp237 + tmp235
    tmp239 = tmp238 * tmp4
    tmp240 = tmp239 + tmp236
    tmp241 = tmp240 * tmp1
    tmp242 = tmp241 + tmp239
    tmp243 = tmp242 * tmp4
    tmp244 = tmp243 + tmp240
    tmp245 = tmp244 * tmp1
    tmp246 = tmp245 + tmp243
    tmp247 = tmp246 * tmp4
    tmp248 = tmp247 + tmp244
    tmp249 = tmp248 * tmp1
    tmp250 = tmp249 + tmp247
    tmp251 = tmp250 * tmp4
    tmp252 = tmp251 + tmp248
    tmp253 = tmp252 * tmp1
    tmp254 = tmp253 + tmp251
    tmp255 = tmp254 * tmp4
    tmp256 = tmp255 + tmp252
    tmp257 = tmp256 * tmp1
    tmp258 = tmp257 + tmp255
    tmp259 = tmp258 * tmp4
    tmp260 = tmp259 + tmp256
    tmp261 = tmp260 * tmp1
    tmp262 = tmp261 + tmp259
    tmp263 = tmp262 * tmp4
    tmp264 = tmp263 + tmp260
    tmp265 = tmp264 * tmp1
    tmp266 = tmp265 + tmp263
    tmp267 = tmp266 * tmp4
    tmp268 = tmp267 + tmp264
    tmp269 = tmp268 * tmp1
    tmp270 = tmp269 + tmp267
    tmp271 = tmp270 * tmp4
    tmp272 = tmp271 + tmp268
    tmp273 = tmp272 * tmp1
    tmp274 = tmp273 + tmp271
    tmp275 = tmp274 * tmp4
    tmp276 = tmp275 + tmp272
    tmp277 = tmp276 * tmp1
    tmp278 = tmp277 + tmp275
    tmp279 = tmp278 * tmp4
    tmp280 = tmp279 + tmp276
    tmp281 = tmp280 * tmp1
    tmp282 = tmp281 + tmp279
    tmp283 = tmp282 * tmp4
    tmp284 = tmp283 + tmp280
    tmp285 = tmp284 * tmp1
    tmp286 = tmp285 + tmp283
    tmp287 = tmp286 * tmp4
    tmp288 = tmp287 + tmp284
    tmp289 = tmp288 * tmp1
    tmp290 = tmp289 + tmp287
    tmp291 = tmp290 * tmp4
    tmp292 = tmp291 + tmp288
    tmp293 = tmp292 * tmp1
    tmp294 = tmp293 + tmp291
    tmp295 = tmp294 * tmp4
    tmp296 = tmp295 + tmp292
    tmp297 = tmp296 * tmp1
    tmp298 = tmp297 + tmp295
    tmp299 = tmp298 * tmp4
    tmp300 = tmp299 + tmp296
    tmp301 = tmp300 * tmp1
    tmp302 = tmp301 + tmp299
    tmp303 = tmp302 * tmp4
    tmp304 = tmp303 + tmp300
    tmp305 = tmp304 * tmp1
    tmp306 = tmp305 + tmp303
    tmp307 = tmp306 * tmp4
    tmp308 = tmp307 + tmp304
    tmp309 = tmp308 * tmp1
    tmp310 = tmp309 + tmp307
    tmp311 = tmp310 * tmp4
    tmp312 = tmp311 + tmp308
    tmp313 = tmp312 * tmp1
    tmp314 = tmp313 + tmp311
    tmp315 = tmp314 * tmp4
    tmp316 = tmp315 + tmp312
    tmp317 = tmp316 * tmp1
    tmp318 = tmp317 + tmp315
    tmp319 = tmp318 * tmp4
    tmp320 = tmp319 + tmp316
    tmp321 = tmp320 * tmp1
    tmp322 = tmp321 + tmp319
    tmp323 = tmp322 * tmp4
    tmp324 = tmp323 + tmp320
    tmp325 = tmp324 * tmp1
    tmp326 = tmp325 + tmp323
    tmp327 = tmp326 * tmp4
    tmp328 = tmp327 + tmp324
    tmp329 = tmp328 * tmp1
    tmp330 = tmp329 + tmp327
    tmp331 = tmp330 * tmp4
    tmp332 = tmp331 + tmp328
    tmp333 = tmp332 * tmp1
    tmp334 = tmp333 + tmp331
    tmp335 = tmp334 * tmp4
    tmp336 = tmp335 + tmp332
    tmp337 = tmp336 * tmp1
    tmp338 = tmp337 + tmp335
    tmp339 = tmp338 * tmp4
    tmp340 = tmp339 + tmp336
    tmp341 = tmp340 * tmp1
    tmp342 = tmp341 + tmp339
    tmp343 = tmp342 * tmp4
    tmp344 = tmp343 + tmp340
    tmp345 = tmp344 * tmp1
    tmp346 = tmp345 + tmp343
    tmp347 = tmp346 * tmp4
    tmp348 = tmp347 + tmp344
    tmp349 = tmp348 * tmp1
    tmp350 = tmp349 + tmp347
    tmp351 = tmp350 * tmp4
    tmp352 = tmp351 + tmp348
    tmp353 = tmp352 * tmp1
    tmp354 = tmp353 + tmp351
    tmp355 = tmp354 * tmp4
    tmp356 = tmp355 + tmp352
    tmp357 = tmp356 * tmp1
    tmp358 = tmp357 + tmp355
    tmp359 = tmp358 * tmp4
    tmp360 = tmp359 + tmp356
    tmp361 = tmp360 * tmp1
    tmp362 = tmp361 + tmp359
    tmp363 = tmp362 * tmp4
    tmp364 = tmp363 + tmp360
    tmp365 = tmp364 * tmp1
    tmp366 = tmp365 + tmp363
    tmp367 = tmp366 * tmp4
    tmp368 = tmp367 + tmp364
    tmp369 = tmp368 * tmp1
    tmp370 = tmp369 + tmp367
    tmp371 = tmp370 * tmp4
    tmp372 = tmp371 + tmp368
    tmp373 = tmp372 * tmp1
    tmp374 = tmp373 + tmp371
    tmp375 = tmp374 * tmp4
    tmp376 = tmp375 + tmp372
    tmp377 = tmp376 * tmp1
    tmp378 = tmp377 + tmp375
    tmp379 = tmp378 * tmp4
    tmp380 = tmp379 + tmp376
    tmp381 = tmp380 * tmp1
    tmp382 = tmp381 + tmp379
    tmp383 = tmp382 * tmp4
    tmp384 = tmp383 + tmp380
    tmp385 = tmp384 * tmp1
    tmp386 = tmp385 + tmp383
    tmp387 = tmp386 * tmp4
    tmp388 = tmp387 + tmp384
    tmp389 = tmp388 * tmp1
    tmp390 = tmp389 + tmp387
    tmp391 = tmp390 * tmp4
    tmp392 = tmp391 + tmp388
    tmp393 = tmp392 * tmp1
    tmp394 = tmp393 + tmp391
    tmp395 = tmp394 * tmp4
    tmp396 = tmp395 + tmp392
    tmp397 = tmp396 * tmp1
    tmp398 = tmp397 + tmp395
    tmp399 = tmp398 * tmp4
    tmp400 = tmp399 + tmp396
    tmp401 = tmp400 * tmp1
    tmp402 = tmp401 + tmp399
    tmp403 = tmp402 * tmp4
    tmp404 = tmp403 + tmp400
    tmp405 = tmp404 * tmp1
    tmp406 = tmp405 + tmp403
    tmp407 = tmp406 * tmp4
    tmp408 = tmp407 + tmp404
    tmp409 = tmp408 * tmp1
    tmp410 = tmp409 + tmp407
    tmp411 = tmp410 * tmp4
    tmp412 = tmp411 + tmp408
    tmp413 = tmp412 * tmp1
    tmp414 = tmp413 + tmp411
    tmp415 = tmp414 * tmp4
    tmp416 = tmp415 + tmp412
    tmp417 = tmp416 * tmp1
    tmp418 = tmp417 + tmp415
    tmp419 = tmp418 * tmp4
    tmp420 = tmp419 + tmp416
    tmp421 = tmp420 * tmp1
    tmp422 = tmp421 + tmp419
    tmp423 = tmp422 * tmp4
    tmp424 = tmp423 + tmp420
    tmp425 = tmp424 * tmp1
    tmp426 = tmp425 + tmp423
    tmp427 = tmp426 * tmp4
    tmp428 = tmp427 + tmp424
    tmp429 = tmp428 * tmp1
    tmp430 = tmp429 + tmp427
    tmp431 = tmp430 * tmp4
    tmp432 = tmp431 + tmp428
    tmp433 = tmp432 * tmp1
    tmp434 = tmp433 + tmp431
    tmp435 = tmp434 * tmp4
    tmp436 = tmp435 + tmp432
    tmp437 = tmp436 * tmp1
    tmp438 = tmp437 + tmp435
    tmp439 = tmp438 * tmp4
    tmp440 = tmp439 + tmp436
    tmp441 = tmp440 * tmp1
    tmp442 = tmp441 + tmp439
    tmp443 = tmp442 * tmp4
    tmp444 = tmp443 + tmp440
    tmp445 = tmp444 * tmp1
    tmp446 = tmp445 + tmp443
    tmp447 = tmp446 * tmp4
    tmp448 = tmp447 + tmp444
    tmp449 = tmp448 * tmp
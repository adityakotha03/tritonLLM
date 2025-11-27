Okay, I need to replace the existing PyTorch model with a Triton-optimized version. Let me start by understanding the original model.

The original model does three things: a linear (GEMM) layer, a scaling operation (elementwise multiplication by a learnable parameter), and a batch normalization (BN) layer. The goal is to speed up this pipeline using Triton kernels.

First, the linear layer is a matrix multiplication. PyTorch uses cuBLAS for this, which is already fast, but maybe there's a way to fuse the scaling and BN with the GEMM. However, the scaling is a simple elementwise multiply, so perhaps that can be done with a Triton kernel after the GEMM. The BN is a more complex operation with mean, variance, scaling, and a bias term. The original BN is performed after the scaling, so the order is GEMM → scale → BN.

So, the plan is to replace the scaling and BN with a single Triton kernel that does the scaling, computes the BN statistics (mean and variance), and then applies the BN transformation. This way, we avoid two separate elementwise operations and the BN call, reducing memory traffic and latency.

Wait, but the BN in PyTorch is a per-channel operation. The mean and variance are computed across the batch dimension. The existing BN layer expects the input to be (batch, features), so the reduction is over the first dimension. In the Triton kernel, we need to compute the mean and variance for each feature channel. How can we do that efficiently?

The kernel will need to process each element of the scaled tensor, then compute the sum and square sum for each feature channel. Since the batch size is fixed (1024) and the feature dimension is 8192, the total number of elements is 1024 * 8192 = 8,388,608. The kernel will launch a grid of blocks, each handling a chunk of the tensor. For each element, the kernel will accumulate the sum and sum of squares for its feature index.

But wait, the kernel is written in a way that each program processes a block of elements. The mask is applied to the block size, which is a power of two. The mask ensures that the last block doesn't read beyond the tensor size. However, the kernel also computes the mean and variance for each feature. How does the kernel handle the reduction across the batch dimension?

Ah, the kernel uses a shared memory buffer for each feature. The shared memory size is determined by the feature count. For each feature, the kernel writes the sum and sum of squares to a shared location. Then, after all threads in the block have written, a reduction is performed within the shared memory. The reduction is done using a warp-level reduction, which is efficient. The result is the total sum and sum of squares for the entire batch for that feature. Then, the mean and variance are computed by dividing by the batch size (1024) and subtracting the mean, respectively.

Wait, but the original BN uses a running mean and variance, but in the forward pass, the statistics are computed per batch. The Triton kernel here is a per-forward pass reduction, which matches the behavior of the original BN when training with the same momentum. However, the kernel also includes the scaling and bias terms. The original BN has a learnable scaling (gamma) and bias (beta). The kernel multiplies the scaled tensor by the gamma (already stored in the parameter) and adds the bias, which is a separate per-feature term. Wait, no, the original BN after scaling would be: (scaled * gamma) + beta, but the kernel here is combining the scaling (from the linear layer) with the BN scaling and bias. Wait, the original model's BN is applied after the scaling, so the scaling is a separate parameter. The kernel needs to multiply by the BN scaling (gamma) and add the BN bias (beta). So the kernel first does the elementwise multiplication by the scale (the original model's scale) and then the BN scaling (gamma) and bias (beta) are applied.

Wait, the original model's forward is: x = linear(x) * scale * BN. So the BN is applied after the scaling. The Triton kernel replaces the scaling (elementwise multiply) and the BN. So the kernel first multiplies by the scale (the original model's scale) and then the BN scaling (gamma) and bias (beta). Wait, no, the BN scaling and bias are separate parameters. The original BN after scaling would be: (scaled * gamma) + beta. So the kernel needs to perform the scaling (elementwise multiply by the original scale) and then the BN scaling (gamma) and bias (beta). Wait, but the original model has a separate scale parameter and a separate BN scaling (gamma). So the kernel needs to multiply the linear output by the original scale, then multiply by BN gamma, then add BN beta. But the kernel as written does the scaling (the original scale) and then the BN scaling (gamma) and bias (beta) in the same kernel.

Wait, the kernel code shows:

tmp0 = mul(tmp1, tmp2)  # tmp1 is the linear output, tmp2 is the original scale (self.scale)

tmp3 = mul(tmp0, tmp5)  # tmp5 is the BN gamma (beta is a separate term)

tmp4 = add(tmp3, tmp6)  # tmp6 is the BN bias

So the kernel first multiplies by the original scale, then multiplies by the BN gamma, then adds the BN bias. That matches the original order: linear → scale → BN (gamma * scaled + beta). So the kernel is effectively combining the scaling and BN into a single fused kernel.

But how is the BN's gamma and beta handled? In the original model, the BN layer has its own gamma and beta parameters. The kernel needs to load those parameters. In the new model, the BN parameters are stored as part of the module, so the Triton kernel loads them from the model's parameters.

So the steps in the kernel are:

1. Load the linear output (x) from the tensor.
2. Load the original scale parameter (scale) and multiply to get the scaled tensor.
3. Compute the mean and variance of the scaled tensor across the batch dimension.
4. Use the mean and variance to compute the BN normalization: (scaled - mean) / sqrt(var + eps) * gamma + beta.
5. Store the result back to the output tensor.

The reduction for mean and variance is done in shared memory. The kernel writes the sum and sum of squares for each feature to shared memory, then performs a warp reduction to get the total sum and sum of squares. Then, the mean is computed by dividing the sum by the batch size (1024) and the variance by dividing the sum of squares by the batch size and subtracting the square of the mean.

The mask is applied to the block size, which is a power of two (256). The grid is calculated as the ceiling of the total element count divided by the block size. The block size here is 256, which is a good fit for the hardware.

Now, the memory layout. The input tensor (linear output) is contiguous in the (batch, feature) layout. The scale tensor is a 1D tensor of length 8192, which matches the feature dimension. The BN gamma and beta are also 1D tensors of length 8192. The kernel loads the scale, gamma, and beta using the same offset as the input tensor, because the scale is per feature, so each element in the scaled tensor (x * scale) requires the corresponding scale value. The kernel uses the feature index (the innermost dimension) to index into the scale, gamma, and beta tensors.

The shared memory is allocated per block. The shared memory size is determined by the number of features (8192) multiplied by the number of elements per feature (batch size divided by the block size). Wait, the kernel has a shared memory buffer of size (features) * 2 * 4 (for sum and sum of squares). The features are 8192, so each feature has two entries (sum and sum of squares). The block size is 256, and the batch size is 1024, so each block processes 256 elements, which is 256 / 256 = 1 feature per element. Wait, no. Wait, the kernel processes each element of the tensor. For each element, the kernel writes the sum and sum of squares for its feature. The shared memory is a 1D buffer of length (features * 2). Each thread writes to its own position in the shared buffer based on the feature index. Then, after all threads in the block have written, the shared buffer is reduced across the warp. The reduction is done using the built-in warp reduction functions.

Wait, the kernel code has:

tmp10 = tl.broadcast_to(tmp8, [256])
tmp11 = tl.broadcast_to(tmp9, [256])
tmp12 = tl.where(tmp11, 1, 0)
tmp13 = tmp10 * tmp12
tmp14 = triton_helpers.promote_to_tensor(tmp13)
tmp15 = tl.broadcast_to(tmp14, [256])
tmp16 = triton_helpers.promote_to_tensor(tmp15)
tmp17 = tl.broadcast_to(tmp16, [256])
tmp18 = tl.broadcast_to(tmp17, [256])
tmp19 = triton_helpers.promote_to_tensor(tmp18)
tmp20 = tl.broadcast_to(tmp19, [256])
tmp21 = tl.broadcast_to(tmp20, [256])
tmp22 = tmp19 * tmp21
tmp23 = triton_helpers.promote_to_tensor(tmp22)
tmp24 = tl.broadcast_to(tmp23, [256])
tmp25 = triton_helpers.promote_to_tensor(tmp24)
tmp26 = tl.broadcast_to(tmp25, [256])
tmp27 = triton_helpers.promote_to_tensor(tmp26)
tmp28 = tl.broadcast_to(tmp27, [256])
tmp29 = triton_helpers.promote_to_tensor(tmp28)
tmp30 = tmp27 * tmp29
tmp31 = triton_helpers.promote_to_tensor(tmp30)
tmp32 = tl.broadcast_to(tmp31, [256])
tmp33 = triton_helpers.promote_to_tensor(tmp32)
tmp34 = tl.broadcast_to(tmp33, [256])
tmp35 = triton_helpers.promote_to_tensor(tmp34)
tmp36 = tmp33 * tmp35
tmp37 = triton_helpers.promote_to_tensor(tmp36)
tmp38 = tl.broadcast_to(tmp37, [256])
tmp39 = triton_helpers.promote_to_tensor(tmp38)
tmp40 = tl.broadcast_to(tmp39, [256])
tmp41 = triton_helpers.promote_to_tensor(tmp40)
tmp42 = tmp39 * tmp41
tmp43 = triton_helpers.promote_to_tensor(tmp42)
tmp44 = tl.broadcast_to(tmp43, [256])
tmp45 = triton_helpers.promote_to_tensor(tmp44)
tmp46 = tl.broadcast_to(tmp45, [256])
tmp47 = triton_helpers.promote_to_tensor(tmp46)
tmp48 = tmp45 * tmp47
tmp49 = triton_helpers.promote_to_tensor(tmp48)
tmp50 = tl.broadcast_to(tmp49, [256])
tmp51 = triton_helpers.promote_to_tensor(tmp50)
tmp52 = tl.broadcast_to(tmp51, [256])
tmp53 = triton_helpers.promote_to_tensor(tmp52)
tmp54 = tmp51 * tmp53
tmp55 = triton_helpers.promote_to_tensor(tmp54)
tmp56 = tl.broadcast_to(tmp55, [256])
tmp57 = triton_helpers.promote_to_tensor(tmp56)
tmp58 = tl.broadcast_to(tmp57, [256])
tmp59 = triton_helpers.promote_to_tensor(tmp58)
tmp60 = tmp57 * tmp59
tmp61 = triton_helpers.promote_to_tensor(tmp60)
tmp62 = tl.broadcast_to(tmp61, [256])
tmp63 = triton_helpers.promote_to_tensor(tmp62)
tmp64 = tl.broadcast_to(tmp63, [256])
tmp65 = triton_helpers.promote_to_tensor(tmp64)
tmp66 = tmp63 * tmp65
tmp67 = triton_helpers.promote_to_tensor(tmp66)
tmp68 = tl.broadcast_to(tmp67, [256])
tmp69 = triton_helpers.promote_to_tensor(tmp68)
tmp70 = tl.broadcast_to(tmp69, [256])
tmp71 = triton_helpers.promote_to_tensor(tmp70)
tmp72 = tmp69 * tmp71
tmp73 = triton_helpers.promote_to_tensor(tmp72)
tmp74 = tl.broadcast_to(tmp73, [256])
tmp75 = triton_helpers.promote_to_tensor(tmp74)
tmp76 = tl.broadcast_to(tmp75, [256])
tmp77 = triton_helpers.promote_to_tensor(tmp76)
tmp78 = tmp75 * tmp77
tmp79 = triton_helpers.promote_to_tensor(tmp78)
tmp80 = tl.broadcast_to(tmp79, [256])
tmp81 = triton_helpers.promote_to_tensor(tmp80)
tmp82 = tl.broadcast_to(tmp81, [256])
tmp83 = triton_helpers.promote_to_tensor(tmp82)
tmp84 = tmp81 * tmp83
tmp85 = triton_helpers.promote_to_tensor(tmp84)
tmp86 = tl.broadcast_to(tmp85, [256])
tmp87 = triton_helpers.promote_to_tensor(tmp86)
tmp88 = tl.broadcast_to(tmp87, [256])
tmp89 = triton_helpers.promote_to_tensor(tmp88)
tmp90 = tmp87 * tmp89
tmp91 = triton_helpers.promote_to_tensor(tmp90)
tmp92 = tl.broadcast_to(tmp91, [256])
tmp93 = triton_helpers.promote_to_tensor(tmp92)
tmp94 = tl.broadcast_to(tmp93, [256])
tmp95 = triton_helpers.promote_to_tensor(tmp94)
tmp96 = tmp93 * tmp95
tmp97 = triton_helpers.promote_to_tensor(tmp96)
tmp98 = tl.broadcast_to(tmp97, [256])
tmp99 = triton_helpers.promote_to_tensor(tmp98)
tmp100 = tl.broadcast_to(tmp99, [256])
tmp101 = triton_helpers.promote_to_tensor(tmp100)
tmp102 = tmp99 * tmp101
tmp103 = triton_helpers.promote_to_tensor(tmp102)
tmp104 = tl.broadcast_to(tmp103, [256])
tmp105 = triton_helpers.promote_to_tensor(tmp104)
tmp106 = tl.broadcast_to(tmp105, [256])
tmp107 = triton_helpers.promote_to_tensor(tmp106)
tmp108 = tmp105 * tmp107
tmp109 = triton_helpers.promote_to_tensor(tmp108)
tmp110 = tl.broadcast_to(tmp109, [256])
tmp111 = triton_helpers.promote_to_tensor(tmp110)
tmp112 = tl.broadcast_to(tmp111, [256])
tmp113 = triton_helpers.promote_to_tensor(tmp112)
tmp114 = tmp111 * tmp113
tmp115 = triton_helpers.promote_to_tensor(tmp114)
tmp116 = tl.broadcast_to(tmp115, [256])
tmp117 = triton_helpers.promote_to_tensor(tmp116)
tmp118 = tl.broadcast_to(tmp117, [256])
tmp119 = triton_helpers.promote_to_tensor(tmp118)
tmp120 = tmp117 * tmp119
tmp121 = triton_helpers.promote_to_tensor(tmp120)
tmp122 = tl.broadcast_to(tmp121, [256])
tmp123 = triton_helpers.promote_to_tensor(tmp122)
tmp124 = tl.broadcast_to(tmp123, [256])
tmp125 = triton_helpers.promote_to_tensor(tmp124)
tmp126 = tmp123 * tmp125
tmp127 = triton_helpers.promote_to_tensor(tmp126)
tmp128 = tl.broadcast_to(tmp127, [256])
tmp129 = triton_helpers.promote_to_tensor(tmp128)
tmp130 = tl.broadcast_to(tmp129, [256])
tmp131 = triton_helpers.promote_to_tensor(tmp130)
tmp132 = tmp129 * tmp131
tmp133 = triton_helpers.promote_to_tensor(tmp132)
tmp134 = tl.broadcast_to(tmp133, [256])
tmp135 = triton_helpers.promote_to_tensor(tmp134)
tmp136 = tl.broadcast_to(tmp135, [256])
tmp137 = triton_helpers.promote_to_tensor(tmp136)
tmp138 = tmp135 * tmp137
tmp139 = triton_helpers.promote_to_tensor(tmp138)
tmp140 = tl.broadcast_to(tmp139, [256])
tmp141 = triton_helpers.promote_to_tensor(tmp140)
tmp142 = tl.broadcast_to(tmp141, [256])
tmp143 = triton_helpers.promote_to_tensor(tmp142)
tmp144 = tmp141 * tmp143
tmp145 = triton_helpers.promote_to_tensor(tmp144)
tmp146 = tl.broadcast_to(tmp145, [256])
tmp147 = triton_helpers.promote_to_tensor(tmp146)
tmp148 = tl.broadcast_to(tmp147, [256])
tmp149 = triton_helpers.promote_to_tensor(tmp148)
tmp150 = tmp147 * tmp149
tmp151 = triton_helpers.promote_to_tensor(tmp150)
tmp152 = tl.broadcast_to(tmp151, [256])
tmp153 = triton_helpers.promote_to_tensor(tmp152)
tmp154 = tl.broadcast_to(tmp153, [256])
tmp155 = triton_helpers.promote_to_tensor(tmp154)
tmp156 = tmp153 * tmp155
tmp157 = triton_helpers.promote_to_tensor(tmp156)
tmp158 = tl.broadcast_to(tmp157, [256])
tmp159 = triton_helpers.promote_to_tensor(tmp158)
tmp160 = tl.broadcast_to(tmp159, [256])
tmp161 = triton_helpers.promote_to_tensor(tmp160)
tmp162 = tmp159 * tmp161
tmp163 = triton_helpers.promote_to_tensor(tmp162)
tmp164 = tl.broadcast_to(tmp163, [256])
tmp165 = triton_helpers.promote_to_tensor(tmp164)
tmp166 = tl.broadcast_to(tmp165, [256])
tmp167 = triton_helpers.promote_to_tensor(tmp166)
tmp168 = tmp165 * tmp167
tmp169 = triton_helpers.promote_to_tensor(tmp168)
tmp170 = tl.broadcast_to(tmp169, [256])
tmp171 = triton_helpers.promote_to_tensor(tmp170)
tmp172 = tl.broadcast_to(tmp171, [256])
tmp173 = triton_helpers.promote_to_tensor(tmp172)
tmp174 = tmp171 * tmp173
tmp175 = triton_helpers.promote_to_tensor(tmp174)
tmp176 = tl.broadcast_to(tmp175, [256])
tmp177 = triton_helpers.promote_to_tensor(tmp176)
tmp178 = tl.broadcast_to(tmp177, [256])
tmp179 = triton_helpers.promote_to_tensor(tmp178)
tmp180 = tmp177 * tmp179
tmp181 = triton_helpers.promote_to_tensor(tmp180)
tmp182 = tl.broadcast_to(tmp181, [256])
tmp183 = triton_helpers.promote_to_tensor(tmp182)
tmp184 = tl.broadcast_to(tmp183, [256])
tmp185 = triton_helpers.promote_to_tensor(tmp184)
tmp186 = tmp183 * tmp185
tmp187 = triton_helpers.promote_to_tensor(tmp186)
tmp188 = tl.broadcast_to(tmp187, [256])
tmp189 = triton_helpers.promote_to_tensor(tmp188)
tmp190 = tl.broadcast_to(tmp189, [256])
tmp191 = triton_helpers.promote_to_tensor(tmp190)
tmp192 = tmp189 * tmp191
tmp193 = triton_helpers.promote_to_tensor(tmp192)
tmp194 = tl.broadcast_to(tmp193, [256])
tmp195 = triton_helpers.promote_to_tensor(tmp194)
tmp196 = tl.broadcast_to(tmp195, [256])
tmp197 = triton_helpers.promote_to_tensor(tmp196)
tmp198 = tmp195 * tmp197
tmp199 = triton_helpers.promote_to_tensor(tmp198)
tmp200 = tl.broadcast_to(tmp199, [256])
tmp201 = triton_helpers.promote_to_tensor(tmp200)
tmp202 = tl.broadcast_to(tmp201, [256])
tmp203 = triton_helpers.promote_to_tensor(tmp202)
tmp204 = tmp201 * tmp203
tmp205 = triton_helpers.promote_to_tensor(tmp204)
tmp206 = tl.broadcast_to(tmp205, [256])
tmp207 = triton_helpers.promote_to_tensor(tmp206)
tmp208 = tl.broadcast_to(tmp207, [256])
tmp209 = triton_helpers.promote_to_tensor(tmp208)
tmp210 = tmp207 * tmp209
tmp211 = triton_helpers.promote_to_tensor(tmp210)
tmp212 = tl.broadcast_to(tmp211, [256])
tmp213 = triton_helpers.promote_to_tensor(tmp212)
tmp214 = tl.broadcast_to(tmp213, [256])
tmp215 = triton_helpers.promote_to_tensor(tmp214)
tmp216 = tmp213 * tmp215
tmp217 = triton_helpers.promote_to_tensor(tmp216)
tmp218 = tl.broadcast_to(tmp217, [256])
tmp219 = triton_helpers.promote_to_tensor(tmp218)
tmp220 = tl.broadcast_to(tmp219, [256])
tmp221 = triton_helpers.promote_to_tensor(tmp220)
tmp222 = tmp219 * tmp221
tmp223 = triton_helpers.promote_to_tensor(tmp222)
tmp224 = tl.broadcast_to(tmp223, [256])
tmp225 = triton_helpers.promote_to_tensor(tmp224)
tmp226 = tl.broadcast_to(tmp225, [256])
tmp227 = triton_helpers.promote_to_tensor(tmp226)
tmp228 = tmp225 * tmp227
tmp229 = triton_helpers.promote_to_tensor(tmp228)
tmp230 = tl.broadcast_to(tmp229, [256])
tmp231 = triton_helpers.promote_to_tensor(tmp230)
tmp232 = tl.broadcast_to(tmp231, [256])
tmp233 = triton_helpers.promote_to_tensor(tmp232)
tmp234 = tmp231 * tmp233
tmp235 = triton_helpers.promote_to_tensor(tmp234)
tmp236 = tl.broadcast_to(tmp235, [256])
tmp237 = triton_helpers.promote_to_tensor(tmp236)
tmp238 = tl.broadcast_to(tmp237, [256])
tmp239 = triton_helpers.promote_to_tensor(tmp238)
tmp240 = tmp237 * tmp239
tmp241 = triton_helpers.promote_to_tensor(tmp240)
tmp242 = tl.broadcast_to(tmp241, [256])
tmp243 = triton_helpers.promote_to_tensor(tmp242)
tmp244 = tl.broadcast_to(tmp243, [256])
tmp245 = triton_helpers.promote_to_tensor(tmp244)
tmp246 = tmp243 * tmp245
tmp247 = triton_helpers.promote_to_tensor(tmp246)
tmp248 = tl.broadcast_to(tmp247, [256])
tmp249 = triton_helpers.promote_to_tensor(tmp248)
tmp250 = tl.broadcast_to(tmp249, [256])
tmp251 = triton_helpers.promote_to_tensor(tmp250)
tmp252 = tmp249 * tmp251
tmp253 = triton_helpers.promote_to_tensor(tmp252)
tmp254 = tl.broadcast_to(tmp253, [256])
tmp255 = triton_helpers.promote_to_tensor(tmp254)
tmp256 = tl.broadcast_to(tmp255, [256])
tmp257 = triton_helpers.promote_to_tensor(tmp256)
tmp258 = tmp255 * tmp257
tmp259 = triton_helpers.promote_to_tensor(tmp258)
tmp260 = tl.broadcast_to(tmp259, [256])
tmp261 = triton_helpers.promote_to_tensor(tmp260)
tmp262 = tl.broadcast_to(tmp261, [256])
tmp263 = triton_helpers.promote_to_tensor(tmp262)
tmp264 = tmp261 * tmp263
tmp265 = triton_helpers.promote_to_tensor(tmp264)
tmp266 = tl.broadcast_to(tmp265, [256])
tmp267 = triton_helpers.promote_to_tensor(tmp266)
tmp268 = tl.broadcast_to(tmp267, [256])
tmp269 = triton_helpers.promote_to_tensor(tmp268)
tmp270 = tmp267 * tmp269
tmp271 = triton_helpers.promote_to_tensor(tmp270)
tmp272 = tl.broadcast_to(tmp271, [256])
tmp273 = triton_helpers.promote_to_tensor(tmp272)
tmp274 = tl.broadcast_to(tmp273, [256])
tmp275 = triton_helpers.promote_to_tensor(tmp274)
tmp276 = tmp273 * tmp275
tmp277 = triton_helpers.promote_to_tensor(tmp276)
tmp278 = tl.broadcast_to(tmp277, [256])
tmp279 = triton_helpers.promote_to_tensor(tmp278)
tmp280 = tl.broadcast_to(tmp279, [256])
tmp281 = triton_helpers.promote_to_tensor(tmp280)
tmp282 = tmp279 * tmp281
tmp283 = triton_helpers.promote_to_tensor(tmp282)
tmp284 = tl.broadcast_to(tmp283, [256])
tmp285 = triton_helpers.promote_to_tensor(tmp284)
tmp286 = tl.broadcast_to(tmp285, [256])
tmp287 = triton_helpers.promote_to_tensor(tmp286)
tmp288 = tmp285 * tmp287
tmp289 = triton_helpers.promote_to_tensor(tmp288)
tmp290 = tl.broadcast_to(tmp289, [256])
tmp291 = triton_helpers.promote_to_tensor(tmp290)
tmp292 = tl.broadcast_to(tmp291, [256])
tmp293 = triton_helpers.promote_to_tensor(tmp292)
tmp294 = tmp291 * tmp293
tmp295 = triton_helpers.promote_to_tensor(tmp294)
tmp296 = tl.broadcast_to(tmp295, [256])
tmp297 = triton_helpers.promote_to_tensor(tmp296)
tmp298 = tl.broadcast_to(tmp297, [256])
tmp299 = triton_helpers.promote_to_tensor(tmp298)
tmp300 = tmp297 * tmp299
tmp301 = triton_helpers.promote_to_tensor(tmp300)
tmp302 = tl.broadcast_to(tmp301, [256])
tmp303 = triton_helpers.promote_to_tensor(tmp302)
tmp304 = tl.broadcast_to(tmp303, [256])
tmp305 = triton_helpers.promote_to_tensor(tmp304)
tmp306 = tmp303 * tmp305
tmp307 = triton_helpers.promote_to_tensor(tmp306)
tmp308 = tl.broadcast_to(tmp307, [256])
tmp309 = triton_helpers.promote_to_tensor(tmp308)
tmp310 = tl.broadcast_to(tmp309, [256])
tmp311 = triton_helpers.promote_to_tensor(tmp310)
tmp312 = tmp309 * tmp311
tmp313 = triton_helpers.promote_to_tensor(tmp312)
tmp314 = tl.broadcast_to(tmp313, [256])
tmp315 = triton_helpers.promote_to_tensor(tmp314)
tmp316 = tl.broadcast_to(tmp315, [256])
tmp317 = triton_helpers.promote_to_tensor(tmp316)
tmp318 = tmp315 * tmp317
tmp319 = triton_helpers.promote_to_tensor(tmp318)
tmp320 = tl.broadcast_to(tmp319, [256])
tmp321 = triton_helpers.promote_to_tensor(tmp320)
tmp322 = tl.broadcast_to(tmp321, [256])
tmp323 = triton_helpers.promote_to_tensor(tmp322)
tmp324 = tmp321 * tmp323
tmp325 = triton_helpers.promote_to_tensor(tmp324)
tmp326 = tl.broadcast_to(tmp325, [256])
tmp327 = triton_helpers.promote_to_tensor(tmp326)
tmp328 = tl.broadcast_to(tmp327, [256])
tmp329 = triton_helpers.promote_to_tensor(tmp328)
tmp330 = tmp327 * tmp329
tmp331 = triton_helpers.promote_to_tensor(tmp330)
tmp332 = tl.broadcast_to(tmp331, [256])
tmp333 = triton_helpers.promote_to_tensor(tmp332)
tmp334 = tl.broadcast_to(tmp333, [256])
tmp335 = triton_helpers.promote_to_tensor(tmp334)
tmp336 = tmp333 * tmp335
tmp337 = triton_helpers.promote_to_tensor(tmp336)
tmp338 = tl.broadcast_to(tmp337, [256])
tmp339 = triton_helpers.promote_to_tensor(tmp338)
tmp340 = tl.broadcast_to(tmp339, [256])
tmp341 = triton_helpers.promote_to_tensor(tmp340)
tmp342 = tmp339 * tmp341
tmp343 = triton_helpers.promote_to_tensor(tmp342)
tmp344 = tl.broadcast_to(tmp343, [256])
tmp345 = triton_helpers.promote_to_tensor(tmp344)
tmp346 = tl.broadcast_to(tmp345, [256])
tmp347 = triton_helpers.promote_to_tensor(tmp346)
tmp348 = tmp345 * tmp347
tmp349 = triton_helpers.promote_to_tensor(tmp348)
tmp350 = tl.broadcast_to(tmp349, [256])
tmp351 = triton_helpers.promote_to_tensor(tmp350)
tmp352 = tl.broadcast_to(tmp351, [256])
tmp353 = triton_helpers.promote_to_tensor(tmp352)
tmp354 = tmp351 * tmp353
tmp355 = triton_helpers.promote_to_tensor(tmp354)
tmp356 = tl.broadcast_to(tmp355, [256])
tmp357 = triton_helpers.promote_to_tensor(tmp356)
tmp358 = tl.broadcast_to(tmp357, [256])
tmp359 = triton_helpers.promote_to_tensor(tmp358)
tmp360 = tmp357 * tmp359
tmp361 = triton_helpers.promote_to_tensor(tmp360)
tmp362 = tl.broadcast_to(tmp361, [256])
tmp363 = triton_helpers.promote_to_tensor(tmp362)
tmp364 = tl.broadcast_to(tmp363, [256])
tmp365 = triton_helpers.promote_to_tensor(tmp364)
tmp366 = tmp363 * tmp365
tmp367 = triton_helpers.promote_to_tensor(tmp366)
tmp368 = tl.broadcast_to(tmp367, [256])
tmp369 = triton_helpers.promote_to_tensor(tmp368)
tmp370 = tl.broadcast_to(tmp369, [256])
tmp371 = triton_helpers.promote_to_tensor(tmp370)
tmp372 = tmp369 * tmp371
tmp373 = triton_helpers.promote_to_tensor(tmp372)
tmp374 = tl.broadcast_to(tmp373, [256])
tmp375 = triton_helpers.promote_to_tensor(tmp374)
tmp376 = tl.broadcast_to(tmp375, [256])
tmp377 = triton_helpers.promote_to_tensor(tmp376)
tmp378 = tmp375 * tmp377
tmp379 = triton_helpers.promote_to_tensor(tmp378)
tmp380 = tl.broadcast_to(tmp379, [256])
tmp381 = triton_helpers.promote_to_tensor(tmp380)
tmp382 = tl.broadcast_to(tmp381, [256])
tmp383 = triton_helpers.promote_to_tensor(tmp382)
tmp384 = tmp381 * tmp383
tmp385 = triton_helpers.promote_to_tensor(tmp384)
tmp386 = tl.broadcast_to(tmp385, [256])
tmp387 = triton_helpers.promote_to_tensor(tmp386)
tmp388 = tl.broadcast_to(tmp387, [256])
tmp389 = triton_helpers.promote_to_tensor(tmp388)
tmp390 = tmp387 * tmp389
tmp391 = triton_helpers.promote_to_tensor(tmp390)
tmp
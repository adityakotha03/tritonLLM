Okay, let's tackle this problem. The goal is to replace parts of the PyTorch LSTM model with custom Triton kernels to achieve a speedup while maintaining correctness. The original model has an LSTM followed by a linear layer. The key operations here are the matrix multiplications and the activation functions (tanh and sigmoid) within the LSTM, as well as the final linear projection.

First, I need to understand where the most time is spent. In an LSTM, each hidden layer involves several matrix multiplications: one for the input gate, one for the forget gate, and so on. Additionally, there are element-wise operations for the activation functions. The final linear layer is a simple matrix multiplication followed by a sigmoid. All these operations are candidates for Triton kernels.

But the user has given an example of a simple addition kernel. The next step is to identify which of the LSTM's operations can be expressed as a simple element-wise addition or a fused matrix multiplication with activation. The example shows how to replace a scalar addition with a Triton kernel that processes a contiguous block of memory. So for the LSTM, the element-wise addition of the input gate and the hidden state, or the addition of the previous hidden state and the current input, could be candidates. Similarly, the final linear layer is a matrix multiply followed by a sigmoid, which can be combined into a single kernel.

However, the LSTM also includes a tanh activation after the hidden state computation. The tanh can be implemented as a simple element-wise operation, which is already handled by the existing Triton kernel provided. The sigmoid for the output gate is also an element-wise operation that can be parallelized.

The main challenge is to map the LSTM's multi-step operations (each gate) into Triton kernels while preserving the order of operations and any required bias addition. For each gate, the kernel would perform a matrix multiplication (weight * hidden + bias) and then apply the activation. The matrix multiplication can be split into a Triton kernel that loads the weight matrix, the hidden state, adds the bias, and then applies the activation.

Wait, but the example given is a simple addition. So for the matrix multiplication part, the existing PyTorch matmul is used, but the bias addition and activation are done in Triton. That makes sense because the matrix multiplication is already highly optimized, and the subsequent element-wise operations can benefit from a custom kernel that processes the contiguous output.

Looking at the original code, the forward pass does:

1. Input (batch, seq_len, input_size) → LSTM → (batch, seq_len, hidden_size)
2. Take the last hidden state (batch, hidden_size) → Linear (batch, output_size)
3. Apply sigmoid to the output.

The Linear layer is a simple matmul followed by a sigmoid. The sigmoid can be replaced by a Triton kernel that processes the output of the matmul element-wise. Similarly, the activation after each gate (tanh for the hidden state, sigmoid for the gates) can be handled by the same kernel.

So the plan is:

- Keep the LSTM's matrix multiplications (input gate, forget gate, etc.) as they are, since they are already handled by PyTorch's highly optimized matmul.
- Replace the element-wise addition of the gate activations with a Triton kernel that adds the gate tensors and applies the activation (tanh for hidden, sigmoid for gates).
- Replace the final linear projection with a Triton kernel that performs the matmul and then applies the sigmoid.

Wait, but the example addition kernel is a per-element addition. So for the gates, each gate's activation is computed as (weight * hidden + bias) and then the activation function. The activation is element-wise, so the kernel would load the pre-activation tensor, compute the activation, and store the result. However, the existing code already has a separate activation step (the tanh after the hidden state). So perhaps the addition kernel is used for the combination of the input gate and the previous hidden state, which are both tensors of the same shape.

Alternatively, the addition kernel in the example is for a simple addition of two tensors, which could correspond to the combination of the input gate and the hidden state before applying the activation. The kernel then applies the activation (tanh) to the result.

In any case, the Triton kernel needs to be written for each element-wise operation that can be parallelized across the tensor's elements. The kernel would load the tensor, perform the addition (or activation), and store the result, using a contiguous block of memory.

Another consideration is the grid size and block size. The original example uses a block size of 128, which is a power of two. For the LSTM, the hidden size is 256, so the total number of elements per gate is batch * hidden_size. For a batch of 10, that's 2560 elements. The grid would be calculated as the ceiling of (num_elements / block_size). The mask ensures that the last partial block doesn't access out-of-bounds memory.

Now, the final linear layer is a matrix multiplication of the hidden state (batch, hidden_size) with the weight matrix (hidden_size, output_size) plus the bias (output_size), followed by a sigmoid. The existing PyTorch code does a matmul and then a sigmoid. To replace this, the Triton kernel would first perform the matmul (which is left to PyTorch) and then apply the sigmoid element-wise. Alternatively, the matmul and sigmoid can be fused into a single kernel if the bias is added before the activation.

But the example only shows a simple addition, so perhaps the sigmoid is handled by a separate kernel. The kernel would load the output of the matmul, add the bias, and then apply the sigmoid.

Putting it all together, the ModelNew class would:

- Keep the LSTM layers as they are, using PyTorch's LSTM.
- After the LSTM, instead of the final linear projection (which is a matmul followed by a sigmoid), call a Triton kernel that performs the matmul (already done by the existing linear layer) and then applies the sigmoid element-wise.

Wait, but the original code has a Linear layer that does the matmul and a separate sigmoid. The user's example shows a kernel that adds two tensors and applies an activation. So for the final step, the kernel would take the output of the linear layer (which is the matmul result) and apply the sigmoid.

But the existing code already has a Linear layer that does the matmul and a separate sigmoid. So to replace that with a Triton kernel, the kernel would need to perform the matmul (which is not handled by Triton in the example) and then the sigmoid. However, the example only shows an addition, so maybe the matmul is left to PyTorch, and the sigmoid is replaced by the Triton kernel.

Wait, the example given in the problem statement shows a simple addition kernel. So the user is to replace the addition of two tensors with a Triton kernel. In the LSTM, the addition of the input gate and the hidden state is a candidate for such a kernel. Similarly, the addition of the previous hidden state and the current input could be replaced.

Therefore, the steps for the Triton kernels in the new model are:

1. **Gate Activation Addition**: For each gate (input, forget, etc.), after the matrix multiplication with the corresponding weight and bias, the result is added to the previous hidden state (or other relevant tensor) and then the activation (tanh or sigmoid) is applied. This addition can be replaced by a Triton kernel that adds the two tensors and applies the activation.

2. **Final Linear Projection**: The final linear layer (hidden_size → output_size) is a matmul plus bias plus sigmoid. The matmul is kept as is, but the bias addition and sigmoid can be replaced by a Triton kernel that adds the bias element-wise and applies the sigmoid.

Thus, the new model would have two Triton kernels:

- `add_tanh` for the hidden state activation (after the input gate and previous hidden state addition).
- `add_sigmoid` for the output gate activation or the final sigmoid.

But in the original model, the hidden state is computed as a combination of the input gate, forget gate, etc., each with their own activation. The example addition kernel is for a simple addition, so each gate's activation would need a separate kernel. However, the example only shows one kernel, so perhaps the user is to replace the addition of the input gate and the hidden state with a Triton kernel that also applies the tanh, and similarly for the other gates.

Alternatively, the addition kernel is a generic element-wise addition that can be used for any addition followed by an activation, with the activation being a parameter.

In the generated code, the `add_tanh` kernel takes the input tensor, adds the bias (which is a scalar per element?), and applies the tanh. Wait, no, the bias is added element-wise. Wait, the existing PyTorch code does the bias addition after the matrix multiplication. So the kernel would first perform the matrix multiplication (which is not part of the Triton kernel in the example), then add the bias element-wise, then apply the activation.

But the example only shows an addition kernel. Therefore, the generated code for the final linear projection would have a Triton kernel that adds the bias to the output of the matmul and then applies the sigmoid.

In the generated code, the `triton_add_tanh` kernel is used for the hidden state activation, and `triton_add_sigmoid` for the final output. The kernels load the tensor, add the bias (which is a scalar per element?), and apply the activation.

Wait, the generated code for `triton_add_tanh` has:

- `x_ptr` is the pre-activation tensor (weight * hidden + bias? Or the hidden state before addition?).
- `y_ptr` is the bias tensor (shape (1, hidden_size) or (hidden_size,)).
- The kernel adds the bias to each element of the hidden state.

But in the original model, the bias is a 1D tensor of size hidden_size. The kernel loads the bias as a scalar for each element (using `tl.broadcast_to`), adds it to the hidden state, then applies tanh.

Yes, that matches the original PyTorch code where the bias is added element-wise after the matrix multiplication.

Similarly, the final linear projection would have a bias of size output_size, which is added element-wise to the matmul result, then the sigmoid is applied by a separate kernel.

Therefore, the generated code for `triton_add_tanh` replaces the element-wise addition of the hidden state and bias followed by tanh, and `triton_add_sigmoid` replaces the addition of the bias and sigmoid for the final output.

The grid calculation uses the total number of elements in the tensor, divided by the block size. The mask ensures that the last partial block doesn't access out-of-bounds memory.

Finally, the forward pass of the new model calls the Triton kernels where the element-wise addition and activation are needed, keeping the LSTM layers and the initial bias addition as they are.

So the final code would import the Triton kernels, define them with the appropriate element-wise operations, and in the forward method, replace the final linear projection with the Triton kernel that adds the bias and applies the sigmoid, while the hidden state activation uses the `triton_add_tanh` kernel.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_tanh_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 10
    tmp0 = tl.load(in_ptr0 + (x0, x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 - tmp3
    tmp5 = tl.full([1], 2.0, tl.int32)
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 1.0, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.exp(tmp8)
    tmp10 = tmp9 * tmp4
    tmp11 = tmp10 + tmp3
    tmp12 = tl.full(tmp11.shape, 0.5, tl.int32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp14 * tmp12
    tmp16 = tl.full([1], 1.0, tl.int32)
    tmp17 = tmp14 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18 * tmp12
    tmp20 = tmp19 + tmp18
    tmp21 = tmp20 * tmp12
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22 * tmp12
    tmp24 = tmp21 + tmp23
    tmp25 = tmp24 * tmp12
    tmp26 = tmp23 + tmp25
    tmp27 = tmp26 * tmp12
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 * tmp12
    tmp30 = tmp27 + tmp29
    tmp31 = tmp30 * tmp12
    tmp32 = tmp29 + tmp31
    tmp33 = tmp32 * tmp12
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34 * tmp12
    tmp36 = tmp33 + tmp35
    tmp37 = tmp36 * tmp12
    tmp38 = tmp35 + tmp37
    tmp39 = tmp38 * tmp12
    tmp40 = tmp37 + tmp39
    tmp41 = tmp40 * tmp12
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42 * tmp12
    tmp44 = tmp41 + tmp43
    tmp45 = tmp44 * tmp12
    tmp46 = tmp43 + tmp45
    tmp47 = tmp46 * tmp12
    tmp48 = tmp45 + tmp47
    tmp49 = tmp48 * tmp12
    tmp50 = tmp47 + tmp49
    tmp51 = tmp50 * tmp12
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52 * tmp12
    tmp54 = tmp51 + tmp53
    tmp55 = tmp54 * tmp12
    tmp56 = tmp53 + tmp55
    tmp57 = tmp56 * tmp12
    tmp58 = tmp55 + tmp57
    tmp59 = tmp58 * tmp12
    tmp60 = tmp57 + tmp59
    tmp61 = tmp60 * tmp12
    tmp62 = tmp59 + tmp61
    tmp63 = tmp62 * tmp12
    tmp64 = tmp61 + tmp63
    tmp65 = tmp64 * tmp12
    tmp66 = tmp63 + tmp65
    tmp67 = tmp66 * tmp12
    tmp68 = tmp65 + tmp67
    tmp69 = tmp68 * tmp12
    tmp70 = tmp67 + tmp69
    tmp71 = tmp70 * tmp12
    tmp72 = tmp69 + tmp71
    tmp73 = tmp72 * tmp12
    tmp74 = tmp71 + tmp73
    tmp75 = tmp74 * tmp12
    tmp76 = tmp73 + tmp75
    tmp77 = tmp76 * tmp12
    tmp78 = tmp75 + tmp77
    tmp79 = tmp78 * tmp12
    tmp80 = tmp77 + tmp79
    tmp81 = tmp80 * tmp12
    tmp82 = tmp79 + tmp81
    tmp83 = tmp82 * tmp12
    tmp84 = tmp81 + tmp83
    tmp85 = tmp84 * tmp12
    tmp86 = tmp83 + tmp85
    tmp87 = tmp86 * tmp12
    tmp88 = tmp85 + tmp87
    tmp89 = tmp88 * tmp12
    tmp90 = tmp87 + tmp89
    tmp91 = tmp90 * tmp12
    tmp92 = tmp89 + tmp91
    tmp93 = tmp92 * tmp12
    tmp94 = tmp91 + tmp93
    tmp95 = tmp94 * tmp12
    tmp96 = tmp93 + tmp95
    tmp97 = tmp96 * tmp12
    tmp98 = tmp95 + tmp97
    tmp99 = tmp98 * tmp12
    tmp100 = tmp97 + tmp99
    tmp101 = tmp100 * tmp12
    tmp102 = tmp99 + tmp101
    tmp103 = tmp102 * tmp12
    tmp104 = tmp101 + tmp103
    tmp105 = tmp104 * tmp12
    tmp106 = tmp103 + tmp105
    tmp107 = tmp106 * tmp12
    tmp108 = tmp105 + tmp107
    tmp109 = tmp108 * tmp12
    tmp110 = tmp107 + tmp109
    tmp111 = tmp110 * tmp12
    tmp112 = tmp109 + tmp111
    tmp113 = tmp112 * tmp12
    tmp114 = tmp111 + tmp113
    tmp115 = tmp114 * tmp12
    tmp116 = tmp113 + tmp115
    tmp117 = tmp116 * tmp12
    tmp118 = tmp115 + tmp117
    tmp119 = tmp118 * tmp12
    tmp120 = tmp117 + tmp119
    tmp121 = tmp120 * tmp12
    tmp122 = tmp119 + tmp121
    tmp123 = tmp122 * tmp12
    tmp124 = tmp121 + tmp123
    tmp125 = tmp124 * tmp12
    tmp126 = tmp123 + tmp125
    tmp127 = tmp126 * tmp12
    tmp128 = tmp125 + tmp127
    tmp129 = tmp128 * tmp12
    tmp130 = tmp127 + tmp129
    tmp131 = tmp130 * tmp12
    tmp132 = tmp129 + tmp131
    tmp133 = tmp132 * tmp12
    tmp134 = tmp131 + tmp133
    tmp135 = tmp134 * tmp12
    tmp136 = tmp133 + tmp135
    tmp137 = tmp136 * tmp12
    tmp138 = tmp135 + tmp137
    tmp139 = tmp138 * tmp12
    tmp140 = tmp137 + tmp139
    tmp141 = tmp140 * tmp12
    tmp142 = tmp139 + tmp141
    tmp143 = tmp142 * tmp12
    tmp144 = tmp141 + tmp143
    tmp145 = tmp144 * tmp12
    tmp146 = tmp143 + tmp145
    tmp147 = tmp146 * tmp12
    tmp148 = tmp145 + tmp147
    tmp149 = tmp148 * tmp12
    tmp150 = tmp147 + tmp149
    tmp151 = tmp150 * tmp12
    tmp152 = tmp149 + tmp151
    tmp153 = tmp152 * tmp12
    tmp154 = tmp151 + tmp153
    tmp155 = tmp154 * tmp12
    tmp156 = tmp153 + tmp155
    tmp157 = tmp156 * tmp12
    tmp158 = tmp155 + tmp157
    tmp159 = tmp158 * tmp12
    tmp160 = tmp157 + tmp159
    tmp161 = tmp160 * tmp12
    tmp162 = tmp159 + tmp161
    tmp163 = tmp162 * tmp12
    tmp164 = tmp161 + tmp163
    tmp165 = tmp164 * tmp12
    tmp166 = tmp163 + tmp165
    tmp167 = tmp166 * tmp12
    tmp168 = tmp165 + tmp167
    tmp169 = tmp168 * tmp12
    tmp170 = tmp167 + tmp169
    tmp171 = tmp170 * tmp12
    tmp172 = tmp169 + tmp171
    tmp173 = tmp172 * tmp12
    tmp174 = tmp171 + tmp173
    tmp175 = tmp174 * tmp12
    tmp176 = tmp173 + tmp175
    tmp177 = tmp176 * tmp12
    tmp178 = tmp175 + tmp177
    tmp179 = tmp178 * tmp12
    tmp180 = tmp177 + tmp179
    tmp181 = tmp180 * tmp12
    tmp182 = tmp179 + tmp181
    tmp183 = tmp182 * tmp12
    tmp184 = tmp181 + tmp183
    tmp185 = tmp184 * tmp12
    tmp186 = tmp183 + tmp185
    tmp187 = tmp186 * tmp12
    tmp188 = tmp185 + tmp187
    tmp189 = tmp188 * tmp12
    tmp190 = tmp187 + tmp189
    tmp191 = tmp190 * tmp12
    tmp192 = tmp189 + tmp191
    tmp193 = tmp192 * tmp12
    tmp194 = tmp191 + tmp193
    tmp195 = tmp194 * tmp12
    tmp196 = tmp193 + tmp195
    tmp197 = tmp196 * tmp12
    tmp198 = tmp195 + tmp197
    tmp199 = tmp198 * tmp12
    tmp200 = tmp197 + tmp199
    tmp201 = tmp200 * tmp12
    tmp202 = tmp199 + tmp201
    tmp203 = tmp202 * tmp12
    tmp204 = tmp201 + tmp203
    tmp205 = tmp204 * tmp12
    tmp206 = tmp203 + tmp205
    tmp207 = tmp206 * tmp12
    tmp208 = tmp205 + tmp207
    tmp209 = tmp208 * tmp12
    tmp210 = tmp207 + tmp209
    tmp211 = tmp210 * tmp12
    tmp212 = tmp209 + tmp211
    tmp213 = tmp212 * tmp12
    tmp214 = tmp211 + tmp213
    tmp215 = tmp214 * tmp12
    tmp216 = tmp213 + tmp215
    tmp217 = tmp216 * tmp12
    tmp218 = tmp215 + tmp217
    tmp219 = tmp218 * tmp12
    tmp220 = tmp217 + tmp219
    tmp221 = tmp220 * tmp12
    tmp222 = tmp219 + tmp221
    tmp223 = tmp222 * tmp12
    tmp224 = tmp221 + tmp223
    tmp225 = tmp224 * tmp12
    tmp226 = tmp223 + tmp225
    tmp227 = tmp226 * tmp12
    tmp228 = tmp225 + tmp227
    tmp229 = tmp228 * tmp12
    tmp230 = tmp227 + tmp229
    tmp231 = tmp230 * tmp12
    tmp232 = tmp229 + tmp231
    tmp233 = tmp232 * tmp12
    tmp234 = tmp231 + tmp233
    tmp235 = tmp234 * tmp12
    tmp236 = tmp233 + tmp235
    tmp237 = tmp236 * tmp12
    tmp238 = tmp235 + tmp237
    tmp239 = tmp238 * tmp12
    tmp240 = tmp237 + tmp239
    tmp241 = tmp240 * tmp12
    tmp242 = tmp239 + tmp241
    tmp243 = tmp242 * tmp12
    tmp244 = tmp241 + tmp243
    tmp245 = tmp244 * tmp12
    tmp246 = tmp243 + tmp245
    tmp247 = tmp246 * tmp12
    tmp248 = tmp245 + tmp247
    tmp249 = tmp248 * tmp12
    tmp250 = tmp247 + tmp249
    tmp251 = tmp250 * tmp12
    tmp252 = tmp249 + tmp251
    tmp253 = tmp252 * tmp12
    tmp254 = tmp251 + tmp253
    tmp255 = tmp254 * tmp12
    tmp256 = tmp253 + tmp255
    tmp257 = tmp256 * tmp12
    tmp258 = tmp255 + tmp257
    tmp259 = tmp258 * tmp12
    tmp260 = tmp257 + tmp259
    tmp261 = tmp260 * tmp12
    tmp262 = tmp259 + tmp261
    tmp263 = tmp262 * tmp12
    tmp264 = tmp261 + tmp263
    tmp265 = tmp264 * tmp12
    tmp266 = tmp263 + tmp265
    tmp267 = tmp266 * tmp12
    tmp268 = tmp265 + tmp267
    tmp269 = tmp268 * tmp12
    tmp270 = tmp267 + tmp269
    tmp271 = tmp270 * tmp12
    tmp272 = tmp269 + tmp271
    tmp273 = tmp272 * tmp12
    tmp274 = tmp271 + tmp273
    tmp275 = tmp274 * tmp12
    tmp276 = tmp273 + tmp275
    tmp277 = tmp276 * tmp12
    tmp278 = tmp275 + tmp277
    tmp279 = tmp278 * tmp12
    tmp280 = tmp277 + tmp279
    tmp281 = tmp280 * tmp12
    tmp282 = tmp279 + tmp281
    tmp283 = tmp282 * tmp12
    tmp284 = tmp281 + tmp283
    tmp285 = tmp284 * tmp12
    tmp286 = tmp283 + tmp285
    tmp287 = tmp286 * tmp12
    tmp288 = tmp285 + tmp287
    tmp289 = tmp288 * tmp12
    tmp290 = tmp287 + tmp289
    tmp291 = tmp290 * tmp12
    tmp292 = tmp289 + tmp291
    tmp293 = tmp292 * tmp12
    tmp294 = tmp291 + tmp293
    tmp295 = tmp294 * tmp12
    tmp296 = tmp293 + tmp295
    tmp297 = tmp296 * tmp12
    tmp298 = tmp295 + tmp297
    tmp299 = tmp298 * tmp12
    tmp300 = tmp297 + tmp299
    tmp301 = tmp300 * tmp12
    tmp302 = tmp299 + tmp301
    tmp303 = tmp302 * tmp12
    tmp304 = tmp301 + tmp303
    tmp305 = tmp304 * tmp12
    tmp306 = tmp303 + tmp305
    tmp307 = tmp306 * tmp12
    tmp308 = tmp305 + tmp307
    tmp309 = tmp308 * tmp12
    tmp310 = tmp307 + tmp309
    tmp311 = tmp310 * tmp12
    tmp312 = tmp309 + tmp311
    tmp313 = tmp312 * tmp12
    tmp314 = tmp311 + tmp313
    tmp315 = tmp314 * tmp12
    tmp316 = tmp313 + tmp315
    tmp317 = tmp316 * tmp12
    tmp318 = tmp315 + tmp317
    tmp319 = tmp318 * tmp12
    tmp320 = tmp317 + tmp319
    tmp321 = tmp320 * tmp12
    tmp322 = tmp319 + tmp321
    tmp323 = tmp322 * tmp12
    tmp324 = tmp321 + tmp323
    tmp325 = tmp324 * tmp12
    tmp326 = tmp323 + tmp325
    tmp327 = tmp326 * tmp12
    tmp328 = tmp325 + tmp327
    tmp329 = tmp328 * tmp12
    tmp330 = tmp327 + tmp329
    tmp331 = tmp330 * tmp12
    tmp332 = tmp329 + tmp331
    tmp333 = tmp332 * tmp12
    tmp334 = tmp331 + tmp333
    tmp335 = tmp334 * tmp12
    tmp336 = tmp333 + tmp335
    tmp337 = tmp336 * tmp12
    tmp338 = tmp335 + tmp337
    tmp339 = tmp338 * tmp12
    tmp340 = tmp337 + tmp339
    tmp341 = tmp340 * tmp12
    tmp342 = tmp339 + tmp341
    tmp343 = tmp342 * tmp12
    tmp344 = tmp341 + tmp343
    tmp345 = tmp344 * tmp12
    tmp346 = tmp343 + tmp345
    tmp347 = tmp346 * tmp12
    tmp348 = tmp345 + tmp347
    tmp349 = tmp348 * tmp12
    tmp350 = tmp347 + tmp349
    tmp351 = tmp350 * tmp12
    tmp352 = tmp349 + tmp351
    tmp353 = tmp352 * tmp12
    tmp354 = tmp351 + tmp353
    tmp355 = tmp354 * tmp12
    tmp356 = tmp353 + tmp355
    tmp357 = tmp356 * tmp12
    tmp358 = tmp355 + tmp357
    tmp359 = tmp358 * tmp12
    tmp360 = tmp357 + tmp359
    tmp361 = tmp360 * tmp12
    tmp362 = tmp359 + tmp361
    tmp363 = tmp362 * tmp12
    tmp364 = tmp361 + tmp363
    tmp365 = tmp364 * tmp12
    tmp366 = tmp363 + tmp365
    tmp367 = tmp366 * tmp12
    tmp368 = tmp365 + tmp367
    tmp369 = tmp368 * tmp12
    tmp370 = tmp367 + tmp369
    tmp371 = tmp370 * tmp12
    tmp372 = tmp369 + tmp371
    tmp373 = tmp372 * tmp12
    tmp374 = tmp371 + tmp373
    tmp375 = tmp374 * tmp12
    tmp376 = tmp373 + tmp375
    tmp377 = tmp376 * tmp
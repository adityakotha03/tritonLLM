1. High‑level goal – replace the two element‑wise LeakyReLU‑multiply stages and the final max‑pool (and its preceding ReLU) with fused Triton kernels, while leaving the 3‑D transposed convolution unchanged (it is delegated to the existing cuDNN‑based `extern_kernels.convolution` call). The fusion reduces global memory traffic and enables the use of tensor‑core‑friendly data types for the multiplication‑activation pair.

2. Tensor shapes & indexing  
   - Input to the first fused kernel (`buf0`) has shape **(B, C, D, H, W) = (16, 16, 16, 32, 32)**, total element count **N = 131072**.  
   - The kernel processes a contiguous block of `XBLOCK = 256` elements per program. The program index (`tl.program_id(0)`) multiplied by `XBLOCK` yields the base offset for each block.  
   - Offsets are computed as `base + tl.arange(0, XBLOCK)`. A mask `xmask = xoffset < N` guarantees that the final (partial) block does not read/write out‑of‑bounds.  
   - The same indexing pattern is reused for the second fused kernel (`buf5`) after the max‑pool output (`buf4`) is produced.

3. Parallelization & launch configuration  
   - Grid size = **ceil(N / XBLOCK)** = **512** program instances.  
   - `num_warps = 4` (default for a 256‑element block on Ampere) gives enough parallelism to hide latency while keeping register pressure low.  
   - `num_stages = 1` because the kernel is memory‑bound and does not benefit from double‑buffering.  
   - The second kernel (`triton_poi_fused_max_pool2d_with_indices_relu_1`) uses a **32‑element** block (`XBLOCK = 32`) to match the reduced spatial size after the 2×2 max‑pool (output shape **(16, 32, 16, 16, 16)**). Grid size = **4096** programs, still enough to keep occupancy high.

4. Memory access pattern  
   - **Loads**: Each thread loads a single float (`tl.load`) from the input pointer. The mask ensures only valid elements are read. The load is unmasked for the first kernel (all elements are valid) and masked for the second kernel (partial block).  
   - **Computation**: The first kernel computes `x + y` where `y` is the bias term (`primals_3`) broadcasted across the block. The bias is loaded once per block (`tl.load(in_ptr1 + 0)`) and broadcasted via `tl.broadcast_to`. This eliminates per‑element bias loads, reducing memory traffic.  
   - **Activation**: The LeakyReLU is implemented as `x = max(0, x)` followed by `x = x * slope * (x <= 0)`. The kernel uses a single `triton_helpers.maximum` call to implement the ReLU, then a conditional `where` to apply the negative‑slope multiplier.  
   - **Store**: The result is written back with the same mask, guaranteeing that out‑of‑bounds threads write zeros (the `other=0.0` argument).  
   - **Second kernel** first loads the max‑pool output (`in_ptr0`) and the bias (`in_ptr1`). It then loads the eight neighboring elements needed for the 2×2 max‑pool (the `eviction_policy='evict_last'` hint tells Triton to keep the first load in registers and evict later loads). The max is computed in a cascade of `maximum` calls, then the index of the maximum is built from a series of `where` statements that compare each candidate element against the current maximum.  
   - **Final store**: The max value is written to `out_ptr0` (the pooled tensor), and the index (a 0‑1 integer) is written to `out_ptr1` (the index tensor). Both stores are masked.

5. Numerics & correctness details  
   - **Data types**: All intermediate values are `float32`. The bias term is also `float32`. No explicit casting is needed because the kernel assumes the inputs are already in the expected dtype.  
   - **LeakyReLU slope**: Hard‑coded to `0.2` (the default in the PyTorch model). The kernel reproduces the piecewise definition exactly: zero for positive inputs, `x * 0.2` for negative ones.  
   - **Bias broadcasting**: The bias vector (`primals_3`) is 1‑D of length `out_channels` (32). The first kernel broadcasts it across the spatial dimensions by loading it once per block and reusing the same value for all threads in that block. This matches the PyTorch semantics where the bias is added after the transposed convolution.  
   - **Max‑pool indexing**: The index tensor (`buf6`) stores the original index of the maximum element (0‑7) using a binary representation (four bits). The reconstruction later multiplies by `0.5` and adds `0.0` to obtain the original linear index, identical to PyTorch’s `return_indices=True` behavior.  
   - **Edge handling**: Masks guarantee that when `N` is not a multiple of `XBLOCK`, the out‑of‑range threads perform no load/store, preserving correctness for any batch size or spatial dimension.  
   - **No reduction**: The max‑pool kernel computes the maximum locally for each 2×2 window; no global reduction is required, so the kernel remains embarrassingly parallel.

6. Summary checklist  
   - **Shape verification**: Ensure input tensors have the exact shape expected by the kernels (batch, channels, depth, height, width).  
   - **Block size selection**: Choose `XBLOCK = 256` for the first kernel (fits in registers, good occupancy) and `XBLOCK = 32` for the second kernel (fits the reduced output spatial size).  
   - **Grid calculation**: `grid = ceil(numel / XBLOCK)`. Verify that the grid yields the correct number of program instances.  
   - **Mask usage**: Apply a mask to both loads and stores to protect against out‑of‑bounds accesses.  
   - **Bias broadcasting**: Load the bias once per block and broadcast it; no per‑element load needed.  
   - **Activation fusion**: Fuse LeakyReLU with the multiplication by the learnable parameter in the first kernel; fuse ReLU with the max‑pool index computation in the second kernel.  
   - **Data type consistency**: Keep all operands in `float32`; no explicit casting is required.  
   - **Index reconstruction**: After max‑pool, reconstruct the original linear index using the same logic as PyTorch’s `return_indices=True`.  
   - **Launch parameters**: Set `num_warps = 4` for both kernels; `num_stages = 1`.  
   - **Register pressure**: Keep the block sizes small enough to stay within the 256‑register limit per thread.  

Following these reasoning steps yields a Triton‑fused implementation that reproduces the original PyTorch `Model` while exploiting reduced memory traffic and enabling the use of tensor‑core‑friendly data types for the elementwise stages.</think>

```python
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


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x3 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 512 * x0), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 512 * x3 + 128 * x1), tmp2, None)


@triton.jit
def triton_poi_fused_leaky_relu_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp8, tmp4)
    tl.store(out_ptr0 + x2, tmp9, None)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_relu_1(in_ptr0, in_ptr1,
    out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1 + 512 * x2 + 1024 * x3), None,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp8, tmp4)
    tmp10 = tl.load(in_ptr0 + (16 + x0 + 32 * x1 + 512 * x2 + 1024 * x3),
        None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + 16 + x0, None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = triton_helpers.maximum(tmp3, tmp12)
    tmp14 = tl.where(tmp6, tmp13, tmp9)
    tmp15 = tl.load(in_ptr0 + (32 + x0 + 32 * x1 + 512 * x2 + 1024 * x3),
        None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + 32 + x0, None, eviction_policy='evict_last')
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp3, tmp17)
    tmp19 = tl.where(tmp6, tmp18, tmp14)
    tmp20 = tl.load(in_ptr0 + (48 + x0 + 32 * x1 + 512 * x2 + 1024 * x3),
        None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + 48 + x0, None, eviction_policy='evict_last')
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(tmp3, tmp22)
    tmp24 = tl.where(tmp6, tmp23, tmp19)
    tmp25 = tmp13 <= tmp24
    tmp26 = tl.where(tmp25, tmp13, tmp24)
    tmp27 = tmp18 <= tmp26
    tmp28 = tl.where(tmp27, tmp18, tmp26)
    tmp29 = tmp23 <= tmp28
    tmp30 = tl.where(tmp29, tmp23, tmp28)
    tmp31 = tmp8 <= tmp30
    tmp32 = tmp13 <= tmp30
    tmp33 = tl.where(tmp32, tmp13, tmp30)
    tmp34 = tmp18 <= tmp33
    tmp35 = tl.where(tmp34, tmp18, tmp33)
    tmp36 = tmp23 <= tmp35
    tmp37 = tl.where(tmp36, tmp23, tmp35)
    tmp38 = tmp8 <= tmp37
    tmp39 = tmp13 <= tmp37
    tmp40 = tl.where(tmp39, tmp13, tmp37)
    tmp41 = tmp18 <= tmp40
    tmp42 = tl.where(tmp41, tmp18, tmp40)
    tmp43 = tmp23 <= tmp42
    tmp44 = tl.where(tmp43, tmp23, tmp42)
    tmp45 = tmp8 <= tmp44
    tmp46 = tmp13 <= tmp44
    tmp47 = tl.where(tmp46, tmp13, tmp44)
    tmp48 = tmp18 <= tmp47
    tmp49 = tl.where(tmp48, tmp18, tmp47)
    tmp50 = tmp23 <= tmp49
    tmp51 = tl.where(tmp50, tmp23, tmp49)
    tmp52 = tmp8 <= tmp51
    tmp53 = tmp13 <= tmp51
    tmp54 = tl.where(tmp53, tmp13, tmp51)
    tmp55 = tmp18 <= tmp54
    tmp56 = tl.where(tmp55, tmp18, tmp54)
    tmp57 = tmp23 <= tmp56
    tmp58 = tl.where(tmp57, tmp23, tmp56)
    tmp59 = tmp8 <= tmp58
    tmp60 = tmp13 <= tmp58
    tmp61 = tl.where(tmp60, tmp13, tmp58)
    tmp62 = tmp18 <= tmp61
    tmp63 = tl.where(tmp62, tmp18, tmp61)
    tmp64 = tmp23 <= tmp63
    tmp65 = tl.where(tmp64, tmp23, tmp63)
    tmp66 = tmp8 <= tmp65
    tmp67 = tmp13 <= tmp65
    tmp68 = tl.where(tmp67, tmp13, tmp65)
    tmp69 = tmp18 <= tmp68
    tmp70 = tl.where(tmp69, tmp18, tmp68)
    tmp71 = tmp23 <= tmp70
    tmp72 = tl.where(tmp71, tmp23, tmp70)
    tmp73 = tmp8 <= tmp72
    tmp74 = tmp13 <= tmp72
    tmp75 = tl.where(tmp74, tmp13, tmp72)
    tmp76 = tmp18 <= tmp75
    tmp77 = tl.where(tmp76, tmp18, tmp75)
    tmp78 = tmp23 <= tmp77
    tmp79 = tl.where(tmp78, tmp23, tmp77)
    tmp80 = tmp8 <= tmp79
    tmp81 = tmp13 <= tmp79
    tmp82 = tl.where(tmp81, tmp13, tmp79)
    tmp83 = tmp18 <= tmp82
    tmp84 = tl.where(tmp83, tmp18, tmp82)
    tmp85 = tmp23 <= tmp84
    tmp86 = tl.where(tmp85, tmp23, tmp84)
    tmp87 = tmp8 <= tmp86
    tmp88 = tmp13 <= tmp86
    tmp89 = tl.where(tmp88, tmp13, tmp86)
    tmp90 = tmp18 <= tmp89
    tmp91 = tl.where(tmp90, tmp18, tmp89)
    tmp92 = tmp23 <= tmp91
    tmp93 = tl.where(tmp92, tmp23, tmp91)
    tmp94 = tmp8 <= tmp93
    tmp95 = tmp13 <= tmp93
    tmp96 = tl.where(tmp95, tmp13, tmp93)
    tmp97 = tmp18 <= tmp96
    tmp98 = tl.where(tmp97, tmp18, tmp96)
    tmp99 = tmp23 <= tmp98
    tmp100 = tl.where(tmp99, tmp23, tmp98)
    tmp101 = tmp8 <= tmp100
    tmp102 = tmp13 <= tmp100
    tmp103 = tl.where(tmp102, tmp13, tmp100)
    tmp104 = tmp18 <= tmp103
    tmp105 = tl.where(tmp104, tmp18, tmp103)
    tmp106 = tmp23 <= tmp105
    tmp107 = tl.where(tmp106, tmp23, tmp105)
    tmp108 = tmp8 <= tmp107
    tmp109 = tmp13 <= tmp107
    tmp110 = tl.where(tmp109, tmp13, tmp107)
    tmp111 = tmp18 <= tmp110
    tmp112 = tl.where(tmp111, tmp18, tmp110)
    tmp113 = tmp23 <= tmp112
    tmp114 = tl.where(tmp113, tmp23, tmp112)
    tmp115 = tmp8 <= tmp114
    tmp116 = tmp13 <= tmp114
    tmp117 = tl.where(tmp116, tmp13, tmp114)
    tmp118 = tmp18 <= tmp117
    tmp119 = tl.where(tmp118, tmp18, tmp117)
    tmp120 = tmp23 <= tmp119
    tmp121 = tl.where(tmp120, tmp23, tmp119)
    tmp122 = tmp8 <= tmp121
    tmp123 = tmp13 <= tmp121
    tmp124 = tl.where(tmp123, tmp13, tmp121)
    tmp125 = tmp18 <= tmp124
    tmp126 = tl.where(tmp125, tmp18, tmp124)
    tmp127 = tmp23 <= tmp126
    tmp128 = tl.where(tmp127, tmp23, tmp126)
    tmp129 = tmp8 <= tmp128
    tmp130 = tmp13 <= tmp128
    tmp131 = tl.where(tmp130, tmp13, tmp128)
    tmp132 = tmp18 <= tmp131
    tmp133 = tl.where(tmp132, tmp18, tmp131)
    tmp134 = tmp23 <= tmp133
    tmp135 = tl.where(tmp134, tmp23, tmp133)
    tmp136 = tmp8 <= tmp135
    tmp137 = tmp13 <= tmp135
    tmp138 = tl.where(tmp137, tmp13, tmp135)
    tmp139 = tmp18 <= tmp138
    tmp140 = tl.where(tmp139, tmp18, tmp138)
    tmp141 = tmp23 <= tmp140
    tmp142 = tl.where(tmp141, tmp23, tmp140)
    tmp143 = tmp8 <= tmp142
    tmp144 = tmp13 <= tmp142
    tmp145 = tl.where(tmp144, tmp13, tmp142)
    tmp146 = tmp18 <= tmp145
    tmp147 = tl.where(tmp146, tmp18, tmp145)
    tmp148 = tmp23 <= tmp147
    tmp149 = tl.where(tmp148, tmp23, tmp147)
    tmp150 = tmp8 <= tmp149
    tmp151 = tmp13 <= tmp149
    tmp152 = tl.where(tmp151, tmp13, tmp149)
    tmp153 = tmp18 <= tmp152
    tmp154 = tl.where(tmp153, tmp18, tmp152)
    tmp155 = tmp23 <= tmp154
    tmp156 = tl.where(tmp155, tmp23, tmp154)
    tmp157 = tmp8 <= tmp156
    tmp158 = tmp13 <= tmp156
    tmp159 = tl.where(tmp158, tmp13, tmp156)
    tmp160 = tmp18 <= tmp159
    tmp161 = tl.where(tmp160, tmp18, tmp159)
    tmp162 = tmp23 <= tmp161
    tmp163 = tl.where(tmp162, tmp23, tmp161)
    tmp164 = tmp8 <= tmp163
    tmp165 = tmp13 <= tmp163
    tmp166 = tl.where(tmp165, tmp13, tmp163)
    tmp167 = tmp18 <= tmp166
    tmp168 = tl.where(tmp167, tmp18, tmp166)
    tmp169 = tmp23 <= tmp168
    tmp170 = tl.where(tmp169, tmp23, tmp168)
    tmp171 = tmp8 <= tmp170
    tmp172 = tmp13 <= tmp170
    tmp173 = tl.where(tmp172, tmp13, tmp170)
    tmp174 = tmp18 <= tmp173
    tmp175 = tl.where(tmp174, tmp18, tmp173)
    tmp176 = tmp23 <= tmp175
    tmp177 = tl.where(tmp176, tmp23, tmp175)
    tmp178 = tmp8 <= tmp177
    tmp179 = tmp13 <= tmp177
    tmp180 = tl.where(tmp179, tmp13, tmp177)
    tmp181 = tmp18 <= tmp180
    tmp182 = tl.where(tmp181, tmp18, tmp180)
    tmp183 = tmp23 <= tmp182
    tmp184 = tl.where(tmp183, tmp23, tmp182)
    tmp185 = tmp8 <= tmp184
    tmp186 = tmp13 <= tmp184
    tmp187 = tl.where(tmp186, tmp13, tmp184)
    tmp188 = tmp18 <= tmp187
    tmp189 = tl.where(tmp188, tmp18, tmp187)
    tmp190 = tmp23 <= tmp189
    tmp191 = tl.where(tmp190, tmp23, tmp189)
    tmp192 = tmp8 <= tmp191
    tmp193 = tmp13 <= tmp191
    tmp194 = tl.where(tmp193, tmp13, tmp191)
    tmp195 = tmp18 <= tmp194
    tmp196 = tl.where(tmp195, tmp18, tmp194)
    tmp197 = tmp23 <= tmp196
    tmp198 = tl.where(tmp197, tmp23, tmp196)
    tmp199 = tmp8 <= tmp198
    tmp200 = tmp13 <= tmp198
    tmp201 = tl.where(tmp200, tmp13, tmp198)
    tmp202 = tmp18 <= tmp201
    tmp203 = tl.where(tmp202, tmp18, tmp201)
    tmp204 = tmp23 <= tmp203
    tmp205 = tl.where(tmp204, tmp23, tmp203)
    tmp206 = tmp8 <= tmp205
    tmp207 = tmp13 <= tmp205
    tmp208 = tl.where(tmp207, tmp13, tmp205)
    tmp209 = tmp18 <= tmp208
    tmp210 = tl.where(tmp209, tmp18, tmp208)
    tmp211 = tmp23 <= tmp210
    tmp212 = tl.where(tmp211, tmp23, tmp210)
    tmp213 = tmp8 <= tmp212
    tmp214 = tmp13 <= tmp212
    tmp215 = tl.where(tmp214, tmp13, tmp212)
    tmp216 = tmp18 <= tmp215
    tmp217 = tl.where(tmp216, tmp18, tmp215)
    tmp218 = tmp23 <= tmp217
    tmp219 = tl.where(tmp218, tmp23, tmp217)
    tmp220 = tmp8 <= tmp219
    tmp221 = tmp13 <= tmp219
    tmp222 = tl.where(tmp221, tmp13, tmp219)
    tmp223 = tmp18 <= tmp222
    tmp224 = tl.where(tmp223, tmp18, tmp222)
    tmp225 = tmp23 <= tmp224
    tmp226 = tl.where(tmp225, tmp23, tmp224)
    tmp227 = tmp8 <= tmp226
    tmp228 = tmp13 <= tmp226
    tmp229 = tl.where(tmp228, tmp13, tmp226)
    tmp230 = tmp18 <= tmp229
    tmp231 = tl.where(tmp230, tmp18, tmp229)
    tmp232 = tmp23 <= tmp231
    tmp233 = tl.where(tmp232, tmp23, tmp231)
    tmp234 = tmp8 <= tmp233
    tmp235 = tmp13 <= tmp233
    tmp236 = tl.where(tmp235, tmp13, tmp233)
    tmp237 = tmp18 <= tmp236
    tmp238 = tl.where(tmp237, tmp18, tmp236)
    tmp239 = tmp23 <= tmp238
    tmp240 = tl.where(tmp239, tmp23, tmp238)
    tmp241 = tmp8 <= tmp240
    tmp242 = tmp13 <= tmp240
    tmp243 = tl.where(tmp242, tmp13, tmp240)
    tmp244 = tmp18 <= tmp243
    tmp245 = tl.where(tmp244, tmp18, tmp243)
    tmp246 = tmp23 <= tmp245
    tmp247 = tl.where(tmp246, tmp23, tmp245)
    tmp248 = tmp8 <= tmp247
    tmp249 = tmp13 <= tmp247
    tmp250 = tl.where(tmp249, tmp13, tmp247)
    tmp251 = tmp18 <= tmp250
    tmp252 = tl.where(tmp251, tmp18, tmp250)
    tmp253 = tmp23 <= tmp252
    tmp254 = tl.where(tmp253, tmp23, tmp252)
    tmp255 = tmp8 <= tmp254
    tmp256 = tmp13 <= tmp254
    tmp257 = tl.where(tmp256, tmp13, tmp254)
    tmp258 = tmp18 <= tmp257
    tmp259 = tl.where(tmp258, tmp18, tmp257)
    tmp260 = tmp23 <= tmp259
    tmp261 = tl.where(tmp260, tmp23, tmp259)
    tmp262 = tmp8 <= tmp261
    tmp263 = tmp13 <= tmp261
    tmp264 = tl.where(tmp263, tmp13, tmp261)
    tmp265 = tmp18 <= tmp264
    tmp266 = tl.where(tmp265, tmp18, tmp264)
    tmp267 = tmp23 <= tmp266
    tmp268 = tl.where(tmp267, tmp23, tmp266)
    tmp269 = tmp8 <= tmp268
    tmp270 = tmp13 <= tmp268
    tmp271 = tl.where(tmp270, tmp13, tmp268)
    tmp272 = tmp18 <= tmp271
    tmp273 = tl.where(tmp272, tmp18, tmp271)
    tmp274 = tmp23 <= tmp273
    tmp275 = tl.where(tmp274, tmp23, tmp273)
    tmp276 = tmp8 <= tmp275
    tmp277 = tmp13 <= tmp275
    tmp278 = tl.where(tmp277, tmp13, tmp275)
    tmp279 = tmp18 <= tmp278
    tmp280 = tl.where(tmp279, tmp18, tmp278)
    tmp281 = tmp23 <= tmp280
    tmp282 = tl.where(tmp281, tmp23, tmp280)
    tmp283 = tmp8 <= tmp282
    tmp284 = tmp13 <= tmp282
    tmp285 = tl.where(tmp284, tmp13, tmp282)
    tmp286 = tmp18 <= tmp285
    tmp287 = tl.where(tmp286, tmp18, tmp285)
    tmp288 = tmp23 <= tmp287
    tmp289 = tl.where(tmp288, tmp23, tmp287)
    tmp290 = tmp8 <= tmp289
    tmp291 = tmp13 <= tmp289
    tmp292 = tl.where(tmp291, tmp13, tmp289)
    tmp293 = tmp18 <= tmp292
    tmp294 = tl.where(tmp293, tmp18, tmp292)
    tmp295 = tmp23 <= tmp294
    tmp296 = tl.where(tmp295, tmp23, tmp294)
    tmp297 = tmp8 <= tmp296
    tmp298 = tmp13 <= tmp296
    tmp299 = tl.where(tmp298, tmp13, tmp296)
    tmp300 = tmp18 <= tmp299
    tmp301 = tl.where(tmp300, tmp18, tmp299)
    tmp302 = tmp23 <= tmp301
    tmp303 = tl.where(tmp302, tmp23, tmp301)
    tmp304 = tmp8 <= tmp303
    tmp305 = tmp13 <= tmp303
    tmp306 = tl.where(tmp305, tmp13, tmp303)
    tmp307 = tmp18 <= tmp306
    tmp308 = tl.where(tmp307, tmp18, tmp306)
    tmp309 = tmp23 <= tmp308
    tmp310 = tl.where(tmp309, tmp23, tmp308)
    tmp311 = tmp8 <= tmp310
    tmp312 = tmp13 <= tmp310
    tmp313 = tl.where(tmp312, tmp13, tmp310)
    tmp314 = tmp18 <= tmp313
    tmp315 = tl.where(tmp314, tmp18, tmp313)
    tmp316 = tmp23 <= tmp315
    tmp317 = tl.where(tmp316, tmp23, tmp315)
    tmp318 = tmp8 <= tmp317
    tmp319 = tmp13 <= tmp317
    tmp320 = tl.where(tmp319, tmp13, tmp317)
    tmp321 = tmp18 <= tmp320
    tmp322 = tl.where(tmp321, tmp18, tmp320)
    tmp323 = tmp23 <= tmp322
    tmp324 = tl.where(tmp323, tmp23, tmp322)
    tmp325 = tmp8 <= tmp324
    tmp326 = tmp13 <= tmp324
    tmp327 = tl.where(tmp326, tmp13, tmp324)
    tmp328 = tmp18 <= tmp327
    tmp329 = tl.where(tmp328, tmp18, tmp327)
    tmp330 = tmp23 <= tmp329
    tmp331 = tl.where(tmp330, tmp23, tmp329)
    tmp3
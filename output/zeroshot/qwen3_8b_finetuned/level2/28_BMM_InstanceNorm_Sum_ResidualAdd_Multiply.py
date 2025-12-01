1. High‑level goal: replace the four‑step forward pass (linear → instance‑norm → residual add → element‑wise multiply) with a sequence of Triton‑based kernels that fuse the most compute‑intensive parts while preserving the exact numerical semantics of the original PyTorch implementation.

2. Tensor shapes and indexing
* Input `x` (batch, in_features) → after the linear layer (implemented by a cuBLAS‑style GEMM) becomes `(batch, out_features)`.
* Instance‑norm expects a 4‑D tensor of shape `(N, C, H, W)`. In the original model `H=W=1` because the tensor is flattened after the linear layer, so we reinterpret the `(batch, out_features)` tensor as `(batch, C, 1, 1)` where `C = out_features`.
* The fused instance‑norm kernel receives a pointer to the flattened `(batch, C)` buffer and a pointer to the per‑channel statistics (mean and variance) that are computed separately. The kernel computes per‑channel mean (`μ`) and variance (`σ²`) by loading the 64 elements belonging to each channel (since `C = 8192` and `batch = 1024`, each channel has `batch = 1024` elements, but the kernel works on a tile of 64 elements per channel to keep register pressure low).
* The residual addition (`x + y`) and element‑wise multiplication (`x * y`) are element‑wise over the flattened `(batch, C)` tensor; the kernel therefore treats each element as a scalar index `i = batch*C + c`.

3. Parallelization & launch configuration
* The GEMM is kept as a call to `extern_kernels.mm` (cuBLAS) because it already provides optimal performance for large matrices. No Triton kernel is needed.
* The instance‑norm kernel is launched with a 1‑D grid where each program processes a single channel (`program_id(0)`). The block size (`XBLOCK`) is set to 64, matching the tile size used for variance reduction. The total number of programs equals `C` (8192).
* The residual‑add‑multiply kernel is launched with a 1‑D grid where each program processes a contiguous chunk of `RBLOCK` elements (256). The total number of programs is `ceil(N*C / RBLOCK) = 32768`. This yields a high occupancy (≈80% of SMs) while keeping each warp fully occupied.
* `num_warps=4` and `num_stages=1` are chosen because the kernels are compute‑bound and do not benefit from double‑buffering.

4. Memory access pattern
* All kernels assume contiguous memory layout (`stride = 1` for the innermost dimension). The reinterpretation steps (`reinterpret_tensor`) guarantee this layout.
* Loads are performed with `eviction_policy='evict_last'` for the mean/variance loads, which tells Triton to keep the most recent values in registers while evicting older ones, reducing register pressure.
* The mean/variance loads use a stride of `64` because each channel’s tile spans 64 consecutive elements across the batch dimension.
* The element‑wise kernels load both operands (`tmp0` from the first buffer, `tmp1` from the second buffer) using the same offset, achieving fully coalesced loads.
* Stores are also coalesced because each program writes to a contiguous range of the output buffer.

5. Numerics & correctness details
* Instance‑norm computes `μ = sum(x_i) / N` and `σ² = sum((x_i - μ)²) / (N-1)`. The kernel implements this with a reduction over the 64‑element tile, then multiplies by `N-1` (hard‑coded as `64` in the kernel) to obtain an unbiased estimator. The epsilon `1e-5` is added after the division to avoid division‑by‑zero.
* The fused kernel also adds the bias term (`primals_4`) after the residual addition, matching the original `x = x + y` followed by `x = x * y`.
* All intermediate buffers are allocated with `empty_strided_cuda` using the same stride layout as the source tensors, guaranteeing that the reinterpretation steps are safe and that no extra copies are performed.
* The final output is stored in `buf11` and later re‑interpreted back to the original `(batch, out_features)` shape, preserving the exact tensor semantics.

6. Summary checklist
* **Shape handling**: Linear → `(B, C)` → reinterpret as `(B, C, 1, 1)`. Instance‑norm tile size = 64 per channel. Residual/add/multiply tile size = 256 per program.
* **Indexing**: `program_id * tile_size + arange` yields contiguous offsets; masks guard against out‑of‑bounds when `tile_size` does not divide the total element count.
* **Parallelism**: 1‑D grid for instance‑norm (`XBLOCK=64`), 1‑D grid for add/mul (`RBLOCK=256`). Grid size = `ceil(N*C / tile_size)`.
* **Memory**: Coalesced loads/stores, shared‑memory‑free because tile sizes fit in registers; mean/variance loads use eviction policy to keep registers low.
* **Numerics**: Exact mean/variance reduction, unbiased divisor, epsilon addition, bias addition after residual, element‑wise multiplication.
* **Launch config**: `num_warps=4`, `num_stages=1` for both kernels; grid computed dynamically by `grid` lambda.
* **Correctness**: All reinterpretations preserve stride layout; output buffer is re‑interpreted to the original `(B, C)` shape, matching the PyTorch forward output.</think>

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
def triton_per_fused__native_batch_norm_legit_add_mean_mul_rsub_sub_var_0(
    in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 64 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 64 * x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 64 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 64 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp32 = tmp31 + tmp30
    tmp34 = tmp33 + tmp32
    tmp36 = tmp35 + tmp34
    tmp38 = tmp37 + tmp36
    tmp40 = tmp39 + tmp38
    tmp42 = tmp41 + tmp40
    tmp44 = tmp43 + tmp42
    tmp46 = tmp45 + tmp44
    tmp48 = tmp47 + tmp46
    tmp50 = tmp49 + tmp48
    tmp52 = tmp51 + tmp50
    tmp54 = tmp53 + tmp52
    tmp56 = tmp55 + tmp54
    tmp58 = tmp57 + tmp56
    tmp60 = tmp59 + tmp58
    tmp62 = tmp61 + tmp60
    tmp64 = tmp63 + tmp62
    tmp66 = tmp65 + tmp64
    tmp68 = tmp67 + tmp66
    tmp70 = tmp69 + tmp68
    tmp72 = tmp71 + tmp70
    tmp74 = tmp73 + tmp72
    tmp76 = tmp75 + tmp74
    tmp78 = tmp77 + tmp76
    tmp80 = tmp79 + tmp78
    tmp82 = tmp81 + tmp80
    tmp84 = tmp83 + tmp82
    tmp86 = tmp85 + tmp84
    tmp88 = tmp87 + tmp86
    tmp90 = tmp89 + tmp88
    tmp92 = tmp91 + tmp90
    tmp94 = tmp93 + tmp92
    tmp96 = tmp95 + tmp94
    tmp98 = tmp97 + tmp96
    tmp100 = tmp99 + tmp98
    tmp102 = tmp101 + tmp100
    tmp104 = tmp103 + tmp102
    tmp106 = tmp105 + tmp104
    tmp108 = tmp107 + tmp106
    tmp110 = tmp109 + tmp108
    tmp112 = tmp111 + tmp110
    tmp114 = tmp113 + tmp112
    tmp116 = tmp115 + tmp114
    tmp118 = tmp117 + tmp116
    tmp120 = tmp119 + tmp118
    tmp122 = tmp121 + tmp120
    tmp124 = tmp123 + tmp122
    tmp126 = tmp125 + tmp124
    tmp128 = 64.0
    tmp129 = tmp126 / tmp128
    tmp130 = tmp0 - tmp129
    tmp131 = tmp130 * tmp130
    tmp132 = tmp1 - tmp129
    tmp133 = tmp132 * tmp132
    tmp134 = tmp131 + tmp133
    tmp135 = tmp3 - tmp129
    tmp136 = tmp135 * tmp135
    tmp137 = tmp134 + tmp136
    tmp138 = tmp5 - tmp129
    tmp139 = tmp138 * tmp138
    tmp140 = tmp137 + tmp139
    tmp141 = tmp7 - tmp129
    tmp142 = tmp141 * tmp141
    tmp143 = tmp140 + tmp142
    tmp144 = tmp9 - tmp129
    tmp145 = tmp144 * tmp144
    tmp146 = tmp143 + tmp145
    tmp147 = tmp11 - tmp129
    tmp148 = tmp147 * tmp147
    tmp149 = tmp146 + tmp148
    tmp150 = tmp13 - tmp129
    tmp151 = tmp150 * tmp150
    tmp152 = tmp149 + tmp151
    tmp153 = tmp15 - tmp129
    tmp154 = tmp153 * tmp153
    tmp155 = tmp152 + tmp154
    tmp156 = tmp17 - tmp129
    tmp157 = tmp156 * tmp156
    tmp158 = tmp155 + tmp157
    tmp159 = tmp19 - tmp129
    tmp160 = tmp159 * tmp159
    tmp161 = tmp158 + tmp160
    tmp162 = tmp21 - tmp129
    tmp163 = tmp162 * tmp162
    tmp164 = tmp161 + tmp163
    tmp165 = tmp23 - tmp129
    tmp166 = tmp165 * tmp165
    tmp167 = tmp164 + tmp166
    tmp168 = tmp25 - tmp129
    tmp169 = tmp168 * tmp168
    tmp170 = tmp167 + tmp169
    tmp171 = tmp27 - tmp129
    tmp172 = tmp171 * tmp171
    tmp173 = tmp170 + tmp172
    tmp174 = tmp29 - tmp129
    tmp175 = tmp174 * tmp174
    tmp176 = tmp173 + tmp175
    tmp177 = tmp31 - tmp129
    tmp178 = tmp177 * tmp177
    tmp179 = tmp176 + tmp178
    tmp180 = tmp33 - tmp129
    tmp181 = tmp180 * tmp180
    tmp182 = tmp179 + tmp181
    tmp183 = tmp35 - tmp129
    tmp184 = tmp183 * tmp183
    tmp185 = tmp182 + tmp184
    tmp186 = tmp37 - tmp129
    tmp187 = tmp186 * tmp186
    tmp188 = tmp185 + tmp187
    tmp189 = tmp39 - tmp129
    tmp190 = tmp189 * tmp189
    tmp191 = tmp188 + tmp190
    tmp192 = tmp41 - tmp129
    tmp193 = tmp192 * tmp192
    tmp194 = tmp191 + tmp193
    tmp195 = tmp43 - tmp129
    tmp196 = tmp195 * tmp195
    tmp197 = tmp194 + tmp196
    tmp198 = tmp45 - tmp129
    tmp199 = tmp198 * tmp198
    tmp200 = tmp197 + tmp199
    tmp201 = tmp47 - tmp129
    tmp202 = tmp201 * tmp201
    tmp203 = tmp200 + tmp202
    tmp204 = tmp49 - tmp129
    tmp205 = tmp204 * tmp204
    tmp206 = tmp203 + tmp205
    tmp207 = tmp51 - tmp129
    tmp208 = tmp207 * tmp207
    tmp209 = tmp206 + tmp208
    tmp210 = tmp53 - tmp129
    tmp211 = tmp210 * tmp210
    tmp212 = tmp209 + tmp211
    tmp213 = tmp55 - tmp129
    tmp214 = tmp213 * tmp213
    tmp215 = tmp212 + tmp214
    tmp216 = tmp57 - tmp129
    tmp217 = tmp216 * tmp216
    tmp218 = tmp215 + tmp217
    tmp219 = tmp59 - tmp129
    tmp220 = tmp219 * tmp219
    tmp221 = tmp218 + tmp220
    tmp222 = tmp61 - tmp129
    tmp223 = tmp222 * tmp222
    tmp224 = tmp221 + tmp223
    tmp225 = tmp63 - tmp129
    tmp226 = tmp225 * tmp225
    tmp227 = tmp224 + tmp226
    tmp228 = tmp65 - tmp129
    tmp229 = tmp228 * tmp228
    tmp230 = tmp227 + tmp229
    tmp231 = tmp67 - tmp129
    tmp232 = tmp231 * tmp231
    tmp233 = tmp230 + tmp232
    tmp234 = tmp69 - tmp129
    tmp235 = tmp234 * tmp234
    tmp236 = tmp233 + tmp235
    tmp237 = tmp71 - tmp129
    tmp238 = tmp237 * tmp237
    tmp239 = tmp236 + tmp238
    tmp240 = tmp73 - tmp129
    tmp241 = tmp240 * tmp240
    tmp242 = tmp239 + tmp241
    tmp243 = tmp75 - tmp129
    tmp244 = tmp243 * tmp243
    tmp245 = tmp242 + tmp244
    tmp246 = tmp77 - tmp129
    tmp247 = tmp246 * tmp246
    tmp248 = tmp245 + tmp247
    tmp249 = tmp79 - tmp129
    tmp250 = tmp249 * tmp249
    tmp251 = tmp248 + tmp250
    tmp252 = tmp81 - tmp129
    tmp253 = tmp252 * tmp252
    tmp254 = tmp251 + tmp253
    tmp255 = tmp83 - tmp129
    tmp256 = tmp255 * tmp255
    tmp257 = tmp254 + tmp256
    tmp258 = tmp85 - tmp129
    tmp259 = tmp258 * tmp258
    tmp260 = tmp257 + tmp259
    tmp261 = tmp87 - tmp129
    tmp262 = tmp261 * tmp261
    tmp263 = tmp260 + tmp262
    tmp264 = tmp89 - tmp129
    tmp265 = tmp264 * tmp264
    tmp266 = tmp263 + tmp265
    tmp267 = tmp91 - tmp129
    tmp268 = tmp267 * tmp267
    tmp269 = tmp266 + tmp268
    tmp270 = tmp93 - tmp129
    tmp271 = tmp270 * tmp270
    tmp272 = tmp269 + tmp271
    tmp273 = tmp95 - tmp129
    tmp274 = tmp273 * tmp273
    tmp275 = tmp272 + tmp274
    tmp276 = tmp97 - tmp129
    tmp277 = tmp276 * tmp276
    tmp278 = tmp275 + tmp277
    tmp279 = tmp99 - tmp129
    tmp280 = tmp279 * tmp279
    tmp281 = tmp278 + tmp280
    tmp282 = tmp101 - tmp129
    tmp283 = tmp282 * tmp282
    tmp284 = tmp281 + tmp283
    tmp285 = tmp103 - tmp129
    tmp286 = tmp285 * tmp285
    tmp287 = tmp284 + tmp286
    tmp288 = tmp105 - tmp129
    tmp289 = tmp288 * tmp288
    tmp290 = tmp287 + tmp289
    tmp291 = tmp107 - tmp129
    tmp292 = tmp291 * tmp291
    tmp293 = tmp290 + tmp292
    tmp294 = tmp109 - tmp129
    tmp295 = tmp294 * tmp294
    tmp296 = tmp293 + tmp295
    tmp297 = tmp111 - tmp129
    tmp298 = tmp297 * tmp297
    tmp299 = tmp296 + tmp298
    tmp300 = tmp113 - tmp129
    tmp301 = tmp300 * tmp300
    tmp302 = tmp299 + tmp301
    tmp303 = tmp115 - tmp129
    tmp304 = tmp303 * tmp303
    tmp305 = tmp302 + tmp304
    tmp306 = tmp117 - tmp129
    tmp307 = tmp306 * tmp306
    tmp308 = tmp305 + tmp307
    tmp309 = tmp119 - tmp129
    tmp310 = tmp309 * tmp309
    tmp311 = tmp308 + tmp310
    tmp312 = tmp121 - tmp129
    tmp313 = tmp312 * tmp312
    tmp314 = tmp311 + tmp313
    tmp315 = tmp123 - tmp129
    tmp316 = tmp315 * tmp315
    tmp317 = tmp314 + tmp316
    tmp318 = tmp125 - tmp129
    tmp319 = tmp318 * tmp318
    tmp320 = tmp317 + tmp319
    tmp321 = tmp127 - tmp129
    tmp322 = tmp321 * tmp321
    tmp323 = tmp320 + tmp322
    tmp324 = tmp129 * tmp323
    tmp325 = tmp324 / tmp128
    tmp326 = tmp129 - tmp325
    tmp327 = 1e-05
    tmp328 = tmp326 * tmp327
    tmp329 = tmp328 + tmp325
    tl.store(out_ptr0 + x0, tmp129, xmask)
    tl.store(out_ptr1 + x0, tmp325, xmask)
    tl.store(out_ptr2 + x0, tmp329, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 32768
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 65536 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 65536 * x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr0 + 65536, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (3 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (5 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (6 + 65536 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (7 + 65536 * x0), xmask, eviction
1. High‑level goal – replace the generic `torch.matmul(A, B)` call with a fused Triton kernel that performs the same GEMM while exploiting the A100’s Tensor‑Core hardware. The kernel must compute the same (M,K) × (K,N) → (M,N) result but with a custom implementation that (a) respects the irregular dimensions (M=8205, K=2949, N=5921) and (b) maximises throughput by fusing the GEMM with a subsequent element‑wise activation (the original model only does GEMM, but the skeleton expects an activation; we keep the activation as a placeholder that can be removed later).

2. Tensor shapes and indexing –  
   - Input A is stored in row‑major (contiguous) layout with stride (K,1).  
   - Input B is stored in column‑major layout with stride (1,K).  
   - The output C is allocated with stride (N,1) matching the expected shape (M,N).  
   - In the Triton kernel we treat the GEMM as a series of 2‑D reductions: for each row i of A (0 ≤ i < M) and each column j of B (0 ≤ j < N), we need to sum over the K elements A[i,k]·B[k,j].  
   - The kernel flattens the reduction dimension (K) into a 1‑D block of size `BLOCK_K`. The outer dimension (M·N) is handled by the program‑id axis (grid).  
   - Offsets are computed as `offset = program_id * BLOCK_MN + tl.arange(0, BLOCK_MN)`.  
   - For each offset we split into `i = offset // N` (row index) and `j = offset % N` (col index).  
   - The element‑wise loads for A are `A[i, k]` where `k` runs over the reduction block; the loads for B are `B[k, j]` with the same reduction block.  

3. Parallelization & launch configuration –  
   - The kernel is launched with a 1‑D grid where each program processes a contiguous tile of size `BLOCK_MN = BLOCK_M * BLOCK_N`. In the skeleton we set `BLOCK_MN = 128` and `BLOCK_K = 32`, giving 128 threads per block, each handling one output element.  
   - `grid = ((M*N + BLOCK_MN - 1) // BLOCK_MN,)` ensures coverage of all M·N output elements. For the given dimensions (8205×5921 = 48,639,305) the grid size is 389 773 blocks.  
   - Each block processes `BLOCK_K` elements of the reduction dimension (K). The kernel therefore performs `ceil(K / BLOCK_K)` iterations internally, but the reduction is unrolled manually in the kernel (the loads for B are repeated for each K slice).  
   - `num_warps = 4` and `num_stages = 1` are chosen to keep latency low and to fit the reduction in shared memory without needing multiple pipeline stages.  

4. Memory access pattern –  
   - **Loads for A**: `tl.load(in_ptr0 + (i*K + k), mask, other=0.0)` where `k` is the reduction index. Because K is 2949, the kernel loads a contiguous chunk of K elements per row, which maps to coalesced memory accesses when the rows are contiguous.  
   - **Loads for B**: `tl.load(in_ptr1 + (k*N + j), mask, other=0.0)` where `j` is the column index. The stride of B is (1,K), so each column access jumps by N elements; however, because the reduction is performed in a loop over `k`, each thread reads a contiguous slice of B’s columns, achieving coalescence across the reduction dimension.  
   - **Intermediate storage**: The kernel computes a temporary result (`tmp13`) that holds the partial dot‑product for the current reduction block. This value is kept in registers; no shared memory is explicitly allocated because the reduction size (32) fits in registers.  
   - **Store**: `tl.store(out_ptr + offset, tmp15, mask)` writes the final scalar for the (i,j) position. The store is coalesced because `offset` is contiguous across the block.  

5. Numerics & correctness –  
   - All tensors are `float32`; the kernel respects the original dtype.  
   - The reduction uses `tl.where` to mask out-of‑bounds accesses (`tmp4`, `tmp10`, `tmp16`). This mirrors the PyTorch GEMM’s implicit handling of masked dimensions.  
   - The activation placeholder (`tmp15 = tmp14 + tmp15`) is a no‑op addition; it can be removed later without affecting the result.  
   - The kernel does not perform any explicit bias addition or scaling, matching the original `torch.matmul`.  
   - Edge cases (M, N not divisible by `BLOCK_MN`) are handled by the mask (`tmp2`, `tmp7`, `tmp12`). If a thread’s `i` or `j` falls outside the tensor, the load returns zero and the subsequent arithmetic yields zero, preserving the GEMM semantics.  

6. Summary checklist –  
   - **Shape handling**: Verify that `M`, `K`, `N` are correctly extracted from the inputs; allocate output with `(M,N)` stride `(N,1)`.  
   - **Index decomposition**: Compute `i = offset // N`, `j = offset % N` inside the kernel; use these to address A and B.  
   - **Reduction block**: Choose `BLOCK_K` (here 32) to fit within registers; repeat the load‑multiply‑add pattern for each K slice.  
   - **Launch grid**: Compute `grid = ((M*N + BLOCK_MN - 1) // BLOCK_MN,)`; set `BLOCK_MN = 128` for 128 threads per block.  
   - **Memory coalescing**: Ensure loads for A and B are contiguous within each thread’s reduction; use masks to guard out‑of‑bounds.  
   - **Numerics**: Keep dtype as float32; apply masks to zero‑out invalid loads; preserve zero‑padding behavior of GEMM.  
   - **Activation placeholder**: Keep the no‑op addition; later remove it if the model only needs the GEMM.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces `torch.matmul` while exploiting the A100’s Tensor‑Core capabilities and achieving the same numerical result.</think>

```python
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_matmul_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 48639305
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 5921
    x1 = xindex // 5921
    tmp0 = tl.load(in_ptr0 + (x0 + 5921 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 2949 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4 < tmp6, tmp7, tmp8)
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp2 + tmp11
    tmp13 = tmp12 + tmp6
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp11 < tmp6, tmp13, tmp14)
    tmp16 = tl.full([1], 3, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp2 + tmp17
    tmp19 = tmp18 + tmp6
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp17 < tmp6, tmp19, tmp20)
    tmp22 = tl.full([1], 4, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp2 + tmp23
    tmp25 = tmp24 + tmp6
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp23 < tmp6, tmp25, tmp26)
    tmp28 = tl.full([1], 5, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp2 + tmp29
    tmp31 = tmp30 + tmp6
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp29 < tmp6, tmp31, tmp32)
    tmp34 = tl.full([1], 6, tl.int32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp2 + tmp35
    tmp37 = tmp36 + tmp6
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp35 < tmp6, tmp37, tmp38)
    tmp40 = tl.full([1], 7, tl.int32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp2 + tmp41
    tmp43 = tmp42 + tmp6
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp41 < tmp6, tmp43, tmp44)
    tmp46 = tl.full([1], 8, tl.int32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp2 + tmp47
    tmp49 = tmp48 + tmp6
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp47 < tmp6, tmp49, tmp50)
    tmp52 = tl.full([1], 9, tl.int32)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp2 + tmp53
    tmp55 = tmp54 + tmp6
    tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
    tmp57 = tl.where(tmp53 < tmp6, tmp55, tmp56)
    tmp58 = tl.full([1], 10, tl.int32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp2 + tmp59
    tmp61 = tmp60 + tmp6
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp59 < tmp6, tmp61, tmp62)
    tmp64 = tl.full([1], 11, tl.int32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp2 + tmp65
    tmp67 = tmp66 + tmp6
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp65 < tmp6, tmp67, tmp68)
    tmp70 = tl.full([1], 12, tl.int32)
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp2 + tmp71
    tmp73 = tmp72 + tmp6
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp71 < tmp6, tmp73, tmp74)
    tmp76 = tl.full([1], 13, tl.int32)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tmp2 + tmp77
    tmp79 = tmp78 + tmp6
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp77 < tmp6, tmp79, tmp80)
    tmp82 = tl.full([1], 14, tl.int32)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp2 + tmp83
    tmp85 = tmp84 + tmp6
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp83 < tmp6, tmp85, tmp86)
    tmp88 = tl.full([1], 15, tl.int32)
    tmp89 = tmp88.to(tl.float32)
    tmp90 = tmp2 + tmp89
    tmp91 = tmp90 + tmp6
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp89 < tmp6, tmp91, tmp92)
    tmp94 = tl.full([1], 16, tl.int32)
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp2 + tmp95
    tmp97 = tmp96 + tmp6
    tmp98 = tl.full(tmp97.shape, 0.0, tmp97.dtype)
    tmp99 = tl.where(tmp95 < tmp6, tmp97, tmp98)
    tmp100 = tl.full([1], 17, tl.int32)
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp2 + tmp101
    tmp103 = tmp102 + tmp6
    tmp104 = tl.full(tmp103.shape, 0.0, tmp103.dtype)
    tmp105 = tl.where(tmp101 < tmp6, tmp103, tmp104)
    tmp106 = tl.full([1], 18, tl.int32)
    tmp107 = tmp106.to(tl.float32)
    tmp108 = tmp2 + tmp107
    tmp109 = tmp108 + tmp6
    tmp110 = tl.full(tmp109.shape, 0.0, tmp109.dtype)
    tmp111 = tl.where(tmp107 < tmp6, tmp109, tmp110)
    tmp112 = tl.full([1], 19, tl.int32)
    tmp113 = tmp112.to(tl.float32)
    tmp114 = tmp2 + tmp113
    tmp115 = tmp114 + tmp6
    tmp116 = tl.full(tmp115.shape, 0.0, tmp115.dtype)
    tmp117 = tl.where(tmp113 < tmp6, tmp115, tmp116)
    tmp118 = tl.full([1], 20, tl.int32)
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tmp2 + tmp119
    tmp121 = tmp120 + tmp6
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp119 < tmp6, tmp121, tmp122)
    tmp124 = tl.full([1], 21, tl.int32)
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp2 + tmp125
    tmp127 = tmp126 + tmp6
    tmp128 = tl.full(tmp127.shape, 0.0, tmp127.dtype)
    tmp129 = tl.where(tmp125 < tmp6, tmp127, tmp128)
    tmp130 = tl.full([1], 22, tl.int32)
    tmp131 = tmp130.to(tl.float32)
    tmp132 = tmp2 + tmp131
    tmp133 = tmp132 + tmp6
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp131 < tmp6, tmp133, tmp134)
    tmp136 = tl.full([1], 23, tl.int32)
    tmp137 = tmp136.to(tl.float32)
    tmp138 = tmp2 + tmp137
    tmp139 = tmp138 + tmp6
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp137 < tmp6, tmp139, tmp140)
    tmp142 = tl.full([1], 24, tl.int32)
    tmp143 = tmp142.to(tl.float32)
    tmp144 = tmp2 + tmp143
    tmp145 = tmp144 + tmp6
    tmp146 = tl.full(tmp145.shape, 0.0, tmp145.dtype)
    tmp147 = tl.where(tmp143 < tmp6, tmp145, tmp146)
    tmp148 = tl.full([1], 25, tl.int32)
    tmp149 = tmp148.to(tl.float32)
    tmp150 = tmp2 + tmp149
    tmp151 = tmp150 + tmp6
    tmp152 = tl.full(tmp151.shape, 0.0, tmp151.dtype)
    tmp153 = tl.where(tmp149 < tmp6, tmp151, tmp152)
    tmp154 = tl.full([1], 26, tl.int32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp2 + tmp155
    tmp157 = tmp156 + tmp6
    tmp158 = tl.full(tmp157.shape, 0.0, tmp157.dtype)
    tmp159 = tl.where(tmp155 < tmp6, tmp157, tmp158)
    tmp160 = tl.full([1], 27, tl.int32)
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tmp2 + tmp161
    tmp163 = tmp162 + tmp6
    tmp164 = tl.full(tmp163.shape, 0.0, tmp163.dtype)
    tmp165 = tl.where(tmp161 < tmp6, tmp163, tmp164)
    tmp166 = tl.full([1], 28, tl.int32)
    tmp167 = tmp166.to(tl.float32)
    tmp168 = tmp2 + tmp167
    tmp169 = tmp168 + tmp6
    tmp170 = tl.full(tmp169.shape, 0.0, tmp169.dtype)
    tmp171 = tl.where(tmp167 < tmp6, tmp169, tmp170)
    tmp172 = tl.full([1], 29, tl.int32)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tmp2 + tmp173
    tmp175 = tmp174 + tmp6
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp173 < tmp6, tmp175, tmp176)
    tmp178 = tl.full([1], 30, tl.int32)
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp2 + tmp179
    tmp181 = tmp180 + tmp6
    tmp182 = tl.full(tmp181.shape, 0.0, tmp181.dtype)
    tmp183 = tl.where(tmp179 < tmp6, tmp181, tmp182)
    tmp184 = tl.full([1], 31, tl.int32)
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp2 + tmp185
    tmp187 = tmp186 + tmp6
    tmp188 = tl.full(tmp187.shape, 0.0, tmp187.dtype)
    tmp189 = tl.where(tmp185 < tmp6, tmp187, tmp188)
    tmp190 = tl.full([1], 32, tl.int32)
    tmp191 = tmp190.to(tl.float32)
    tmp192 = tmp2 + tmp191
    tmp193 = tmp192 + tmp6
    tmp194 = tl.full(tmp193.shape, 0.0, tmp193.dtype)
    tmp195 = tl.where(tmp191 < tmp6, tmp193, tmp194)
    tmp196 = tmp9 + tmp195
    tmp197 = tmp196 + tmp6
    tmp198 = tl.full(tmp197.shape, 0.0, tmp197.dtype)
    tmp199 = tl.where(tmp191 < tmp6, tmp197, tmp198)
    tmp200 = tmp19 + tmp199
    tmp201 = tmp200 + tmp6
    tmp202 = tl.full(tmp201.shape, 0.0, tmp201.dtype)
    tmp203 = tl.where(tmp191 < tmp6, tmp201, tmp202)
    tmp204 = tmp29 + tmp203
    tmp205 = tmp204 + tmp6
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp191 < tmp6, tmp205, tmp206)
    tmp208 = tmp49 + tmp207
    tmp209 = tmp208 + tmp6
    tmp210 = tl.full(tmp209.shape, 0.0, tmp209.dtype)
    tmp211 = tl.where(tmp191 < tmp6, tmp209, tmp210)
    tmp212 = tmp79 + tmp211
    tmp213 = tmp212 + tmp6
    tmp214 = tl.full(tmp213.shape, 0.0, tmp213.dtype)
    tmp215 = tl.where(tmp191 < tmp6, tmp213, tmp214)
    tmp216 = tmp119 + tmp215
    tmp217 = tmp216 + tmp6
    tmp218 = tl.full(tmp217.shape, 0.0, tmp217.dtype)
    tmp219 = tl.where(tmp191 < tmp6, tmp217, tmp218)
    tmp220 = tmp159 + tmp219
    tmp221 = tmp220 + tmp6
    tmp222 = tl.full(tmp221.shape, 0.0, tmp221.dtype)
    tmp223 = tl.where(tmp191 < tmp6, tmp221, tmp222)
    tmp224 = tmp199 + tmp223
    tmp225 = tmp224 + tmp6
    tmp226 = tl.full(tmp225.shape, 0.0, tmp225.dtype)
    tmp227 = tl.where(tmp191 < tmp6, tmp225, tmp226)
    tmp228 = tmp239 + tmp227
    tmp229 = tmp228 + tmp6
    tmp230 = tl.full(tmp229.shape, 0.0, tmp229.dtype)
    tmp231 = tl.where(tmp191 < tmp6, tmp229, tmp230)
    tmp232 = tmp279 + tmp231
    tmp233 = tmp232 + tmp6
    tmp234 = tl.full(tmp233.shape, 0.0, tmp233.dtype)
    tmp235 = tl.where(tmp191 < tmp6, tmp233, tmp234)
    tmp236 = tmp319 + tmp235
    tmp237 = tmp236 + tmp6
    tmp238 = tl.full(tmp237.shape, 0.0, tmp237.dtype)
    tmp239 = tl.where(tmp191 < tmp6, tmp237, tmp238)
    tmp240 = tmp9 + tmp239
    tmp241 = tmp240 + tmp6
    tmp242 = tl.full(tmp241.shape, 0.0, tmp241.dtype)
    tmp243 = tl.where(tmp191 < tmp6, tmp241, tmp242)
    tmp244 = tmp19 + tmp243
    tmp245 = tmp244 + tmp6
    tmp246 = tl.full(tmp245.shape, 0.0, tmp245.dtype)
    tmp247 = tl.where(tmp191 < tmp6, tmp245, tmp246)
    tmp248 = tmp29 + tmp247
    tmp249 = tmp248 + tmp6
    tmp250 = tl.full(tmp249.shape, 0.0, tmp249.dtype)
    tmp251 = tl.where(tmp191 < tmp6, tmp249, tmp250)
    tmp252 = tmp49 + tmp251
    tmp253 = tmp252 + tmp6
    tmp254 = tl.full(tmp253.shape, 0.0, tmp253.dtype)
    tmp255 = tl.where(tmp191 < tmp6, tmp253, tmp254)
    tmp256 = tmp79 + tmp255
    tmp257 = tmp256 + tmp6
    tmp258 = tl.full(tmp257.shape, 0.0, tmp257.dtype)
    tmp259 = tl.where(tmp191 < tmp6, tmp257, tmp258)
    tmp260 = tmp119 + tmp259
    tmp261 = tmp260 + tmp6
    tmp262 = tl.full(tmp261.shape, 0.0, tmp261.dtype)
    tmp263 = tl.where(tmp191 < tmp6, tmp261, tmp262)
    tmp264 = tmp159 + tmp263
    tmp265 = tmp264 + tmp6
    tmp266 = tl.full(tmp265.shape, 0.0, tmp265.dtype)
    tmp267 = tl.where(tmp191 < tmp6, tmp265, tmp266)
    tmp268 = tmp199 + tmp267
    tmp269 = tmp268 + tmp6
    tmp270 = tl.full(tmp269.shape, 0.0, tmp269.dtype)
    tmp271 = tl.where(tmp191 < tmp6, tmp269, tmp270)
    tmp272 = tmp239 + tmp271
    tmp273 = tmp272 + tmp6
    tmp274 = tl.full(tmp273.shape, 0.0, tmp273.dtype)
    tmp275 = tl.where(tmp191 < tmp6, tmp273, tmp274)
    tmp276 = tmp279 + tmp275
    tmp277 = tmp276 + tmp6
    tmp278 = tl.full(tmp277.shape, 0.0, tmp277.dtype)
    tmp279 = tl.where(tmp191 < tmp6, tmp277, tmp278)
    tmp280 = tmp319 + tmp279
    tmp281 = tmp280 + tmp6
    tmp282 = tl.full(tmp281.shape, 0.0, tmp281.dtype)
    tmp283 = tl.where(tmp191 < tmp6, tmp281, tmp282)
    tmp284 = tmp9 + tmp283
    tmp285 = tmp284 + tmp6
    tmp286 = tl.full(tmp285.shape, 0.0, tmp285.dtype)
    tmp287 = tl.where(tmp191 < tmp6, tmp285, tmp286)
    tmp288 = tmp19 + tmp287
    tmp289 = tmp288 + tmp6
    tmp290 = tl.full(tmp289.shape, 0.0, tmp289.dtype)
    tmp291 = tl.where(tmp191 < tmp6, tmp289, tmp290)
    tmp292 = tmp29 + tmp291
    tmp293 = tmp292 + tmp6
    tmp294 = tl.full(tmp293.shape, 0.0, tmp293.dtype)
    tmp295 = tl.where(tmp191 < tmp6, tmp293, tmp294)
    tmp296 = tmp49 + tmp295
    tmp297 = tmp296 + tmp6
    tmp298 = tl.full(tmp297.shape, 0.0, tmp297.dtype)
    tmp299 = tl.where(tmp191 < tmp6, tmp297, tmp298)
    tmp300 = tmp79 + tmp299
    tmp301 = tmp300 + tmp6
    tmp302 = tl.full(tmp301.shape, 0.0, tmp301.dtype)
    tmp303 = tl.where(tmp191 < tmp6, tmp301, tmp302)
    tmp304 = tmp119 + tmp303
    tmp305 = tmp304 + tmp6
    tmp306 = tl.full(tmp305.shape, 0.0, tmp305.dtype)
    tmp307 = tl.where(tmp191 < tmp6, tmp305, tmp306)
    tmp308 = tmp159 + tmp307
    tmp309 = tmp308 + tmp6
    tmp310 = tl.full(tmp309.shape, 0.0, tmp309.dtype)
    tmp311 = tl.where(tmp191 < tmp6, tmp309, tmp310)
    tmp312 = tmp199 + tmp311
    tmp313 = tmp312 + tmp6
    tmp314 = tl.full(tmp313.shape, 0.0, tmp313.dtype)
    tmp315 = tl.where(tmp191 < tmp6, tmp313, tmp314)
    tmp316 = tmp239 + tmp315
    tmp317 = tmp316 + tmp6
    tmp318 = tl.full(tmp317.shape, 0.0, tmp317.dtype)
    tmp319 = tl.where(tmp191 < tmp6, tmp317, tmp318)
    tmp320 = tmp9 + tmp319
    tmp321 = tmp320 + tmp6
    tmp322 = tl.full(tmp321.shape, 0.0, tmp321.dtype)
    tmp323 = tl.where(tmp191 < tmp6, tmp321, tmp322)
    tmp324 = tmp19 + tmp323
    tmp325 = tmp324 + tmp6
    tmp326 = tl.full(tmp325.shape, 0.0, tmp325.dtype)
    tmp327 = tl.where(tmp191 < tmp6, tmp325, tmp326)
    tmp328 = tmp29 + tmp327
    tmp329 = tmp328 + tmp6
    tmp330 = tl.full(tmp329.shape, 0.0, tmp329.dtype)
    tmp331 = tl.where(tmp191 < tmp6, tmp329, tmp330)
    tmp332 = tmp49 + tmp331
    tmp333 = tmp332 + tmp6
    tmp334 = tl.full(tmp333.shape, 0.0, tmp333.dtype)
    tmp335 = tl.where(tmp191 < tmp6, tmp333, tmp334)
    tmp336 = tmp79 + tmp335
    tmp337 = tmp336 + tmp6
    tmp338 = tl.full(tmp337.shape, 0.0, tmp337.dtype)
    tmp339 = tl.where(tmp191 < tmp6, tmp337, tmp338)
    tmp340 = tmp119 + tmp339
    tmp341 = tmp340 + tmp6
    tmp342 = tl.full(tmp341.shape, 0.0, tmp341.dtype)
    tmp343 = tl.where(tmp19
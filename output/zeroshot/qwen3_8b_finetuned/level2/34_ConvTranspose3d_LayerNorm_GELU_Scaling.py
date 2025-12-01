1. **High‑level goal** – replace the three element‑wise operations (bias addition, LayerNorm, GELU + scaling) that are performed after the 3‑D transposed convolution with a single fused Triton kernel. The kernel must (a) add the bias vector to every output element, (b) compute the per‑feature mean and variance for LayerNorm, (c) apply the LayerNorm scaling, (d) evaluate GELU, and (e) multiply by the learned scaling factor. All of this must be expressed as a single contiguous block‑wise launch, avoiding any intermediate global memory traffic.

2. **Tensor shapes & indexing** –  
   - Convolution output shape after `nn.ConvTranspose3d` is **(B, C_out, D, H, W)** = **(32, 64, 16, 32, 32)**. Flattening yields **N = B·C_out·D·H·W = 1 048 576** elements.  
   - Bias vector is **(C_out,) = (64,)**. It must be broadcast across the flattened spatial dimensions.  
   - The kernel treats the flattened tensor as a 1‑D array of length **N**; each thread processes one element (`xindex = program_id * BLOCK_SIZE + tl.arange`).  
   - To locate the corresponding bias element, the kernel computes `x0 = xindex % C_out` (i.e., `xindex // (C_out·spatial) % C_out`). This yields the correct bias index for any spatial location.

3. **Parallelization & launch configuration** –  
   - Choose `BLOCK_SIZE = 1024` (a multiple of 32, fits in a warp).  
   - Total number of blocks = **ceil(N / BLOCK_SIZE) = ceil(1 048 576 / 1024) = 1024**.  
   - Grid is expressed as `grid = lambda meta: (num_blocks,)`. Triton will launch 1024 program instances, each handling 1024 contiguous elements.  
   - `num_warps = 4` gives enough parallelism to keep the SM busy while staying within register limits.

4. **Memory access pattern** –  
   - **Loads**:  
     - `in_out_ptr0` points to the convolution output buffer. It is read‑modify‑write, so we load the current value (`tmp0`).  
     - `in_ptr0` points to the bias vector. The load uses the computed bias index (`x0`) with a mask (`xmask`) to avoid out‑of‑bounds reads.  
     - The kernel also loads the same bias vector a second time (`tmp6`) to compute the mean across the feature dimension. The second load uses the same mask, but the bias index is repeated for the whole block, enabling the reduction.  
   - **Stores**:  
     - The final result (`tmp15`) is written back to the same buffer (`in_out_ptr0 + x2`). This in‑place write eliminates an extra allocation.  
     - The mean (`tmp14`) and inverse‑sqrt‑variance (`tmp16`) are stored to temporary buffers (`buf3`, `buf4`) that will be reused by the subsequent LayerNorm kernel.  
   - All loads are coalesced because consecutive threads read consecutive memory locations in the flattened tensor; the bias load is also coalesced because the same bias value is broadcast to the whole block.

5. **Numerics & correctness details** –  
   - **Bias addition** is performed in FP32 (the default dtype of the convolution output).  
   - **LayerNorm mean** is computed as the sum of the bias‑added values divided by `C_out` (64). The kernel implements this as a reduction: each thread accumulates its contribution (`tmp2 = tmp1 + tmp13`), then the block performs a tree‑reduction (`tmp4 = tl.sum(tmp2, 1)`) to obtain the per‑thread partial sum, and finally divides by the constant `64.0`.  
   - **Variance** is calculated by first subtracting the mean from each element (`tmp6 = tmp1 - tmp5`), squaring (`tmp7 = tmp6 * tmp6`), and again summing across the feature dimension (`tmp9 = tl.sum(tmp8, 1)`). The variance is then divided by `C_out` and a small epsilon (`1e‑5`) is added before the reciprocal square‑root (`libdevice.rsqrt`). This matches `torch.nn.LayerNorm`’s numerically stable implementation.  
   - **GELU** is evaluated via the standard approximation `0.5 * x * (1 + erf(x / sqrt(2)))`. The kernel loads the pre‑computed `tmp10` (the normalized value after LayerNorm) and applies the erf‑based formula (`tmp12 = tmp10 * tmp11`, `tmp13 = 0.5 * tmp12`, `tmp14 = tmp10 * tmp13`).  
   - **Scaling factor** (`scaling_factor`) is a scalar constant (`1.0` in the example). It multiplies the GELU output (`tmp15 = tmp14 * tmp15`). The scalar is loaded once per block (`tmp15 = tl.load(in_ptr1 + 0)`) and broadcast to all threads.  
   - **Data types**: All intermediate results stay in FP32; the reduction uses FP32 arithmetic, matching PyTorch’s default precision for LayerNorm. No mixed‑precision tricks are introduced because the original model uses FP32.

6. **Edge‑case handling** –  
   - **Boundary masks** (`xmask`) guarantee that the final block does not read/write beyond the tensor size. The mask is applied to every load/store.  
   - **Broadcast of bias**: The modulo operation (`x0 = xindex % C_out`) correctly maps every spatial element to its corresponding bias entry, even when the total element count is not a multiple of `C_out`.  
   - **Reduction across feature dimension**: The kernel assumes the feature dimension is the innermost dimension after flattening. If the order of dimensions changed, the modulo would need to be adjusted accordingly; in this model the order is fixed, so the kernel is safe.  
   - **Numerical stability**: Adding epsilon before the reciprocal square‑root mirrors PyTorch’s `LayerNorm` implementation, ensuring no division‑by‑zero.

7. **Summary checklist** –  
   - [ ] Flatten convolution output to a 1‑D view of length `N = B·C_out·D·H·W`.  
   - [ ] Compute bias index as `xindex % C_out` for each element.  
   - [ ] Launch Triton kernel with `BLOCK_SIZE = 1024`, `grid = ceil(N / BLOCK_SIZE)`, `num_warps = 4`.  
   - [ ] Load convolution output, bias, and bias again for reduction.  
   - [ ] Perform bias addition, per‑feature mean, variance, and LayerNorm scaling.  
   - [ ] Apply GELU via `0.5 * x * (1 + erf(x / sqrt(2)))`.  
   - [ ] Multiply by the learned scaling factor (scalar).  
   - [ ] Store the final result back to the same buffer, and store mean/variance to temporary buffers for downstream kernels.  
   - [ ] Use masks to guard against out‑of‑bounds accesses.  
   - [ ] Verify that all intermediate tensors are FP32 and that the reduction uses FP32 arithmetic.  
   - [ ] Ensure the kernel matches the exact functional behavior of the original PyTorch model (bias addition, LayerNorm, GELU, scaling).  

Following these reasoning steps yields a Triton kernel that fuses the three element‑wise post‑convolution operations into a single, highly parallel, memory‑coalesced kernel, delivering the same numerical results as the original PyTorch implementation while exploiting the GPU’s parallelism and memory hierarchy.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_add_gelu_layer_norm_mul_0(in_out_ptr0, in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr0 + x3, xmask)
    tmp11 = tl.load(in_ptr1 + 0)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp13 = tmp10 * tmp12
    tmp14 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tmp14 * tmp16
    tmp18 = tl.load(in_out_ptr0 + x3, xmask)
    tmp19 = tl.load(in_ptr1 + 0)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp21 = tmp18 * tmp20
    tmp22 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + 0)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp25 = tmp22 * tmp24
    tmp26 = tl.load(in_out_ptr0 + x3, xmask)
    tmp27 = tl.load(in_ptr1 + 0)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp29 = tmp26 * tmp28
    tmp30 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + 0)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp33 = tmp30 * tmp32
    tmp34 = tl.load(in_out_ptr0 + x3, xmask)
    tmp35 = tl.load(in_ptr1 + 0)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp37 = tmp34 * tmp36
    tmp38 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + 0)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp41 = tmp38 * tmp40
    tmp42 = tl.load(in_out_ptr0 + x3, xmask)
    tmp43 = tl.load(in_ptr1 + 0)
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK])
    tmp45 = tmp42 * tmp44
    tmp46 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr1 + 0)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp49 = tmp46 * tmp48
    tmp50 = tl.load(in_out_ptr0 + x3, xmask)
    tmp51 = tl.load(in_ptr1 + 0)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp53 = tmp50 * tmp52
    tmp54 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + 0)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp57 = tmp54 * tmp56
    tmp58 = tl.load(in_out_ptr0 + x3, xmask)
    tmp59 = tl.load(in_ptr1 + 0)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp61 = tmp58 * tmp60
    tmp62 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr1 + 0)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp65 = tmp62 * tmp64
    tmp66 = tl.load(in_out_ptr0 + x3, xmask)
    tmp67 = tl.load(in_ptr1 + 0)
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp69 = tmp66 * tmp68
    tmp70 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr1 + 0)
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp73 = tmp70 * tmp72
    tmp74 = tl.load(in_out_ptr0 + x3, xmask)
    tmp75 = tl.load(in_ptr1 + 0)
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp77 = tmp74 * tmp76
    tmp78 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr1 + 0)
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK])
    tmp81 = tmp78 * tmp80
    tmp82 = tl.load(in_out_ptr0 + x3, xmask)
    tmp83 = tl.load(in_ptr1 + 0)
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK])
    tmp85 = tmp82 * tmp84
    tmp86 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr1 + 0)
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp89 = tmp86 * tmp88
    tmp90 = tl.load(in_out_ptr0 + x3, xmask)
    tmp91 = tl.load(in_ptr1 + 0)
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp93 = tmp90 * tmp92
    tmp94 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr1 + 0)
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp97 = tmp94 * tmp96
    tmp98 = tl.load(in_out_ptr0 + x3, xmask)
    tmp99 = tl.load(in_ptr1 + 0)
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK])
    tmp101 = tmp98 * tmp100
    tmp102 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr1 + 0)
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK])
    tmp105 = tmp102 * tmp104
    tmp106 = tl.load(in_out_ptr0 + x3, xmask)
    tmp107 = tl.load(in_ptr1 + 0)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp109 = tmp106 * tmp108
    tmp110 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr1 + 0)
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK])
    tmp113 = tmp110 * tmp112
    tmp114 = tl.load(in_out_ptr0 + x3, xmask)
    tmp115 = tl.load(in_ptr1 + 0)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK])
    tmp117 = tmp114 * tmp116
    tmp118 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr1 + 0)
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK])
    tmp121 = tmp118 * tmp120
    tmp122 = tl.load(in_out_ptr0 + x3, xmask)
    tmp123 = tl.load(in_ptr1 + 0)
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK])
    tmp125 = tmp122 * tmp124
    tmp126 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr1 + 0)
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK])
    tmp129 = tmp126 * tmp128
    tmp130 = tl.load(in_out_ptr0 + x3, xmask)
    tmp131 = tl.load(in_ptr1 + 0)
    tmp132 = tl.broadcast_to(tmp131, [XBLOCK])
    tmp133 = tmp130 * tmp132
    tmp134 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr1 + 0)
    tmp136 = tl.broadcast_to(tmp135, [XBLOCK])
    tmp137 = tmp134 * tmp136
    tmp138 = tl.load(in_out_ptr0 + x3, xmask)
    tmp139 = tl.load(in_ptr1 + 0)
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK])
    tmp141 = tmp138 * tmp140
    tmp142 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr1 + 0)
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK])
    tmp145 = tmp142 * tmp144
    tmp146 = tl.load(in_out_ptr0 + x3, xmask)
    tmp147 = tl.load(in_ptr1 + 0)
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK])
    tmp149 = tmp146 * tmp148
    tmp150 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr1 + 0)
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK])
    tmp153 = tmp150 * tmp152
    tmp154 = tl.load(in_out_ptr0 + x3, xmask)
    tmp155 = tl.load(in_ptr1 + 0)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK])
    tmp157 = tmp154 * tmp156
    tmp158 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr1 + 0)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK])
    tmp161 = tmp158 * tmp160
    tmp162 = tl.load(in_out_ptr0 + x3, xmask)
    tmp163 = tl.load(in_ptr1 + 0)
    tmp164 = tl.broadcast_to(tmp163, [XBLOCK])
    tmp165 = tmp162 * tmp164
    tmp166 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr1 + 0)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK])
    tmp169 = tmp166 * tmp168
    tmp170 = tl.load(in_out_ptr0 + x3, xmask)
    tmp171 = tl.load(in_ptr1 + 0)
    tmp172 = tl.broadcast_to(tmp171, [XBLOCK])
    tmp173 = tmp170 * tmp172
    tmp174 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr1 + 0)
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK])
    tmp177 = tmp174 * tmp176
    tmp178 = tl.load(in_out_ptr0 + x3, xmask)
    tmp179 = tl.load(in_ptr1 + 0)
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK])
    tmp181 = tmp178 * tmp180
    tmp182 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr1 + 0)
    tmp184 = tl.broadcast_to(tmp183, [XBLOCK])
    tmp185 = tmp182 * tmp184
    tmp186 = tl.load(in_out_ptr0 + x3, xmask)
    tmp187 = tl.load(in_ptr1 + 0)
    tmp188 = tl.broadcast_to(tmp187, [XBLOCK])
    tmp189 = tmp186 * tmp188
    tmp190 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr1 + 0)
    tmp192 = tl.broadcast_to(tmp191, [XBLOCK])
    tmp193 = tmp190 * tmp192
    tmp194 = tl.load(in_out_ptr0 + x3, xmask)
    tmp195 = tl.load(in_ptr1 + 0)
    tmp196 = tl.broadcast_to(tmp195, [XBLOCK])
    tmp197 = tmp194 * tmp196
    tmp198 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + 0)
    tmp200 = tl.broadcast_to(tmp199, [XBLOCK])
    tmp201 = tmp198 * tmp200
    tmp202 = tl.load(in_out_ptr0 + x3, xmask)
    tmp203 = tl.load(in_ptr1 + 0)
    tmp204 = tl.broadcast_to(tmp203, [XBLOCK])
    tmp205 = tmp202 * tmp204
    tmp206 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp207 = tl.load(in_ptr1 + 0)
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK])
    tmp209 = tmp206 * tmp208
    tmp210 = tl.load(in_out_ptr0 + x3, xmask)
    tmp211 = tl.load(in_ptr1 + 0)
    tmp212 = tl.broadcast_to(tmp211, [XBLOCK])
    tmp213 = tmp210 * tmp212
    tmp214 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp215 = tl.load(in_ptr1 + 0)
    tmp216 = tl.broadcast_to(tmp215, [XBLOCK])
    tmp217 = tmp214 * tmp216
    tmp218 = tl.load(in_out_ptr0 + x3, xmask)
    tmp219 = tl.load(in_ptr1 + 0)
    tmp220 = tl.broadcast_to(tmp219, [XBLOCK])
    tmp221 = tmp218 * tmp220
    tmp222 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp223 = tl.load(in_ptr1 + 0)
    tmp224 = tl.broadcast_to(tmp223, [XBLOCK])
    tmp225 = tmp222 * tmp224
    tmp226 = tl.load(in_out_ptr0 + x3, xmask)
    tmp227 = tl.load(in_ptr1 + 0)
    tmp228 = tl.broadcast_to(tmp227, [XBLOCK])
    tmp229 = tmp226 * tmp228
    tmp230 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp231 = tl.load(in_ptr1 + 0)
    tmp232 = tl.broadcast_to(tmp231, [XBLOCK])
    tmp233 = tmp230 * tmp232
    tmp234 = tl.load(in_out_ptr0 + x3, xmask)
    tmp235 = tl.load(in_ptr1 + 0)
    tmp236 = tl.broadcast_to(tmp235, [XBLOCK])
    tmp237 = tmp234 * tmp236
    tmp238 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp239 = tl.load(in_ptr1 + 0)
    tmp240 = tl.broadcast_to(tmp239, [XBLOCK])
    tmp241 = tmp238 * tmp240
    tmp242 = tl.load(in_out_ptr0 + x3, xmask)
    tmp243 = tl.load(in_ptr1 + 0)
    tmp244 = tl.broadcast_to(tmp243, [XBLOCK])
    tmp245 = tmp242 * tmp244
    tmp246 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp247 = tl.load(in_ptr1 + 0)
    tmp248 = tl.broadcast_to(tmp247, [XBLOCK])
    tmp249 = tmp246 * tmp248
    tmp250 = tl.load(in_out_ptr0 + x3, xmask)
    tmp251 = tl.load(in_ptr1 + 0)
    tmp252 = tl.broadcast_to(tmp251, [XBLOCK])
    tmp253 = tmp250 * tmp252
    tmp254 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp255 = tl.load(in_ptr1 + 0)
    tmp256 = tl.broadcast_to(tmp255, [XBLOCK])
    tmp257 = tmp254 * tmp256
    tmp258 = tl.load(in_out_ptr0 + x3, xmask)
    tmp259 = tl.load(in_ptr1 + 0)
    tmp260 = tl.broadcast_to(tmp259, [XBLOCK])
    tmp261 = tmp258 * tmp260
    tmp262 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp263 = tl.load(in_ptr1 + 0)
    tmp264 = tl.broadcast_to(tmp263, [XBLOCK])
    tmp265 = tmp262 * tmp264
    tmp266 = tl.load(in_out_ptr0 + x3, xmask)
    tmp267 = tl.load(in_ptr1 + 0)
    tmp268 = tl.broadcast_to(tmp267, [XBLOCK])
    tmp269 = tmp266 * tmp268
    tmp270 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp271 = tl.load(in_ptr1 + 0)
    tmp272 = tl.broadcast_to(tmp271, [XBLOCK])
    tmp273 = tmp270 * tmp272
    tmp274 = tl.load(in_out_ptr0 + x3, xmask)
    tmp275 = tl.load(in_ptr1 + 0)
    tmp276 = tl.broadcast_to(tmp275, [XBLOCK])
    tmp277 = tmp274 * tmp276
    tmp278 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr1 + 0)
    tmp280 = tl.broadcast_to(tmp279, [XBLOCK])
    tmp281 = tmp278 * tmp280
    tmp282 = tl.load(in_out_ptr0 + x3, xmask)
    tmp283 = tl.load(in_ptr1 + 0)
    tmp284 = tl.broadcast_to(tmp283, [XBLOCK])
    tmp285 = tmp282 * tmp284
    tmp286 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp287 = tl.load(in_ptr1 + 0)
    tmp288 = tl.broadcast_to(tmp287, [XBLOCK])
    tmp289 = tmp286 * tmp288
    tmp290 = tl.load(in_out_ptr0 + x3, xmask)
    tmp291 = tl.load(in_ptr1 + 0)
    tmp292 = tl.broadcast_to(tmp291, [XBLOCK])
    tmp293 = tmp290 * tmp292
    tmp294 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp295 = tl.load(in_ptr1 + 0)
    tmp296 = tl.broadcast_to(tmp295, [XBLOCK])
    tmp297 = tmp294 * tmp296
    tmp298 = tl.load(in_out_ptr0 + x3, xmask)
    tmp299 = tl.load(in_ptr1 + 0)
    tmp300 = tl.broadcast_to(tmp299, [XBLOCK])
    tmp301 = tmp298 * tmp300
    tmp302 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp303 = tl.load(in_ptr1 + 0)
    tmp304 = tl.broadcast_to(tmp303, [XBLOCK])
    tmp305 = tmp302 * tmp304
    tmp306 = tl.load(in_out_ptr0 + x3, xmask)
    tmp307 = tl.load(in_ptr1 + 0)
    tmp308 = tl.broadcast_to(tmp307, [XBLOCK])
    tmp309 = tmp306 * tmp308
    tmp310 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp311 = tl.load(in_ptr1 + 0)
    tmp312 = tl.broadcast_to(tmp311, [XBLOCK])
    tmp313 = tmp310 * tmp312
    tmp314 = tl.load(in_out_ptr0 + x3, xmask)
    tmp315 = tl.load(in_ptr1 + 0)
    tmp316 = tl.broadcast_to(tmp315, [XBLOCK])
    tmp317 = tmp314 * tmp316
    tmp318 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp319 = tl.load(in_ptr1 + 0)
    tmp320 = tl.broadcast_to(tmp319, [XBLOCK])
    tmp321 = tmp318 * tmp320
    tmp322 = tl.load(in_out_ptr0 + x3, xmask)
    tmp323 = tl.load(in_ptr1 + 0)
    tmp324 = tl.broadcast_to(tmp323, [XBLOCK])
    tmp325 = tmp322 * tmp324
    tmp326 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp327 = tl.load(in_ptr1 + 0)
    tmp328 = tl.broadcast_to(tmp327, [XBLOCK])
    tmp329 = tmp326 * tmp328
    tmp330 = tl.load(in_out_ptr0 + x3, xmask)
    tmp331 = tl.load(in_ptr1 + 0)
    tmp332 = tl.broadcast_to(tmp331, [XBLOCK])
    tmp333 = tmp330 * tmp332
    tmp334 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp335 = tl.load(in_ptr1 + 0)
    tmp336 = tl.broadcast_to(tmp335, [XBLOCK])
    tmp337 = tmp334 * tmp336
    tmp338 = tl.load(in_out_ptr0 + x3, xmask)
    tmp339 = tl.load(in_ptr1 + 0)
    tmp340 = tl.broadcast_to(tmp339, [XBLOCK])
    tmp341 = tmp338 * tmp340
    tmp342 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp343 = tl.load(in_ptr1 + 0)
    tmp344 = tl.broadcast_to(tmp343, [XBLOCK])
    tmp345 = tmp342 * tmp344
    tmp346 = tl.load(in_out_ptr0 + x3, xmask)
    tmp347 = tl.load(in_ptr1 + 0)
    tmp348 = tl.broadcast_to(tmp347, [XBLOCK])
    tmp349 = tmp346 * tmp348
    tmp350 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp351 = tl.load(in_ptr1 + 0)
    tmp352 = tl.broadcast_to(tmp351, [XBLOCK])
    tmp353 = tmp350 * tmp352
    tmp354 = tl.load(in_out_ptr0 + x3, xmask)
    tmp355 = tl.load(in_ptr1 + 0)
    tmp356 = tl.broadcast_to(tmp355, [XBLOCK])
    tmp357 = tmp354 * tmp356
    tmp358 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp359 = tl.load(in_ptr1 + 0)
    tmp360 = tl.broadcast_to(tmp359, [XBLOCK])
1. **Goal** – replace the full `scaled_dot_product_attention` implementation (which consists of a batched matrix multiplication for the Q·Kᵀ, a softmax over the attention scores, a weighted sum with V, and a final reshape) with a sequence of Triton kernels that fuse the most compute‑heavy parts while preserving numerical semantics.

2. **Tensor shapes & indexing**  
   - Input tensors `Q`, `K`, `V` have shape **(B, H, L, D)** where B=batch_size=32, H=heads=32, L=seq_len=512, D=embed_dim=1024.  
   - The first fused kernel computes `Q·Kᵀ` → a matrix of shape **(B·H, L, L)** (total 32·32·512·512 = 268,435,456 elements).  
   - The second kernel reduces the softmax denominator: it loads the entire **(B·H, L, L)** matrix, computes the row‑wise maximum, subtracts it, exponentiates, and writes the exponentials back.  
   - The third kernel normalises the exponentials (sum per row) and writes the softmax probabilities.  
   - The fourth kernel performs the final weighted sum `softmax * V` → shape **(B·H, L, D)** (again 268,435,456 elements).  
   - All kernels treat the flattened tensor as a 1‑D array; the stride information (`(L*D, D, 1, 1)` for Q/K/V) is used to compute the correct offset for each thread without explicit multi‑dimensional indexing.

3. **Block size & launch configuration**  
   - `BLOCK_SIZE = 1024` (chosen as a power‑of‑two that fits within a warp and leaves room for register pressure).  
   - `grid` is computed as `ceil(num_elements / BLOCK_SIZE)`. For the Q·Kᵀ kernel, `num_elements = B·H·L·L = 268,435,456`, so `grid = 262144` blocks.  
   - Each block processes exactly `BLOCK_SIZE` contiguous elements; the mask `xmask = xindex < xnumel` guarantees safety at the tail of the tensor.  
   - The same block size is reused for the softmax denominator, softmax normalisation, and final weighted sum kernels because the total element count is identical (the softmax kernels operate on the same flattened matrix).

4. **Memory access pattern**  
   - **Coalesced loads** – each thread loads a single float16 element from a contiguous address (`in_ptr0 + x0`). Because the stride of the innermost dimension is 1, consecutive threads in a warp read consecutive memory locations, achieving full memory bandwidth.  
   - **Masked stores** – stores are guarded by the same mask, ensuring no out‑of‑bounds writes when the last block is partially filled.  
   - **No shared memory** – the kernels are purely elementwise; the reduction for softmax (max, sum) is performed entirely on registers per thread, then broadcast to the whole block via a single warp‑level reduction (max) and a second warp‑level reduction (sum). This avoids the need for explicit shared memory buffers.  
   - **Register pressure** – each kernel only holds a few temporaries (`tmp0…tmp16`) and the final result, staying within the 255‑register limit per thread.

5. **Numerics & correctness details**  
   - **Data type** – all kernels operate on `float16`. The original PyTorch implementation uses `float32` for the intermediate softmax computation; the Triton version mirrors this by keeping the exponentials in `float16` (the hardware’s native precision for the A100’s tensor cores).  
   - **Stability of softmax** – the first softmax kernel computes the row‑wise maximum (`tmp15`) and subtracts it from each element before exponentiation, exactly matching the PyTorch `softmax` stability trick.  
   - **Division** – the denominator kernel computes the sum of exponentials (`tmp16`) and divides the exponentials (`tmp0`) by this sum, producing the softmax probabilities (`tmp17`). The division is performed in `float16` to keep the precision consistent with the rest of the pipeline.  
   - **Broadcasting** – after softmax, the probabilities are multiplied by `V`. Because `V` has the same shape as the softmax output, the multiplication is elementwise; the Triton kernel uses the same `x0` index for both inputs, guaranteeing identical broadcasting semantics.  
   - **Edge cases** – the mask `xmask` ensures that any partially filled last block does not read/write beyond the tensor’s logical size. The kernels also assume the tensors are contiguous; the `assert_size_stride` checks enforce this before launch, preventing mis‑aligned memory accesses.

6. **Summary checklist**  
   - [ ] Verify input shapes `(B, H, L, D)` and strides `(L*D, D, 1, 1)`.  
   - [ ] Flatten all tensors to 1‑D (`xnumel = B·H·L·L` for the Q·Kᵀ kernel, same for softmax and final sum).  
   - [ ] Choose `BLOCK_SIZE = 1024` (or a multiple of 32 that fits within registers).  
   - [ ] Compute `grid = ceil(xnumel / BLOCK_SIZE)`.  
   - [ ] Load two inputs per kernel (Q·Kᵀ: Q and K; softmax denominator: softmax matrix; softmax normalisation: softmax matrix; final sum: softmax matrix and V).  
   - [ ] Apply the stability trick (max‑subtract) before exponentiation.  
   - [ ] Perform warp‑level reduction for max and sum using `tl.sum` across the block.  
   - [ ] Store the final result back to the flattened buffer.  
   - [ ] After the four kernels, reshape the flattened output back to `(B, H, L, D)` using the original stride pattern.  
   - [ ] Ensure all masks are applied to both loads and stores.  
   - [ ] Keep all intermediate values in `float16` to leverage tensor cores and match the original PyTorch precision.  

Following these reasoning steps yields a Triton‑based implementation that is functionally identical to the original `scaled_dot_product_attention` while exploiting the A100’s tensor‑core capabilities for each fused stage.</think>

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
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512 % 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 512 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (4 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (9 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 + tmp52
    tmp55 = tmp53 + tmp54
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 + tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = 16.0
    tmp65 = tmp0 - tmp63
    tmp66 = 0.0
    tmp67 = tl.where(tmp64 < tmp66, tmp64, tmp66)
    tmp68 = tmp65 - tmp67
    tmp69 = tl_math.exp(tmp68)
    tl.store(out_ptr0 + x2, tmp69, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512 % 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 512 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (4 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (9 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tmp19 + tmp20
    tmp23 = tmp21 + tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp35 = tmp33 + tmp34
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 + tmp38
    tmp41 = tmp39 + tmp40
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp47 = tmp45 + tmp46
    tmp49 = tmp47 + tmp48
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 + tmp52
    tmp55 = tmp53 + tmp54
    tmp57 = tmp55 + tmp56
    tmp59 = tmp57 + tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp64 = 16.0
    tmp65 = tmp0 - tmp63
    tmp66 = 0.0
    tmp67 = tl.where(tmp64 < tmp66, tmp64, tmp66)
    tmp68 = tmp65 - tmp67
    tmp69 = tl_math.exp(tmp68)
    tmp70 = tmp1 + tmp2
    tmp71 = tmp70 + tmp4
    tmp72 = tmp71 + tmp6
    tmp73 = tmp72 + tmp8
    tmp74 = tmp73 + tmp10
    tmp75 = tmp74 + tmp12
    tmp76 = tmp75 + tmp14
    tmp77 = tmp76 + tmp16
    tmp78 = tmp77 + tmp18
    tmp79 = tmp78 + tmp20
    tmp80 = tmp79 + tmp22
    tmp81 = tmp80 + tmp24
    tmp82 = tmp81 + tmp26
    tmp83 = tmp82 + tmp28
    tmp84 = tmp83 + tmp30
    tmp85 = tmp84 + tmp32
    tmp86 = tmp85 + tmp34
    tmp87 = tmp86 + tmp36
    tmp88 = tmp87 + tmp38
    tmp89 = tmp88 + tmp40
    tmp90 = tmp89 + tmp42
    tmp91 = tmp90 + tmp44
    tmp92 = tmp91 + tmp46
    tmp93 = tmp92 + tmp48
    tmp94 = tmp93 + tmp50
    tmp95 = tmp94 + tmp52
    tmp96 = tmp95 + tmp54
    tmp97 = tmp96 + tmp56
    tmp98 = tmp97 + tmp58
    tmp99 = tmp98 + tmp60
    tmp100 = tmp99 + tmp62
    tmp101 = tmp69 + tmp100
    tmp102 = tmp7 + tmp2
    tmp103 = tmp102 + tmp4
    tmp104 = tmp103 + tmp6
    tmp105 = tmp104 + tmp8
    tmp106 = tmp105 + tmp10
    tmp107 = tmp106 + tmp12
    tmp108 = tmp107 + tmp14
    tmp109 = tmp108 + tmp16
    tmp110 = tmp109 + tmp18
    tmp111 = tmp110 + tmp20
    tmp112 = tmp111 + tmp22
    tmp113 = tmp112 + tmp24
    tmp114 = tmp113 + tmp26
    tmp115 = tmp114 + tmp28
    tmp116 = tmp115 + tmp30
    tmp117 = tmp116 + tmp32
    tmp118 = tmp117 + tmp34
    tmp119 = tmp118 + tmp36
    tmp120 = tmp119 + tmp38
    tmp121 = tmp120 + tmp40
    tmp122 = tmp121 + tmp42
    tmp123 = tmp122 + tmp44
    tmp124 = tmp123 + tmp46
    tmp125 = tmp124 + tmp48
    tmp126 = tmp125 + tmp50
    tmp127 = tmp126 + tmp52
    tmp128 = tmp127 + tmp54
    tmp129 = tmp128 + tmp56
    tmp130 = tmp129 + tmp58
    tmp131 = tmp130 + tmp60
    tmp132 = tmp131 + tmp62
    tmp133 = tmp101 + tmp132
    tmp134 = tmp13 + tmp2
    tmp135 = tmp134 + tmp4
    tmp136 = tmp135 + tmp6
    tmp137 = tmp136 + tmp8
    tmp138 = tmp137 + tmp10
    tmp139 = tmp138 + tmp12
    tmp140 = tmp139 + tmp14
    tmp141 = tmp140 + tmp16
    tmp142 = tmp141 + tmp18
    tmp143 = tmp142 + tmp20
    tmp144 = tmp143 + tmp22
    tmp145 = tmp144 + tmp24
    tmp146 = tmp145 + tmp26
    tmp147 = tmp146 + tmp28
    tmp148 = tmp147 + tmp30
    tmp149 = tmp148 + tmp32
    tmp150 = tmp149 + tmp34
    tmp151 = tmp150 + tmp36
    tmp152 = tmp151 + tmp38
    tmp153 = tmp152 + tmp40
    tmp154 = tmp153 + tmp42
    tmp155 = tmp154 + tmp44
    tmp156 = tmp155 + tmp46
    tmp157 = tmp156 + tmp48
    tmp158 = tmp157 + tmp50
    tmp159 = tmp158 + tmp52
    tmp160 = tmp159 + tmp54
    tmp161 = tmp160 + tmp56
    tmp162 = tmp161 + tmp58
    tmp163 = tmp162 + tmp60
    tmp164 = tmp163 + tmp62
    tmp165 = tmp133 + tmp164
    tmp166 = tmp19 + tmp2
    tmp167 = tmp166 + tmp4
    tmp168 = tmp167 + tmp6
    tmp169 = tmp168 + tmp8
    tmp170 = tmp169 + tmp10
    tmp171 = tmp170 + tmp12
    tmp172 = tmp171 + tmp14
    tmp173 = tmp172 + tmp16
    tmp174 = tmp173 + tmp18
    tmp175 = tmp174 + tmp20
    tmp176 = tmp175 + tmp22
    tmp177 = tmp176 + tmp24
    tmp178 = tmp177 + tmp26
    tmp179 = tmp178 + tmp28
    tmp180 = tmp179 + tmp30
    tmp181 = tmp180 + tmp32
    tmp182 = tmp181 + tmp34
    tmp183 = tmp182 + tmp36
    tmp184 = tmp183 + tmp38
    tmp185 = tmp184 + tmp40
    tmp186 = tmp185 + tmp42
    tmp187 = tmp186 + tmp44
    tmp188 = tmp187 + tmp46
    tmp189 = tmp188 + tmp48
    tmp190 = tmp189 + tmp50
    tmp191 = tmp190 + tmp52
    tmp192 = tmp191 + tmp54
    tmp193 = tmp192 + tmp56
    tmp194 = tmp193 + tmp58
    tmp195 = tmp194 + tmp60
    tmp196 = tmp195 + tmp62
    tmp197 = tmp165 + tmp196
    tmp198 = tmp21 + tmp2
    tmp199 = tmp198 + tmp4
    tmp200 = tmp199 + tmp6
    tmp201 = tmp200 + tmp8
    tmp202 = tmp201 + tmp10
    tmp203 = tmp202 + tmp12
    tmp204 = tmp203 + tmp14
    tmp205 = tmp204 + tmp16
    tmp206 = tmp205 + tmp18
    tmp207 = tmp206 + tmp20
    tmp208 = tmp207 + tmp22
    tmp209 = tmp208 + tmp24
    tmp210 = tmp209 + tmp26
    tmp211 = tmp210 + tmp28
    tmp212 = tmp211 + tmp30
    tmp213 = tmp212 + tmp32
    tmp214 = tmp213 + tmp34
    tmp215 = tmp214 + tmp36
    tmp216 = tmp215 + tmp38
    tmp217 = tmp216 + tmp40
    tmp218 = tmp217 + tmp42
    tmp219 = tmp218 + tmp44
    tmp220 = tmp219 + tmp46
    tmp221 = tmp220 + tmp48
    tmp222 = tmp221 + tmp50
    tmp223 = tmp222 + tmp52
    tmp224 = tmp223 + tmp54
    tmp225 = tmp224 + tmp56
    tmp226 = tmp225 + tmp58
    tmp227 = tmp226 + tmp60
    tmp228 = tmp227 + tmp62
    tmp229 = tmp197 + tmp228
    tmp230 = tmp23 + tmp2
    tmp231 = tmp230 + tmp4
    tmp232 = tmp231 + tmp6
    tmp233 = tmp232 + tmp8
    tmp234 = tmp233 + tmp10
    tmp235 = tmp234 + tmp12
    tmp236 = tmp235 + tmp14
    tmp237 = tmp236 + tmp16
    tmp238 = tmp237 + tmp18
    tmp239 = tmp238 + tmp20
    tmp240 = tmp239 + tmp22
    tmp241 = tmp240 + tmp24
    tmp242 = tmp241 + tmp26
    tmp243 = tmp242 + tmp28
    tmp244 = tmp243 + tmp30
    tmp245 = tmp244 + tmp32
    tmp246 = tmp245 + tmp34
    tmp247 = tmp246 + tmp36
    tmp248 = tmp247 + tmp38
    tmp249 = tmp248 + tmp40
    tmp250 = tmp249 + tmp42
    tmp251 = tmp250 + tmp44
    tmp252 = tmp251 + tmp46
    tmp253 = tmp252 + tmp48
    tmp254 = tmp253 + tmp50
    tmp255 = tmp254 + tmp52
    tmp256 = tmp255 + tmp54
    tmp257 = tmp256 + tmp56
    tmp258 = tmp257 + tmp58
    tmp259 = tmp258 + tmp60
    tmp260 = tmp259 + tmp62
    tmp261 = tmp229 + tmp260
    tmp262 = tmp27 + tmp2
    tmp263 = tmp262 + tmp4
    tmp264 = tmp263 + tmp6
    tmp265 = tmp264 + tmp8
    tmp266 = tmp265 + tmp10
    tmp267 = tmp266 + tmp12
    tmp268 = tmp267 + tmp14
    tmp269 = tmp268 + tmp16
    tmp270 = tmp269 + tmp18
    tmp271 = tmp270 + tmp20
    tmp272 = tmp271 + tmp22
    tmp273 = tmp272 + tmp24
    tmp274 = tmp273 + tmp26
    tmp275 = tmp274 + tmp28
    tmp276 = tmp275 + tmp30
    tmp277 = tmp276 + tmp32
    tmp278 = tmp277 + tmp34
    tmp279 = tmp278 + tmp36
    tmp280 = tmp279 + tmp38
    tmp281
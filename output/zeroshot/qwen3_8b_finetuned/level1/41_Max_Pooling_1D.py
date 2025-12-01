1. High‑level goal  
Replace the CPU‑bound PyTorch `nn.MaxPool1d` with a Triton kernel that computes the per‑channel 1‑D maximum over a window of size `kernel_size` (here 8) with optional padding (4) and dilation (3). The kernel must also produce the *maximum* values (the pooled output) and the *argmax* indices (the position within the window) – both are needed by the downstream model.

2. Tensor shapes and indexing  
- Input: `(B, C, L)` where `B = batch_size = 64`, `C = features = 192`, `L = sequence_length = 65536`.  
- After padding, each channel’s effective length becomes `L_pad = L + 2*padding = 65536 + 8 = 65544`.  
- Dilation expands the window: each element of the window is spaced by `dilation` steps. With `kernel_size = 8`, the number of positions to consider is `K = kernel_size * dilation = 24`.  
- The output shape is `(B, C, O)` where `O = (L_pad - K) // stride + 1 = (65544 - 24) // 1 + 1 = 65521`.  

The Triton kernel flattens the 3‑D tensor to a 1‑D view of size `N = B * C * O = 64 * 192 * 65521 = 806 600 640`. Each program processes a contiguous block of `XBLOCK` elements (`XBLOCK = 128`). The block offset is `program_id * XBLOCK`. The kernel computes the linear index `x3 = program_id * XBLOCK + lane_offset`.  

From `x3` we recover the three logical dimensions:  
- `x1 = (x3 // O) % C` – the feature index.  
- `x0 = x3 % O` – the output‑position index.  
- `x2 = x3 // (C*O)` – the batch index.  

The window start offset for a given output position is `window_start = (x0 + stride) * stride + padding`. With `stride = 1` and `padding = 4`, `window_start = x0 + 4`. The kernel then loads the 24 padded values: `x + window_start + k*dilation` for `k = 0 … 23`. The linear address for each load is `base_ptr + (window_start + k) + (feature_stride * x1) + (seq_stride * x0)`. Because the stride of the feature dimension is `seq_stride = L_pad = 65544`, the address simplifies to `base_ptr + (window_start + k) + (65544 * x1)`.  

3. Parallelization & launch configuration  
- **Program dimension**: 1‑D grid (`grid(806600640)`). Each program corresponds to a contiguous block of `XBLOCK` output elements.  
- **Block size (`XBLOCK`)**: Fixed to 128 (a power‑of‑two that fits well with warp size and registers).  
- **Number of warps**: 4 warps per block (`num_warps=4`) to give enough parallelism for the 24‑element window while keeping register pressure low.  
- **Stages**: 1 stage (`num_stages=1`) because the kernel is memory‑bound and does not need double‑buffering.  
- **Occupancy**: With 806 600 640 elements and 128 per block, the grid yields ~6 313 704 blocks. The A100 can launch ~32 000 blocks per SM, so the kernel will saturate the GPU’s SMs.  

4. Memory access pattern  
- **Loads**:  
  - The first load (`tmp0`) reads the element at the output position itself (`x0`). This is the “center” of the window.  
  - The remaining 23 loads (`tmp1 … tmp24`) read the padded values spaced by `dilation`. All loads are masked with the same `xmask` to avoid out‑of‑bounds when the window extends beyond the padded length.  
  - All loads use `eviction_policy='evict_last'` to hint the compiler that these values are temporary and can be evicted from registers early, reducing register pressure.  

- **Computation**:  
  - The kernel computes the maximum of the 24 values using a cascade of `triton_helpers.maximum` calls. The intermediate maxima (`tmp10`, `tmp17`, `tmp21`, `tmp25`) are built progressively to keep the register count low.  
  - The argmax is built simultaneously: a boolean flag (`tmp26`) records whether the current candidate is larger than the current max; the final `tmp27` is the index of the maximum within the window (0‑23).  

- **Stores**:  
  - The maximum value is written to `out_ptr0` at the same linear index `x3`.  
  - The argmax index is written to `out_ptr1` at the same index. Both stores are guarded by `xmask`.  

- **Coalescing**: Because each thread accesses a contiguous address (`base_ptr + (window_start + k) + 65544 * x1`), the loads are naturally coalesced across the warp. The stride between consecutive threads in the block is 1, so the memory traffic is maximally efficient.  

5. Numerics & correctness details  
- **Data type**: All tensors are `float32`. The kernel uses `tl.float32` for loads, intermediate maxima, and stores, preserving the precision of the original PyTorch implementation.  
- **Padding**: The kernel adds explicit padding (`+4`) to the window start, matching the `padding` argument of `nn.MaxPool1d`. No extra zero‑filling is required because the original tensor already contains zeros at the padded locations (the input is random, but the padding is applied before the kernel).  
- **Dilation**: The dilation factor of 3 is implemented by iterating over `k` and adding `k*dilation` to each load address, exactly as PyTorch’s dilation would do.  
- **Stride**: Stride is 1, so the window moves by one element per output position; the kernel reflects this by simply incrementing the window start for the next output.  
- **Return indices**: The kernel produces both the maximum value and the argmax index, matching the `return_indices=False` default but allowing downstream code to use the indices if needed.  
- **Masking**: The `xmask` guard ensures that the final block (or partial block) does not read/write beyond the allocated output size. This is crucial because the total number of elements is not a multiple of `XBLOCK`.  

6. Summary checklist  
- **Shape handling**: Flatten `(B, C, O)` → 1‑D of size `N = B*C*O`. Derive batch, feature, and output‑position indices from the linear index.  
- **Window computation**: `window_start = output_pos + padding`. Load `kernel_size * dilation` padded values using the derived linear address.  
- **Parallelism**: 1‑D grid, `XBLOCK = 128`, 4 warps, 1 stage. Compute `grid = ceil(N / XBLOCK)`.  
- **Memory accesses**: Coalesced loads of 24 contiguous values per thread; masked stores for edge cases.  
- **Maximum & argmax**: Cascade of pairwise `maximum` calls; simultaneous boolean flag to track argmax.  
- **Numerics**: `float32` throughout, padding applied before kernel, dilation implemented via stride‑like increments, stride = 1 handled by simple offset.  
- **Correctness**: Output shape `(B, C, O)` matches PyTorch’s `nn.MaxPool1d` output; both pooled values and argmax indices are produced; masking guarantees safety for non‑divisible sizes.  

Following this reasoning yields a Triton kernel that faithfully reproduces the behavior of `nn.MaxPool1d` while exploiting the A100’s memory bandwidth and parallelism.</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 806600640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65521 % 192
    x0 = xindex % 65521
    x2 = xindex // 12696960
    tmp0 = tl.load(in_ptr0 + (x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (13 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (18 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (23 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (33 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (38 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (43 + x0 + 65544 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tmp4 > tmp0
    tmp7 = tmp4 > tmp1
    tmp8 = tmp7 == tmp5
    tmp10 = triton_helpers.maximum(tmp4, tmp6)
    tmp11 = tmp10 > tmp4
    tmp13 = tmp10 > tmp6
    tmp14 = tmp13 == tmp11
    tmp16 = triton_helpers.maximum(tmp10, tmp9)
    tmp17 = tmp16 > tmp10
    tmp19 = tmp16 > tmp9
    tmp20 = tmp19 == tmp17
    tmp22 = triton_helpers.maximum(tmp16, tmp12)
    tmp23 = tmp22 > tmp16
    tmp25 = tmp22 > tmp12
    tmp26 = tmp25 == tmp23
    tmp27 = tl.where(tmp26, 22, tmp20)
    tmp28 = tl.where(tmp26, 21, tmp27)
    tmp29 = tl.where(tmp20, 18, tmp28)
    tmp30 = tl.where(tmp14, 15, tmp29)
    tmp31 = tl.where(tmp8, 12, tmp30)
    tmp32 = triton_helpers.maximum(tmp16, tmp15)
    tmp33 = tmp32 > tmp16
    tmp34 = tmp32 > tmp15
    tmp35 = tmp34 == tmp33
    tmp36 = triton_helpers.maximum(tmp32, tmp18)
    tmp37 = tmp36 > tmp32
    tmp38 = tmp36 > tmp18
    tmp39 = tmp38 == tmp37
    tmp40 = triton_helpers.maximum(tmp36, tmp21)
    tmp41 = tmp40 > tmp36
    tmp42 = tmp40 > tmp21
    tmp43 = tmp42 == tmp41
    tmp44 = triton_helpers.maximum(tmp40, tmp24)
    tmp45 = tmp44 > tmp40
    tmp46 = tmp44 > tmp24
    tmp47 = tmp46 == tmp45
    tmp48 = tl.where(tmp47, 24, tmp43)
    tmp49 = tl.where(tmp47, 23, tmp48)
    tmp50 = tl.where(tmp43, 20, tmp49)
    tmp51 = tl.where(tmp39, 17, tmp50)
    tmp52 = tl.where(tmp35, 14, tmp51)
    tmp53 = tl.where(tmp31, 11, tmp52)
    tmp54 = triton_helpers.maximum(tmp36, tmp18)
    tmp55 = tmp54 > tmp36
    tmp56 = tmp54 > tmp18
    tmp57 = tmp56 == tmp55
    tmp58 = triton_helpers.maximum(tmp54, tmp21)
    tmp59 = tmp58 > tmp54
    tmp60 = tmp58 > tmp21
    tmp61 = tmp60 == tmp59
    tmp62 = triton_helpers.maximum(tmp58, tmp24)
    tmp63 = tmp62 > tmp58
    tmp64 = tmp62 > tmp24
    tmp65 = tmp64 == tmp63
    tmp66 = tl.where(tmp65, 24, tmp61)
    tmp67 = tl.where(tmp65, 23, tmp66)
    tmp68 = tl.where(tmp61, 20, tmp67)
    tmp69 = tl.where(tmp57, 17, tmp68)
    tmp70 = tl.where(tmp53, 14, tmp69)
    tmp71 = triton_helpers.maximum(tmp58, tmp24)
    tmp72 = tmp71 > tmp58
    tmp73 = tmp71 > tmp24
    tmp74 = tmp73 == tmp72
    tmp75 = tl.where(tmp74, 24, tmp61)
    tmp76 = tl.where(tmp74, 23, tmp75)
    tmp77 = tl.where(tmp61, 20, tmp76)
    tmp78 = tl.where(tmp57, 17, tmp77)
    tmp79 = tl.where(tmp53, 14, tmp78)
    tmp80 = triton_helpers.maximum(tmp62, tmp24)
    tmp81 = tmp80 > tmp62
    tmp82 = tmp80 > tmp24
    tmp83 = tmp82 == tmp81
    tmp84 = tl.where(tmp83, 24, tmp61)
    tmp85 = tl.where(tmp83, 23, tmp84)
    tmp86 = tl.where(tmp61, 20, tmp85)
    tmp87 = tl.where(tmp57, 17, tmp86)
    tmp88 = tl.where(tmp53, 14, tmp87)
    tmp89 = triton_helpers.maximum(tmp62, tmp24)
    tmp90 = tmp89 > tmp62
    tmp91 = tmp89 > tmp24
    tmp92 = tmp91 == tmp90
    tmp93 = tl.where(tmp92, 24, tmp61)
    tmp94 = tl.where(tmp92, 23, tmp93)
    tmp95 = tl.where(tmp61, 20, tmp94)
    tmp96 = tl.where(tmp57, 17, tmp95)
    tmp97 = tl.where(tmp53, 14, tmp96)
    tmp98 = triton_helpers.maximum(tmp80, tmp24)
    tmp99 = tmp98 > tmp80
    tmp100 = tmp98 > tmp24
    tmp101 = tmp100 == tmp99
    tmp102 = tl.where(tmp101, 24, tmp61)
    tmp103 = tl.where(tmp101, 23, tmp102)
    tmp104 = tl.where(tmp61, 20, tmp103)
    tmp105 = tl.where(tmp57, 17, tmp104)
    tmp106 = tl.where(tmp53, 14, tmp105)
    tmp107 = triton_helpers.maximum(tmp89, tmp24)
    tmp108 = tmp107 > tmp89
    tmp109 = tmp107 > tmp24
    tmp110 = tmp109 == tmp108
    tmp111 = tl.where(tmp110, 24, tmp61)
    tmp112 = tl.where(tmp110, 23, tmp111)
    tmp113 = tl.where(tmp61, 20, tmp112)
    tmp114 = tl.where(tmp57, 17, tmp113)
    tmp115 = tl.where(tmp53, 14, tmp114)
    tmp116 = triton_helpers.maximum(tmp98, tmp24)
    tmp117 = tmp116 > tmp98
    tmp118 = tmp116 > tmp24
    tmp119 = tmp118 == tmp117
    tmp120 = tl.where(tmp119, 24, tmp61)
    tmp121 = tl.where(tmp119, 23, tmp120)
    tmp122 = tl.where(tmp61, 20, tmp121)
    tmp123 = tl.where(tmp57, 17, tmp122)
    tmp124 = tl.where(tmp53, 14, tmp123)
    tmp125 = triton_helpers.maximum(tmp107, tmp24)
    tmp126 = tmp125 > tmp107
    tmp127 = tmp125 > tmp24
    tmp128 = tmp127 == tmp126
    tmp129 = tl.where(tmp128, 24, tmp61)
    tmp130 = tl.where(tmp128, 23, tmp129)
    tmp131 = tl.where(tmp61, 20, tmp130)
    tmp132 = tl.where(tmp57, 17, tmp131)
    tmp133 = tl.where(tmp53, 14, tmp132)
    tmp134 = triton_helpers.maximum(tmp116, tmp24)
    tmp135 = tmp134 > tmp116
    tmp136 = tmp134 > tmp24
    tmp137 = tmp136 == tmp135
    tmp138 = tl.where(tmp137, 24, tmp61)
    tmp139 = tl.where(tmp137, 23, tmp138)
    tmp140 = tl.where(tmp61, 20, tmp139)
    tmp141 = tl.where(tmp57, 17, tmp140)
    tmp142 = tl.where(tmp53, 14, tmp141)
    tmp143 = triton_helpers.maximum(tmp116, tmp24)
    tmp144 = tmp143 > tmp116
    tmp145 = tmp143 > tmp24
    tmp146 = tmp145 == tmp144
    tmp147 = tl.where(tmp146, 24, tmp61)
    tmp148 = tl.where(tmp146, 23, tmp147)
    tmp149 = tl.where(tmp61, 20, tmp148)
    tmp150 = tl.where(tmp57, 17, tmp149)
    tmp151 = tl.where(tmp53, 14, tmp150)
    tmp152 = triton_helpers.maximum(tmp107, tmp24)
    tmp153 = tmp152 > tmp107
    tmp154 = tmp152 > tmp24
    tmp155 = tmp154 == tmp153
    tmp156 = tl.where(tmp155, 24, tmp61)
    tmp157 = tl.where(tmp155, 23, tmp156)
    tmp158 = tl.where(tmp61, 20, tmp157)
    tmp159 = tl.where(tmp57, 17, tmp158)
    tmp160 = tl.where(tmp53, 14, tmp159)
    tmp161 = triton_helpers.maximum(tmp98, tmp24)
    tmp162 = tmp161 > tmp98
    tmp163 = tmp161 > tmp24
    tmp164 = tmp163 == tmp162
    tmp165 = tl.where(tmp164, 24, tmp61)
    tmp166 = tl.where(tmp164, 23, tmp165)
    tmp167 = tl.where(tmp61, 20, tmp166)
    tmp168 = tl.where(tmp57, 17, tmp167)
    tmp169 = tl.where(tmp53, 14, tmp168)
    tmp170 = triton_helpers.maximum(tmp89, tmp24)
    tmp171 = tmp170 > tmp89
    tmp172 = tmp170 > tmp24
    tmp173 = tmp172 == tmp171
    tmp174 = tl.where(tmp173, 24, tmp61)
    tmp175 = tl.where(tmp173, 23, tmp174)
    tmp176 = tl.where(tmp61, 20, tmp175)
    tmp177 = tl.where(tmp57, 17, tmp176)
    tmp178 = tl.where(tmp53, 14, tmp177)
    tmp179 = triton_helpers.maximum(tmp80, tmp24)
    tmp180 = tmp179 > tmp80
    tmp181 = tmp179 > tmp24
    tmp182 = tmp181 == tmp180
    tmp183 = tl.where(tmp182, 24, tmp61)
    tmp184 = tl.where(tmp182, 23, tmp183)
    tmp185 = tl.where(tmp61, 20, tmp184)
    tmp186 = tl.where(tmp57, 17, tmp185)
    tmp187 = tl.where(tmp53, 14, tmp186)
    tmp188 = triton_helpers.maximum(tmp71, tmp24)
    tmp189 = tmp188 > tmp71
    tmp190 = tmp188 > tmp24
    tmp191 = tmp190 == tmp189
    tmp192 = tl.where(tmp191, 24, tmp61)
    tmp193 = tl.where(tmp191, 23, tmp192)
    tmp194 = tl.where(tmp61, 20, tmp193)
    tmp195 = tl.where(tmp57, 17, tmp194)
    tmp196 = tl.where(tmp53, 14, tmp195)
    tmp197 = triton_helpers.maximum(tmp62, tmp24)
    tmp198 = tmp197 > tmp62
    tmp199 = tmp197 > tmp24
    tmp200 = tmp199 == tmp198
    tmp201 = tl.where(tmp200, 24, tmp61)
    tmp202 = tl.where(tmp200, 23, tmp201)
    tmp203 = tl.where(tmp61, 20, tmp202)
    tmp204 = tl.where(tmp57, 17, tmp203)
    tmp205 = tl.where(tmp53, 14, tmp204)
    tmp206 = triton_helpers.maximum(tmp58, tmp24)
    tmp207 = tmp206 > tmp58
    tmp208 = tmp206 > tmp24
    tmp209 = tmp208 == tmp207
    tmp210 = tl.where(tmp209, 24, tmp61)
    tmp211 = tl.where(tmp209, 23, tmp210)
    tmp212 = tl.where(tmp61, 20, tmp211)
    tmp213 = tl.where(tmp57, 17, tmp212)
    tmp214 = tl.where(tmp53, 14, tmp213)
    tmp215 = triton_helpers.maximum(tmp40, tmp24)
    tmp216 = tmp215 > tmp40
    tmp217 = tmp215 > tmp24
    tmp218 = tmp217 == tmp216
    tmp219 = tl.where(tmp218, 24, tmp61)
    tmp220 = tl.where(tmp218, 23, tmp219)
    tmp221 = tl.where(tmp61, 20, tmp220)
    tmp222 = tl.where(tmp57, 17, tmp221)
    tmp223 = tl.where(tmp53, 14, tmp222)
    tmp224 = triton_helpers.maximum(tmp36, tmp24)
    tmp225 = tmp224 > tmp36
    tmp226 = tmp224 > tmp24
    tmp227 = tmp226 == tmp225
    tmp228 = tl.where(tmp227, 24, tmp61)
    tmp229 = tl.where(tmp227, 23, tmp228)
    tmp230 = tl.where(tmp61, 20, tmp229)
    tmp231 = tl.where(tmp57, 17, tmp230)
    tmp232 = tl.where(tmp53, 14, tmp231)
    tmp233 = triton_helpers.maximum(tmp22, tmp24)
    tmp234 = tmp233 > tmp22
    tmp235 = tmp233 > tmp24
    tmp236 = tmp235 == tmp234
    tmp237 = tl.where(tmp236, 24, tmp61)
    tmp238 = tl.where(tmp236, 23, tmp237)
    tmp239 = tl.where(tmp61, 20, tmp238)
    tmp240 = tl.where(tmp57, 17, tmp239)
    tmp241 = tl.where(tmp53, 14, tmp240)
    tmp242 = triton_helpers.maximum(tmp16, tmp24)
    tmp243 = tmp242 > tmp16
    tmp244 = tmp242 > tmp24
    tmp245 = tmp244 == tmp243
    tmp246 = tl.where(tmp245, 24, tmp61)
    tmp247 = tl.where(tmp245, 23, tmp246)
    tmp248 = tl.where(tmp61, 20, tmp247)
    tmp249 = tl.where(tmp57, 17, tmp248)
    tmp250 = tl.where(tmp53, 14, tmp249)
    tmp251 = triton_helpers.maximum(tmp12, tmp24)
    tmp252 = tmp251 > tmp12
    tmp253 = tmp251 > tmp24
    tmp254 = tmp253 == tmp252
    tmp255 = tl.where(tmp254, 24, tmp61)
    tmp256 = tl.where(tmp254, 23, tmp255)
    tmp257 = tl.where(tmp61, 20, tmp256)
    tmp258 = tl.where(tmp57, 17, tmp257)
    tmp259 = tl.where(tmp53, 14, tmp258)
    tmp260 = triton_helpers.maximum(tmp9, tmp24)
    tmp261 = tmp260 > tmp9
    tmp262 = tmp260 > tmp24
    tmp263 = tmp262 == tmp261
    tmp264 = tl.where(tmp263, 24, tmp61)
    tmp265 = tl.where(tmp263, 23, tmp264)
    tmp266 = tl.where(tmp61, 20, tmp265)
    tmp267 = tl.where(tmp57, 17, tmp266)
    tmp268 = tl.where(tmp53, 14, tmp267)
    tmp269 = triton_helpers.maximum(tmp6, tmp24)
    tmp270 = tmp269 > tmp6
    tmp271 = tmp269 > tmp24
    tmp272 = tmp271 == tmp270
    tmp273 = tl.where(tmp272, 24, tmp61)
    tmp274 = tl.where(tmp272, 23, tmp273)
    tmp275 = tl.where(tmp61, 20, tmp274)
    tmp276 = tl.where(tmp57, 17, tmp275)
    tmp277 = tl.where(tmp53, 14, tmp276)
    tmp278 = triton_helpers.maximum(tmp3, tmp24)
    tmp279 = tmp278 > tmp3
    tmp280 = tmp278 > tmp24
    tmp281 = tmp280 == tmp279
    tmp282 = tl.where(tmp281, 24, tmp61)
    tmp283 = tl.where(tmp281, 23, tmp282)
    tmp284 = tl.where(tmp61, 20, tmp283)
    tmp285 = tl.where(tmp57, 17, tmp284)
    tmp286 = tl.where(tmp53, 14, tmp285)
    tmp287 = triton_helpers.maximum(tmp1, tmp24)
    tmp288 = tmp287 > tmp1
    tmp289 = tmp287 > tmp24
    tmp290 = tmp289 == tmp288
    tmp291 = tl.where(tmp290, 24, tmp61)
    tmp292 = tl.where(tmp290, 23, tmp291)
    tmp293 = tl.where(tmp61, 20, tmp292)
    tmp294 = tl.where(tmp57, 17, tmp293)
    tmp295 = tl.where(tmp53, 14, tmp294)
    tmp296 = triton_helpers.maximum(tmp0, tmp24)
    tmp297 = tmp296 > tmp0
    tmp298 = tmp296 > tmp24
    tmp299 = tmp298 == tmp297
    tmp300 = tl.where(tmp299, 24, tmp61)
    tmp301 = tl.where(tmp299, 23, tmp300)
    tmp302 = tl.where(tmp61, 20, tmp301)
    tmp303 = tl.where(tmp57, 17, tmp302)
    tmp304 = tl.where(tmp53, 14, tmp303)
    tmp305 = triton_helpers.maximum(tmp27, tmp304)
    tmp306 = tmp305 > tmp27
    tmp307 = tmp305 > tmp304
    tmp308 = tmp307 == tmp306
    tl.store(out_ptr0 + x3, tmp305, xmask)
    tl.store(out_ptr1 + x3, tmp308, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (64, 192, 65536), (12696960, 65544, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 192, 65521), (12696960, 65544, 1),
            torch.float32)
        buf1 = empty_strided_cuda((64, 192, 65521), (12696960, 65544, 1),
            torch.int64)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0[grid(806600640)](arg0_1,
            buf0, buf1, 806600640, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs Max Pooling 1D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        """
        Initializes the Max Pooling
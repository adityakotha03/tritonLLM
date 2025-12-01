1. **Goal** – Replace the elementwise addition used in the original `ModelNew` (the call to `triton_add`) with a fused Triton kernel that also performs the subsequent `clamp` and `max_pool2d` operations, while preserving the exact mathematical semantics of the original PyTorch forward pass.

2. **High‑level decomposition**  
   - **Stage 0 (convolution + bias)** – Delegated to the existing `extern_kernels.convolution` call; the kernel receives the raw output of the convolution and the bias tensor as arguments.  
   - **Stage 1 (group‑norm)** – Implemented by `triton_poi_fused_native_group_norm_0`. This kernel computes the mean and variance per group, normalises the input, and writes both the normalised values and the per‑group mean/variance back to temporary buffers.  
   - **Stage 2 (scale multiplication + clamp)** – Implemented by `triton_poi_fused_add_clamp_mul_1`. This kernel reads the normalised values, the learnable scale tensor, multiplies them element‑wise, adds the bias term (the per‑group mean from Stage 1), clamps the result to `[clamp_min, clamp_max]`, and writes the final scaled‑and‑clamped tensor to a temporary buffer.  
   - **Stage 3 (max‑pool2d)** – Implemented by `triton_poi_fused_max_pool2d_with_indices_2`. This kernel computes the 4×4 max‑pool over the spatial dimension, also producing the arg‑max indices that the original PyTorch `max_pool2d` would return. The kernel reads the scaled‑and‑clamped tensor, reduces across the 4×4 window, and writes both the pooled values and the indices to the final output buffers.  

3. **Data‑layout reasoning**  
   - All tensors are contiguous in the memory‑order expected by the Triton kernels: `(batch, out_channels, height, width)` with stride `(out_channels*height*width, width*height, width, 1)`.  
   - The group‑norm kernel works on a flattened view of size `N = batch * out_channels * height * width` (here `N = 128 × 64 × 128 × 128 = 1 31 072`). It processes the tensor in chunks of `XBLOCK = 512` elements per program, i.e. 512 contiguous memory locations per block.  
   - The scale‑multiply‑clamp kernel processes the same flattened size (`N = 1 31 072`) with `XBLOCK = 1024`, giving two programs per 1024‑element block. This larger block size reduces launch overhead while still fitting within the 164 KB shared‑memory limit.  
   - The max‑pool kernel processes a flattened size of `N = batch * out_channels * height' * width' = 128 × 64 × 32 × 32 = 819 200`. It uses a 4‑element block (`XBLOCK = 16`) because the reduction window is 4×4, and the kernel must compute the max across the four output channels of each 4×4 window.  

4. **Memory access pattern**  
   - **Loads** – All kernels use `tl.load` with a mask that guards against out‑of‑bounds accesses. The mask is derived from `xmask = xindex < xnumel`. Offsets are computed as `xoffset + tl.arange(0, XBLOCK)`.  
   - **Coalescing** – Within each block, threads access consecutive memory addresses because `xoffset` increments by 1 for each thread. This yields fully coalesced global memory reads/writes, matching the hardware’s optimal memory bandwidth (1935 GB/s).  
   - **Shared memory** – The group‑norm kernel does not explicitly allocate shared memory; it relies on the compiler’s implicit tiling and register allocation. The max‑pool kernel uses a small reduction across four elements, which fits entirely in registers (no shared memory needed).  
   - **Stores** – Results are written back with the same mask, ensuring that only valid elements are stored. The `triton_helpers.promote_to_tensor` call guarantees that the stored tensors retain the original tensor type and layout.  

5. **Numerics & correctness details**  
   - **Group‑norm** – The kernel computes the per‑group mean (`tmp13`) and variance (`tmp22`) using `tl.sum` over the group dimension. The divisor `16.0` corresponds to the number of channels per group (`num_groups`). The epsilon (`1e‑5`) is added before the reciprocal‑sqrt to avoid division‑by‑zero.  
   - **Scale multiplication** – The learnable scale tensor is broadcast over the spatial dimensions; the kernel loads it once per group (`x1 = xindex // 512 % 64`) and multiplies it with the normalised value (`tmp4`).  
   - **Bias addition** – The bias term is the per‑group mean from Stage 1 (`tmp13`). The kernel adds it after scaling, reproducing the `x = x * scale + bias` semantics.  
   - **Clamping** – The clamp is performed with `triton_helpers.maximum` and `triton_helpers.minimum` to avoid overflow/underflow and to match PyTorch’s `torch.clamp`. The clamp thresholds are constants (`0.0` and `1.0`).  
   - **Max‑pool reduction** – The kernel computes the max across the 4×4 window by iterating over the four channel indices (`tmp6`, `tmp10`, `tmp14`, `tmp18`) and using pairwise `maximum` calls. The arg‑max indices are stored as `int64` values ranging `0…15`.  

6. **Launch configuration**  
   - **Grid size** – Determined by the lambda `grid = lambda meta: ((xnumel + XBLOCK - 1) // XBLOCK,)`. For group‑norm (`xnumel = 1 31 072`, `XBLOCK = 512`) → `grid = (256,)`. For scale‑clamp (`xnumel = 1 31 072`, `XBLOCK = 1024`) → `grid = (128,)`. For max‑pool (`xnumel = 819 200`, `XBLOCK = 16`) → `grid = (51 200,)`.  
   - **Block dimensions** – All kernels use a single‑dimensional grid (`program_id(0)`). Each block processes `XBLOCK` elements; the total number of blocks equals `ceil(xnumel / XBLOCK)`.  
   - **Warps & stages** – The launch parameters (`num_warps` and `num_stages`) are chosen by the autotuner: 4 warps for group‑norm (enough to cover 512‑element blocks), 4 warps for scale‑clamp (covers 1024‑element blocks), and 4 warps for max‑pool (covers 16‑element blocks). Stages are set to 1 for all kernels because the reduction is simple and fits in registers.  

7. **Summary checklist**  
   - **Convolution** – delegated to `extern_kernels.convolution`; output shape `(128,64,128,128)`.  
   - **Group‑norm** – fused kernel `triton_poi_fused_native_group_norm_0` computes mean/variance, normalises, writes both results.  
   - **Scale + bias + clamp** – fused kernel `triton_poi_fused_add_clamp_mul_1` multiplies by scale, adds bias, clamps.  
   - **Max‑pool** – fused kernel `triton_poi_fused_max_pool2d_with_indices_2` reduces 4×4 window, writes pooled values and arg‑max indices.  
   - **Memory layout** – all tensors are contiguous with stride `(out_channels*height*width, width*height, width, 1)`.  
   - **Coalesced accesses** – each block processes a contiguous chunk; masks guard boundaries.  
   - **Numerics** – epsilon, divisor, clamp thresholds, and reduction order exactly match PyTorch semantics.  
   - **Launch config** – grid = `ceil(numel / XBLOCK)`, XBLOCK = 512, 1024, or 16, warps = 4, stages = 1.  

Following this reasoning, the Triton‑fused kernels together reproduce the original `Model` forward pass while eliminating intermediate Python‑level elementwise ops and reducing memory traffic.</think>

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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512 % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 512 * x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 512 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (512 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (513 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (514 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (515 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (1024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (1025 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (1026 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (1027 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (1536 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (1537 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (1538 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (1539 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (2048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (2049 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (2050 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (2051 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (2560 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (2561 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (2562 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (2563 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (3072 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (3073 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (3074 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (3075 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (3584 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (3585 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (3586 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (3587 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (4096 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (4097 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (4098 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (4099 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (4608 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (4609 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (4610 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (4611 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (5120 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (5121 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (5122 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (5123 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (5632 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (5633 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (5634 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (5635 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (6144 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (6145 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (6146 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (6147 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (6656 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (6657 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (6658 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (6659 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (7168 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (7169 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (7170 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (7171 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (7680 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (7681 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (7682 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (7683 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (8192 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (8193 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (8194 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (8195 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (8704 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (8705 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (8706 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (8707 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (9216 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (9217 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (9218 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (9219 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (9728 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (9729 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (9730 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (9731 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (10240 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (10241 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (10242 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (10243 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (10752 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (10753 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (10754 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (10755 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (11264 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (11265 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (11266 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (11267 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (11776 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (11777 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (11778 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (11779 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (12288 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (12289 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (12290 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (12291 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (12800 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (12801 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (12802 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (12803 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (13312 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (13313 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (13314 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (13315 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (13824 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (13825 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (13826 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (13827 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (14336 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (14337 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (14338 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (14339 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (14848 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (14849 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (14850 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (14851 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (15360 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (15361 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (15362 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (15363 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (15872 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (15873 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (15874 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (15875 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (16384 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (16385 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (16386 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (16387 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (16896 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (16897 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (16898 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (16899 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (17408 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_ptr0 + (17409 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (17410 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (17411 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (17920 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (17921 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (17922 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp286 = tl.load(in_ptr0 + (17923 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (18432 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (18433 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (18434 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (18435 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (18944 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp298 = tl.load(in_ptr0 + (18945 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (18946 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (18947 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (19456 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (19457 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (19458 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (19459 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (19968 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (
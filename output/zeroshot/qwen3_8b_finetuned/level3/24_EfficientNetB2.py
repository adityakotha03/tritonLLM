1. High‑level goal: replace every element‑wise activation (ReLU, Sigmoid, Add, AvgPool) and every fused batch‑norm (scale‑bias, clamp) with a Triton kernel that runs on the same GPU memory layout while preserving numerical semantics.

2. Tensor shapes & indexing  
   *All tensors are stored in contiguous NCHW layout with stride pattern (N·C·H·W, C·H·W, H·W, 1).*  
   *For a given block of size `BLOCK_SIZE` the kernel computes a base offset `base = program_id * BLOCK_SIZE` and then iterates over `offsets = base + tl.arange(0, BLOCK_SIZE)`. The mask `offsets < n_elements` guarantees safety for the final partial block.*  
   *When handling a 2‑D tensor (e.g., a 32×32 feature map) the kernel flattens the tensor to a 1‑D view of length `N = C·H·W`. The same flattened index is used for both load and store, preserving the original layout because the stride between adjacent elements is 1.*

3. Parallelization & launch configuration  
   *Program IDs map directly to independent blocks of work; each block processes `BLOCK_SIZE` contiguous elements.*  
   *The grid size is derived from the total element count: `grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)`. This ensures full coverage even when `n_elements` is not a multiple of `BLOCK_SIZE`.*  
   *The chosen `BLOCK_SIZE` values (128, 256, 512) are powers of two that align with warp sizes (32) and fit within the 164 KB shared‑memory limit per SM. They also respect the register budget (max 255 registers per thread).*  
   *For fused kernels (e.g., ReLU + clamp) the same block processes the same flattened tensor, eliminating the need for extra program IDs.*

4. Memory access pattern  
   *Loads are performed with `tl.load(ptr + offsets, mask, other=0.0)`. The `other` argument supplies a safe default (zero) for out‑of‑bounds threads, preventing illegal memory reads.*  
   *Stores use the same `offsets` vector, guaranteeing coalesced writes because consecutive threads write consecutive addresses.*  
   *Batch‑norm kernels load the per‑channel mean (`buf0`) and variance (`buf1`) once per block using `tl.load(ptr + offset, mask)` with `eviction_policy='evict_last'` to keep them in registers across the block.*  
   *Depthwise‑convolution kernels that produce intermediate activation maps reuse the same flattened indexing, avoiding extra transposes.*

5. Numerics & correctness details  
   *All kernels operate in the same dtype as the PyTorch model (float32 for activations, float32 for batch‑norm statistics). No casting is required.*  
   *ReLU is implemented as `max(0, x)`. The Triton helper `triton_helpers.maximum` matches PyTorch’s `torch.relu` semantics.*  
   *Sigmoid is realized with `tl.sigmoid`, which is numerically identical to `torch.sigmoid` for float32.*  
   *Clamping after ReLU (`x > 0`) uses a comparison `tmp0 > 0.0` followed by a mask; the masked load/stores preserve the original zero‑filled padding.*  
   *Batch‑norm fusion computes `y = (x - mean) * rsqrt(var + eps) * weight + bias`. The kernels first load the mean and rsqrt(var+eps) for each output element, then apply the same scaling and bias, exactly reproducing `nn.BatchNorm2d` with its default eps=1e‑5 and affine=True.*  
   *AdaptiveAvgPool2d is replaced by a reduction kernel that sums the 32×32 patch and divides by 1024 (the total number of elements). The reduction uses `tl.sum` across the block dimension, matching the exact arithmetic of `torch.nn.AdaptiveAvgPool2d((1,1))`.*  
   *The final linear layer (`nn.Linear`) remains a standard GEMM; no Triton kernel is needed because the existing cuBLAS path already provides optimal performance.*

6. Summary checklist  
   - [ ] Verify every element‑wise activation (ReLU, Sigmoid, Add) is replaced by a Triton kernel using the flattened index scheme.  
   - [ ] Ensure batch‑norm (scale‑bias, clamp) is fused into a single kernel that loads mean/var once per block.  
   - [ ] Compute grid size as `ceil(numel / BLOCK_SIZE)` for each kernel; use `tl.program_id(0)` to index blocks.  
   - [ ] Apply masks (`offsets < numel`) to all loads/stores to handle non‑divisible dimensions.  
   - [ ] Keep all tensors in NCHW contiguous layout; no transposes required.  
   - [ ] Choose `BLOCK_SIZE` (128/256/512) that fits within SM register budget and shared‑memory limit.  
   - [ ] Preserve dtype consistency (float32 throughout).  
   - [ ] Validate that fused kernels produce identical outputs to the original PyTorch model for a few random inputs.  

Following these reasoning steps yields a Triton‑accelerated `ModelNew` that mirrors the original EfficientNetB2 architecture while exploiting GPU parallelism for every element‑wise operation and batch‑norm fusion.</think>

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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.001953125
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = 0.00390625
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 + tmp5
    tmp11 = tmp7 + tmp10
    tmp12 = tmp1 + tmp11
    tmp13 = 0.0078125
    tmp14 = tmp2 * tmp13
    tmp15 = tmp14 + tmp5
    tmp16 = tmp12 + tmp15
    tmp17 = tmp1 + tmp16
    tmp18 = 0.015625
    tmp19 = tmp2 * tmp18
    tmp20 = tmp19 + tmp5
    tmp21 = tmp17 + tmp20
    tmp22 = tmp1 + tmp21
    tmp23 = 0.03125
    tmp24 = tmp2 * tmp23
    tmp25 = tmp24 + tmp5
    tmp26 = tmp22 + tmp25
    tmp27 = tmp1 + tmp26
    tmp28 = 0.0625
    tmp29 = tmp2 * tmp28
    tmp30 = tmp29 + tmp5
    tmp31 = tmp27 + tmp30
    tmp32 = tmp1 + tmp31
    tmp33 = 0.125
    tmp34 = tmp2 * tmp33
    tmp35 = tmp34 + tmp5
    tmp36 = tmp32 + tmp35
    tmp37 = tmp1 + tmp36
    tmp38 = 0.25
    tmp39 = tmp2 * tmp38
    tmp40 = tmp39 + tmp5
    tmp41 = tmp37 + tmp40
    tmp42 = tmp1 + tmp41
    tmp43 = 0.5
    tmp44 = tmp2 * tmp43
    tmp45 = tmp44 + tmp5
    tmp46 = tmp42 + tmp45
    tmp47 = tmp1 + tmp46
    tmp48 = 1.0
    tmp49 = tmp2 * tmp48
    tmp50 = tmp49 + tmp5
    tmp51 = tmp47 + tmp50
    tmp52 = tmp1 + tmp51
    tmp53 = 2.0
    tmp54 = tmp2 * tmp53
    tmp55 = tmp54 + tmp5
    tmp56 = tmp52 + tmp55
    tmp57 = tmp1 + tmp56
    tmp58 = 4.0
    tmp59 = tmp2 * tmp58
    tmp60 = tmp59 + tmp5
    tmp61 = tmp57 + tmp60
    tmp62 = tmp1 + tmp61
    tmp63 = 8.0
    tmp64 = tmp2 * tmp63
    tmp65 = tmp64 + tmp5
    tmp66 = tmp62 + tmp65
    tmp67 = tmp1 + tmp66
    tmp68 = 16.0
    tmp69 = tmp2 * tmp68
    tmp70 = tmp69 + tmp5
    tmp71 = tmp67 + tmp70
    tmp72 = tmp1 + tmp71
    tmp73 = 32.0
    tmp74 = tmp2 * tmp73
    tmp75 = tmp74 + tmp5
    tmp76 = tmp72 + tmp75
    tmp77 = tmp1 + tmp76
    tmp78 = 64.0
    tmp79 = tmp2 * tmp78
    tmp80 = tmp79 + tmp5
    tmp81 = tmp77 + tmp80
    tmp82 = tmp1 + tmp81
    tmp83 = 128.0
    tmp84 = tmp2 * tmp83
    tmp85 = tmp84 + tmp5
    tmp86 = tmp82 + tmp85
    tmp87 = tmp1 + tmp86
    tmp88 = 256.0
    tmp89 = tmp2 * tmp88
    tmp90 = tmp89 + tmp5
    tmp91 = tmp87 + tmp90
    tmp92 = tmp1 + tmp91
    tmp93 = 512.0
    tmp94 = tmp2 * tmp93
    tmp95 = tmp94 + tmp5
    tmp96 = tmp92 + tmp95
    tl.store(out_ptr0 + x2, tmp96, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.001953125
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = 0.00390625
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 + tmp5
    tmp11 = tmp7 + tmp10
    tmp12 = tmp1 + tmp11
    tmp13 = 0.0078125
    tmp14 = tmp2 * tmp13
    tmp15 = tmp14 + tmp5
    tmp16 = tmp12 + tmp15
    tmp17 = tmp1 + tmp16
    tmp18 = 0.015625
    tmp19 = tmp2 * tmp18
    tmp20 = tmp19 + tmp5
    tmp21 = tmp17 + tmp20
    tmp22 = tmp1 + tmp21
    tmp23 = 0.03125
    tmp24 = tmp2 * tmp23
    tmp25 = tmp24 + tmp5
    tmp26 = tmp22 + tmp25
    tmp27 = tmp1 + tmp26
    tmp28 = 0.0625
    tmp29 = tmp2 * tmp28
    tmp30 = tmp29 + tmp5
    tmp31 = tmp27 + tmp30
    tmp32 = tmp1 + tmp31
    tmp33 = 0.125
    tmp34 = tmp2 * tmp33
    tmp35 = tmp34 + tmp5
    tmp36 = tmp32 + tmp35
    tmp37 = tmp1 + tmp36
    tmp38 = 0.25
    tmp39 = tmp2 * tmp38
    tmp40 = tmp39 + tmp5
    tmp41 = tmp37 + tmp40
    tmp42 = tmp1 + tmp41
    tmp43 = 0.5
    tmp44 = tmp2 * tmp43
    tmp45 = tmp44 + tmp5
    tmp46 = tmp42 + tmp45
    tmp47 = tmp1 + tmp46
    tmp48 = 1.0
    tmp49 = tmp2 * tmp48
    tmp50 = tmp49 + tmp5
    tmp51 = tmp47 + tmp50
    tmp52 = tmp1 + tmp51
    tmp53 = 2.0
    tmp54 = tmp2 * tmp53
    tmp55 = tmp54 + tmp5
    tmp56 = tmp52 + tmp55
    tmp57 = tmp1 + tmp56
    tmp58 = 4.0
    tmp59 = tmp2 * tmp58
    tmp60 = tmp59 + tmp5
    tmp61 = tmp57 + tmp60
    tmp62 = tmp1 + tmp61
    tmp63 = 8.0
    tmp64 = tmp2 * tmp63
    tmp65 = tmp64 + tmp5
    tmp66 = tmp62 + tmp65
    tmp67 = tmp1 + tmp66
    tmp68 = 16.0
    tmp69 = tmp2 * tmp68
    tmp70 = tmp69 + tmp5
    tmp71 = tmp67 + tmp70
    tmp72 = tmp1 + tmp71
    tmp73 = 32.0
    tmp74 = tmp2 * tmp73
    tmp75 = tmp74 + tmp5
    tmp76 = tmp72 + tmp75
    tmp77 = tmp1 + tmp76
    tmp78 = 64.0
    tmp79 = tmp2 * tmp78
    tmp80 = tmp79 + tmp5
    tmp81 = tmp77 + tmp80
    tmp82 = tmp1 + tmp81
    tmp83 = 128.0
    tmp84 = tmp2 * tmp83
    tmp85 = tmp84 + tmp5
    tmp86 = tmp82 + tmp85
    tmp87 = tmp1 + tmp86
    tmp88 = 256.0
    tmp89 = tmp2 * tmp88
    tmp90 = tmp89 + tmp5
    tmp91 = tmp87 + tmp90
    tmp92 = tmp1 + tmp91
    tmp93 = 512.0
    tmp94 = tmp2 * tmp93
    tmp95 = tmp94 + tmp5
    tmp96 = tmp92 + tmp95
    tl.store(out_ptr0 + x2, tmp96, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 32 * x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (5 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (6 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp18 = tl.load(in_ptr0 + (7 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (8 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp24 = tl.load(in_ptr0 + (9 + 32 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (10 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (11 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (12 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (13 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (14 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (15 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (16 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (17 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (18 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (19 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (20 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (21 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (22 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (23 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (24 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (25 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (26 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (27 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (28 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (29 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (30 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (31 + 32 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 0.001953125
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp7 = tmp1 + tmp6
    tmp8 = 0.00390625
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 + tmp5
    tmp11 = tmp7 + tmp10
    tmp12 = tmp1 + tmp11
    tmp13 = 0.0078125
    tmp14 = tmp2 * tmp13
    tmp15 = tmp14 + tmp5
    tmp16 = tmp12 + tmp15
    tmp17 = tmp1 + tmp16
    tmp18 = 0.015625
    tmp19 = tmp2 * tmp18
    tmp20 = tmp19 + tmp5
    tmp21 = tmp17 + tmp20
    tmp22 = tmp1 + tmp21
    tmp23 = 0.03125
    tmp24 = tmp2 * tmp23
    tmp25 = tmp24 + tmp5
    tmp26 = tmp22 + tmp25
    tmp27 = tmp1 + tmp26
    tmp28 = 0.0625
    tmp29 = tmp2 * tmp28
    tmp30 = tmp29 + tmp5
    tmp31 = tmp27 + tmp30
    tmp32 = tmp1 + tmp31
    tmp33 = 0.125
    tmp34 = tmp2 * tmp33
    tmp35 = tmp34 + tmp5
    tmp36 = tmp32 + tmp35
    tmp37 = tmp1 + tmp36
    tmp38 = 0.25
    tmp39 = tmp2 * tmp38
    tmp40 = tmp39 + tmp5
    tmp41 = tmp37 + tmp40
    tmp42 = tmp1 + tmp41
    tmp43 = 0.5
    tmp44 = tmp2 * tmp43
    tmp45 = tmp44 + tmp5
    tmp46 = tmp42 + tmp45
    tmp47 = tmp1 + tmp46
    tmp48 = 1.0
    tmp49 = tmp2 * tmp48
    tmp50 = tmp49 + tmp5
    tmp51 = tmp47 + tmp50
    tmp52 = tmp1 + tmp51
    tmp53 = 2.0
    tmp54 = tmp2 * tmp53
    tmp55 = tmp54 + tmp5
    tmp56 = tmp52 + tmp55
    tmp57 = tmp1 + tmp56
    tmp58 = 4.0
    tmp59 = tmp2 * tmp58
    tmp60 = tmp59 + tmp5
    tmp61 = tmp57 + tmp60
    tmp62 = tmp1 + tmp61
    tmp63 = 8.0
    tmp64 = tmp2 * tmp63
    tmp65 = tmp64 + tmp5
    tmp66 = tmp62 + tmp65
    tmp67 = tmp1 + tmp66
    tmp68 = 16.0
    tmp69 = tmp2 * tmp
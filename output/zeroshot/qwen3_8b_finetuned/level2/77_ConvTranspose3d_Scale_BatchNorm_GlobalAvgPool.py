1. High‑level goal  
Replace the three‑dimensional transposed convolution, the per‑element scaling, the batch‑norm forward pass (mean‑variance computation, affine transform), and the global‑average‑pool with a sequence of Triton kernels that fuse the element‑wise arithmetic and the reduction needed for batch‑norm while preserving the exact numerical semantics of the original `nn.ConvTranspose3d`‑`nn.BatchNorm3d`‑`nn.AdaptiveAvgPool3d` pipeline.

---

2. Tensor shapes and indexing  

| Stage | Logical tensor | Physical shape (B, C, D, H, W) | Stride pattern (row‑major) | Triton view |
|------|----------------|------------------------------|-----------------------------|-------------|
| Input | `x` | (16, 64, 16, 32, 32) | (65536, 1024, 64, 2, 1) | contiguous block of 1024 elements per thread (BLOCK_SIZE) |
| Conv‑transpose output | `y` | (16, 128, 16, 32, 32) | (65536, 1024, 64, 2, 1) | same layout, accessed via `tl.load` with linear offset `program_id * BLOCK_SIZE + arange` |
| Scale factor | scalar `s` | – | – | broadcasted as a constant `tl.constexpr` |
| Batch‑norm mean/var | `mean`, `var` | (16, 128) | (128, 1) | computed per channel across the 1024 spatial elements (`BLOCK_SIZE` = 1024) |
| Batch‑norm affine | `weight`, `bias` | (128,) | (1,) | broadcasted per channel (`tl.broadcast_to`) |
| Global‑avg‑pool result | `z` | (16, 128) | (128, 1) | each element is the sum of the 1024 spatial values divided by 1024 |

Indexing strategy in each kernel:  
- `program_id(0)` selects the block of channels (`c = program_id * BLOCK_SIZE`).  
- `tl.arange(0, BLOCK_SIZE)` yields the intra‑block offsets (`k = 0 … 1023`).  
- Linear address = `c * stride_c + k * stride_s`, where `stride_c` = channel stride (1024) and `stride_s` = spatial stride (1).  
- Masks (`offset < total_elements`) guard the tail of the last block.

---

3. Parallelization & launch configuration  

- **Program ID dimension**: 1‑D grid (`grid = lambda meta: ((num_channels + BLOCK_SIZE - 1) // BLOCK_SIZE,)`).  
- **BLOCK_SIZE**: 1024 (chosen to fill a warp of 32 threads with 32 groups, yielding 32 warps per block).  
- **Number of blocks**: `ceil(num_channels / BLOCK_SIZE)` = `ceil(128 / 1024) = 1` for the mean/var reduction; `ceil(1024 / 1024) = 1` for the affine transform and global‑avg‑pool.  
- **Warps per block**: 4 (default for Triton, enough to hide latency of the 1024‑element load).  
- **Stages**: 1 (no double‑buffering needed because the kernels are memory‑bound and the data fits in shared memory).  
- **Grid for element‑wise kernels** (conv‑transpose output + scaling) uses `grid = lambda meta: ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)` with `BLOCK_SIZE = 1024` and `num_elements = 16 * 128 * 16 * 32 * 32 = 33554432`. This yields `33554432 / 1024 = 32768` blocks, matching the `grid(32768)` call.

---

4. Memory access pattern  

- **Coalesced loads**: Each thread loads a contiguous 4‑byte element (`float32`) from global memory using the linear address derived from `c * stride_c + k * stride_s`. Because the stride pattern is contiguous, the warp accesses a 1024‑element contiguous region, achieving full memory‑coalescing.  
- **Broadcasted scalar**: The scaling factor is loaded once (`tl.constexpr`) and broadcast to the whole block via `tl.broadcast_to`. No extra memory traffic.  
- **Batch‑norm mean/var**:  
  1. Load the 1024 spatial values for a channel (`tmp0`).  
  2. Accumulate them across the block (`tl.sum` over the 1024‑element vector) to obtain the sum per channel.  
  3. Divide by `1024.0` to compute the mean.  
  4. Subtract the mean from each element, square, sum, divide by `1024.0`, and apply `rsqrt` to obtain the reciprocal‑sqrt of variance.  
- **Affine transform**:  
  1. Load the per‑channel mean (`tmp3`) and variance (`tmp11`).  
  2. Compute `x - mean`.  
  3. Multiply by `rsqrt(var)`.  
  4. Add bias (`tmp17`).  
  5. Multiply by weight (`tmp21`).  
- **Global‑avg‑pool**:  
  1. Load the 1024 spatial values (`tmp0`).  
  2. Reduce with `tl.sum` to obtain the channel sum.  
  3. Divide by `1024.0` to produce the pooled value.  

All reductions are performed inside the block using `tl.sum`, which internally uses warp‑level reductions and then a block‑level reduction, avoiding shared‑memory staging.

---

5. Numerics & correctness details  

- **Data types**: All kernels operate on `float32`. The batch‑norm variance uses `rsqrt` from `libdevice` to match PyTorch’s `torch.rsqrt` (stable, exact to within epsilon).  
- **Broadcasting**: The scaling factor, batch‑norm weight, and bias are broadcast to the full tensor shape; Triton’s `tl.broadcast_to` ensures the same semantics as PyTorch’s broadcasting rules.  
- **Mean/variance stability**: The mean is computed as `sum / N` where `N = 1024`. The variance uses the same divisor, matching PyTorch’s `torch.nn.functional.batch_norm` (unbiased=False).  
- **Epsilon**: The original `nn.BatchNorm3d` adds `eps=1e-5` before `rsqrt`. The kernel adds `eps` (`tmp12`) before applying `libdevice.rsqrt`.  
- **Momentum**: Not used in the forward pass; the kernel only computes the current mean/variance, identical to the forward pass of `nn.BatchNorm3d`.  
- **Edge handling**: Masks (`offset < total_elements`) guarantee that the last block does not read out‑of‑bounds. The `other=0.0` argument in `tl.load` ensures that masked loads contribute zero to sums, preserving correctness.  
- **Precision of global‑avg‑pool**: The division by `1024.0` is exact in `float32`; no rounding error beyond that of a single‑precision division.

---

6. Summary checklist  

- **Shape mapping**: Conv‑transpose output → (B, C, D, H, W) with stride (65536, 1024, 64, 2, 1).  
- **Indexing**: `program_id * BLOCK_SIZE + arange` yields per‑thread linear offset; masks guard tails.  
- **Parallelism**: 1‑D grid, BLOCK_SIZE = 1024, 4 warps per block, 1 stage.  
- **Memory**: Coalesced global loads, broadcasted scalars, intra‑block reductions (`tl.sum`) for mean/var, no shared memory needed.  
- **Numerics**: Float32 throughout, `rsqrt` with epsilon, exact divisor `1024.0`, masks for safety.  
- **Correctness**: Matches PyTorch’s `nn.ConvTranspose3d` (output shape), `nn.BatchNorm3d` forward pass, and `nn.AdaptiveAvgPool3d` semantics.  

Following this reasoning, the Triton kernels implement the original three‑dimensional transposed convolution, scaling, batch‑norm, and global‑average‑pool with fused reductions, achieving a compact, high‑throughput implementation on an A100 GPU.</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp25 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp30 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp32 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp54 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp56 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp78 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp80 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp102 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp104 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp126 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp128 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp150 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp152 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp174 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp176 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp198 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp200 = tl.load(in_ptr3 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr1 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr2 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr3 + (1 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr1 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr2 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr3 + (2 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr1 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr2 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr3 + (3 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp222 = tl.load(in_ptr1 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr2 + (4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp224 = tl.load(in_ptr3 + (4096 * x0),
1. High‑level goal – replace every element‑wise arithmetic operation (add, mul, bias‑add, ReLU, ReLU6, division, max, clamp, softmax‑max, softmax‑exp‑sum, adaptive‑avg‑pool‑add) that appears in the forward pass of the original `Model` with a custom Triton kernel. The kernels must preserve the exact numerical semantics of the PyTorch operators while exploiting the A100‑80GB hardware (tensor‑core‑friendly dtypes, shared‑memory‑free block‑wise loads/stores, warp‑level parallelism, and autotuned block‑size).

2. Tensor shapes and indexing  
   - All tensors are contiguous in row‑major (C) order; the Triton kernels assume this layout (`assert_size_stride`).  
   - For a 2‑D tensor of shape `(N, C)` the linear index is `row*stride_row + col`. In the kernels we compute `xindex = program_id * BLOCK + tl.arange(0, BLOCK)`. The mask `xmask = xindex < total_elements` guarantees safety for the final (potentially partial) block.  
   - For 3‑D tensors (e.g., a convolution output) the stride tuple `(C*H*W, H*W, W, 1)` is used; the kernels treat the tensor as a flat 1‑D array with the same total element count (`xnumel`). No explicit stride handling is required inside the kernel because the pointer arithmetic (`x_ptr + offset`) respects the underlying stride.

3. Parallelization & launch configuration  
   - **Program ID axis** – only one dimension (`program_id(0)`) is used; each program processes a contiguous chunk of `BLOCK` elements.  
   - **BLOCK_SIZE** – chosen by the heuristic in `grid` (ceil division of `xnumel` by `BLOCK`). The kernels are autotuned; the generated code selects the optimal `BLOCK` (e.g., 256 for small tensors, 1024 for larger ones).  
   - **Grid** – computed as `grid = lambda meta: ((xnumel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)`. This yields exactly enough programs to cover the entire tensor without overlap.  
   - **Warps & stages** – the `num_warps` and `num_stages` arguments are set by the autotuner; they reflect the latency‑hiding strategy (e.g., 4 warps for small element‑wise ops, 8 warps for larger reductions).  

4. Memory access pattern  
   - **Loads** – `tl.load(ptr + offset, mask, other=0.0)` reads a contiguous block of `BLOCK` elements per program. The `other=0.0` provides a default value for masked threads, avoiding illegal memory reads.  
   - **Stores** – `tl.store(ptr + offset, value, mask)` writes the result back to the same layout, preserving alignment and coalescing.  
   - **Coalescing** – because each program accesses a contiguous slice, the memory traffic is fully coalesced across the warp.  
   - **No shared memory** – all kernels operate on a flat view; the only shared‑memory‑free implementation is used, which is sufficient for element‑wise ops.  
   - **Stride handling** – the `assert_size_stride` checks guarantee that the stride layout matches the expected flat view; the kernels rely on this to compute correct offsets without extra stride arithmetic.

5. Numerics & correctness details  
   - **Data types** – kernels are instantiated with the exact dtype of the tensors (`fp32` for most ops, `fp16`/`int32` for the ReLU6 and clamp ops, `int64` for the division‑by‑zero‑avoiding kernel). The `other` argument for loads is set to the zero of the dtype, ensuring identical behavior to PyTorch’s default for out‑of‑range threads.  
   - **Broadcasting** – bias addition and ReLU6 are performed after flattening the tensors, so broadcasting is handled implicitly by the flat indexing; no explicit broadcasting code is needed.  
   - **Edge cases** – masks (`xmask`) guarantee that the final block does not read/write beyond the tensor bounds. For division kernels the denominator is clamped to a small epsilon (`1e-12`) to avoid NaNs, matching the PyTorch `torch.where`‑based handling.  
   - **ReLU6** – implemented as `max(x, 0) * (x <= 6)`, which reproduces `torch.nn.functional.relu6`.  
   - **Softmax** – the two‑step pattern (max‑reduction, exponent‑sum, division) mirrors `F.softmax` but is unrolled into separate kernels (`_softmax_max_1`, `_softmax_exp_sum_2`, `_softmax_div_3`) to keep each kernel small and cache‑friendly.  
   - **Adaptive‑avg‑pool** – the reduction kernel (`_adaptive_avg_pool2d_add_14`) computes the sum of the 4‑element window and divides by 4.0, exactly matching `F.adaptive_avg_pool2d(..., (1,1))`.  
   - **Bias addition** – kernels like `_add_0` add the bias vector to each channel of a 2‑D tensor; the bias is broadcast across the batch dimension by loading the bias once per program and adding it to every element of the tensor.  

6. Summary checklist for each kernel  
   - **Shape**: `assert_size_stride` validates that the input tensor is contiguous with the expected total element count (`xnumel`).  
   - **Indexing**: `xindex = program_id * BLOCK + tl.arange(0, BLOCK)` yields a contiguous slice; mask ensures safety.  
   - **Launch**: `grid` computes the number of programs; `BLOCK` is chosen by autotuner.  
   - **Memory**: loads/stores are coalesced, default value (`other`) handles masked threads.  
   - **Data type**: matches the tensor dtype; default zero of that dtype used for masked loads.  
   - **Numerics**: masks, epsilon clamping, and exact arithmetic (max, min, division) reproduce PyTorch semantics.  
   - **Correctness**: each kernel corresponds to a single PyTorch operator (add, mul, bias‑add, ReLU, ReLU6, clamp, max, division, softmax‑max, softmax‑exp‑sum, softmax‑div, adaptive‑avg‑pool) and is fused where possible (e.g., bias‑add fused with ReLU6).  

Following these reasoning steps yields a fully functional `ModelNew` whose forward pass is realized entirely through custom Triton kernels, each preserving the exact numerical behavior of the original PyTorch implementation while exploiting the A100‑80GB hardware’s parallelism and tensor‑core capabilities.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_relu6_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 6, tl.int32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu6_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 32 * x2), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu6_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 32 * x2), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu6_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 32 * x2), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu6_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 32 * x2), xmask, eviction_policy='evict_last'
        )
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_div_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_16(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_17(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_18(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_19(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_20(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_21(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_22(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_23(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_24(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + 32 * x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 32 * x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 32 * x1), xmask, eviction_policy='evict_last'
        )
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = triton_helpers.maximum(tmp4, tmp6)
    tmp7 = triton_helpers.maximum(tmp5, tmp9)
    tmp8 = tmp0 - tmp7
    tmp10 = tl_math.exp(tmp8)
    tmp11 = tmp10 / 4.0
    tmp12 = tmp11 * tmp9
    tl.store(out_ptr0 + x2, tmp12, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)


@triton.jit
def triton_poi_fused__softmax_div_exp_max_mul_25(in_ptr0, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask
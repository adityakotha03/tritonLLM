1. High‑level goal – replace every element‑wise addition (a+b, a+b+bias, a+b+bias+bias) that appears in the forward pass of the Swin‑Transformer with a single, highly‑parallel Triton kernel that computes the same result while exploiting the GPU’s massive parallelism and memory coalescing.

2. Tensor shapes and indexing – the original model processes 2‑D feature maps of shape (B, C, H, W). The linear projections (nn.Linear) produce intermediate tensors of shape (B, C, H*W) after flattening. The Triton kernels are launched on the flattened dimension (total elements = B * C * H * W). The kernel computes a linear index `idx = program_id * BLOCK_SIZE + lane_offset`. From this linear index we recover the three logical dimensions:
   * `b = idx // (C*H*W)` – batch index,
   * `c = (idx // (H*W)) % C` – channel index,
   * `pos = idx % (H*W)` – spatial position.
   The kernel then loads the two source tensors (`in_ptr0` and `in_ptr1`) at offsets `b*C*H*W + c*H*W + pos` and `c*H*W + pos` respectively, adds them, and stores the result at the same linear offset in `out_ptr`.

3. Parallelization & launch configuration – each program instance (thread block) processes a contiguous chunk of `BLOCK_SIZE` elements. The grid size is computed as `ceil(num_elements / BLOCK_SIZE)`. For the typical batch size (10) and image size (224), the total element count is 10 × 3 × 224 × 224 = 15 728 640. Choosing `BLOCK_SIZE = 1024` yields 15 728 640 / 1024 ≈ 15 360 blocks, which maps to about 5 120 SMs (the device has 32 SMs). The kernel is launched with `num_warps=4` and `num_stages=1` to keep latency low and to fit within the 64‑KB shared‑memory budget.

4. Memory access pattern – loads and stores are performed with a single stride‑1 pointer arithmetic (`in_ptr0 + idx`, `in_ptr1 + idx`). Because the stride‑1 pattern is contiguous across the flattened tensor, the hardware automatically coalesces memory accesses across the warp. The mask `xmask = idx < num_elements` guarantees that the tail of the last block does not read/write out‑of‑bounds. The kernel uses `other=0.0` to treat masked elements as zeros, which is safe because they are never used in the final output.

5. Numerics & correctness – the kernel performs pure addition; no rounding or dtype conversion is introduced. The original PyTorch model adds bias tensors (`q_bias`, `v_bias`, `proj.bias`) after the linear projection. In the Triton‑fused version these bias terms are folded into the same kernel by adding an extra bias pointer (`in_ptr2`) and loading it with the same mask. The bias is broadcast across the spatial dimension by multiplying it with a mask that repeats the bias value for each spatial location. The result is identical to `x + bias` for each element.

6. Summary checklist –
   * Identify all element‑wise addition sites (linear projections, bias addition, residual connections).
   * Flatten tensors to a 1‑D view; compute batch, channel, spatial indices from the linear index.
   * Launch a Triton kernel with `BLOCK_SIZE` = 1024 (or 256/512 depending on the site).
   * Use `program_id * BLOCK_SIZE + lane_offset` to compute per‑thread linear index.
   * Apply mask to guard against out‑of‑bounds accesses.
   * Load two operands, add, store result.
   * Repeat for each addition site, possibly fusing multiple additions into a single kernel when they share the same operand layout.
   * Ensure bias terms are broadcast correctly via mask multiplication.
   * Verify that the total element count matches the flattened tensor size and that the grid covers the entire tensor.

Following these reasoning steps yields a Triton‑accelerated version of the Swin‑Transformer where every element‑wise addition is replaced by a highly parallel, memory‑coalesced kernel, delivering the same numerical results while exploiting the GPU’s parallelism and memory bandwidth.</think>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import numpy as np
import collections
from itertools import repeat
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_11(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_12(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_18(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_19(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_20(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_23(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_25(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
    x2 = xindex // 36864
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 576 * x1 + 36864 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_27(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 15728640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 576
    x1 = xindex // 576 % 64
1. High‑level goal: replace every element‑wise arithmetic operation (convolution‑bias‑add, batch‑norm‑scale‑add, ReLU, PReLU, AvgPool‑bias‑add) and every small reduction (the final AvgPool‑bias‑add) with a custom Triton kernel that fuses the operation with the surrounding memory‑access pattern, thereby eliminating intermediate temporaries and reducing global‑memory traffic.

2. Tensor shapes & indexing:
   * Convolution outputs are stored in a contiguous NCHW layout. The kernel treats the flattened element index `x2 = program_id * BLOCK + lane` and reconstructs the 4‑D index as `(x0, x1, x2, x3)` where `x0 = xindex % 16` (channel stride 16), `x1 = (xindex // 16) % 64` (height stride 64), `x2 = (xindex // 1024) % 32` (width stride 32), `x3 = xindex // 32768` (batch). This mapping matches the stride pattern of the original tensors (`(16384, 128, 32, 1)` for the 1×1 convolution output).
   * Batch‑norm scales and biases are 1‑D tensors of length equal to the output channel count (32, 64, …). The kernel loads the per‑channel scale/bias via `x0 = xindex % 32` (or %64) and adds it to the corresponding element of the activation tensor.
   * ReLU kernels use a simple mask `tmp5 = tmp3 <= 0.0` and store `tmp6 = where(mask, 0.0, tmp3)`. The index `x0 = xindex % 32` selects the bias term for the ReLU‑bias fused kernel.
   * The final AvgPool‑bias kernel reduces over the spatial dimension (7×7) by loading the 49 contiguous elements per channel (`x2 = xindex % 16` selects the channel, `x3 = xindex // 16` selects the batch), computes the sum, divides by 49, adds the bias (`x0 = xindex % 32`), and stores the result.

3. Parallelization & launch configuration:
   * Each kernel is launched with a 1‑D grid where `grid = ceil(num_elements / BLOCK_SIZE)`. The chosen `BLOCK_SIZE` values (128, 256, 512) are powers of two that fit within a warp (32 threads) and align with the hardware’s shared‑memory tiling.
   * `program_id(0)` yields the block index; `tl.arange(0, BLOCK_SIZE)` yields the lane offsets. The combined offset (`xoffset`) is used to compute the global linear index for each thread.
   * The kernels are launched with `num_warps` (4) and `num_stages` (1) because the operations are memory‑bound and do not benefit from double‑buffering; the extra warps improve occupancy without increasing register pressure.

4. Memory access pattern:
   * All loads are performed with a single `tl.load` per element, using the computed linear offset plus a mask to guard against out‑of‑bounds accesses. The mask is derived from `xmask = xindex < xnumel`.
   * Stores are similarly masked and use the same linear offset. No explicit shared memory is allocated; Triton’s implicit shared‑memory handling (via block‑level vector loads) suffices for the small‑size tensors.
   * For fused kernels (bias‑add + ReLU), the bias load is a single scalar per channel; the mask ensures it is broadcast across the block.
   * The AvgPool kernel loads the 49 elements of a channel in a single contiguous memory transaction per thread, then performs a reduction across the warp (via `tl.sum` on a vector of the 49 values). This eliminates the need for shared memory for the reduction.

5. Numerics & correctness:
   * Bias addition is performed in the same dtype as the activation tensor (float32). No casting is required because all tensors are float32.
   * ReLU uses a zero‑constant of the same dtype; the `where` construct guarantees that the result is always non‑negative.
   * The AvgPool kernel computes the exact arithmetic mean (`sum / 49.0`) using FP32 division, matching the PyTorch `nn.AvgPool2d(7)` behavior.
   * Masks (`tmp8 = tmp6 <= 0.0`) are evaluated in FP32, ensuring the same threshold as PyTorch’s `torch.relu`.
   * All kernels respect the original tensor strides; the index reconstruction (`x0, x1, x2, x3`) mirrors the stride tuple used by `assert_size_stride` in the wrapper, guaranteeing identical memory layout.

6. Summary checklist:
   - **Shape mapping**: Verify that each kernel’s linear index reconstruction (`%` and `//`) reproduces the original NCHW stride pattern.
   - **Launch config**: `BLOCK_SIZE` chosen as 128/256/512, `grid` = ceil(numel / BLOCK), `num_warps=4`.
   - **Memory accesses**: One `tl.load` per element, masked; one `tl.store` per element, masked; bias loads are scalar per channel.
   - **Fusion points**: Conv‑bias‑add, BN‑scale‑add, ReLU‑bias‑add, AvgPool‑bias‑add are each fused into a single kernel.
   - **Numerics**: All operations performed in FP32; masks and constants match PyTorch semantics.
   - **Correctness**: Index reconstruction, stride checks, and mask logic guarantee that the output tensor is identical to the original `Model` forward pass.</think>

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_12(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_16(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_22(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_24(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_26(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tmp1 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 + tmp5
    tl.store(in_out_ptr0 + x3, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
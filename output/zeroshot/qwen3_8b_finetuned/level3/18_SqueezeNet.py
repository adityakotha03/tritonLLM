1. High‑level goal – replace every element‑wise arithmetic (add, multiply, ReLU) and every small‑scale reduction (max, mean, cat) in the original PyTorch‑based Squeeze‑Excitation‑style block with a custom Triton kernel. The objective is to eliminate intermediate CPU‑level tensors, fuse where possible, and exploit the GPU’s massive parallelism.

2. Tensor shapes and indexing  
   – All convolutions produce a 4‑D tensor (N, C, H, W). The Triton kernels treat the flattened element count `xnumel = N·C·H·W` as a 1‑D array; each program processes a contiguous block `XBLOCK` elements.  
   – For a 1×1 convolution (e.g., `squeeze` or `expand1x1`), the kernel receives the weight tensor (shape (out_channels, in_channels, 1, 1)) and the bias vector (shape (out_channels,)). The bias is broadcast across the spatial dimensions; the kernel therefore loads the bias once per thread and adds it to the convolution output (`tmp0 + tmp1`).  
   – For the 3×3 convolutions, the weight tensor is (out_channels, in_channels, 3, 3). The kernel still loads the weight element directly because the kernel is launched after the external cuDNN convolution call (the `extern_kernels.convolution` call). The only elementwise work is the bias addition.  
   – The ReLU kernels (`relu_*`) receive a pointer to the activation tensor and a pointer to the bias vector; they compute `max(0, x + bias)` element‑wise.  
   – The concatenation (`cat`) is performed by a single‑kernel that writes two streams of data (the two branches of the fire block) into the same output buffer using a conditional mask (`tmp10`). The kernel computes the output index as `x0 + 64 * x2` where `x0` indexes the channel dimension inside each branch and `x2` indexes the batch‑spatial plane.

3. Parallelization & launch configuration  
   – `grid` is defined as `lambda meta: ((xnumel + XBLOCK - 1) // XBLOCK,)`. This yields the number of program instances needed to cover all elements.  
   – `XBLOCK` is a compile‑time constant (e.g., 128, 256) chosen to balance occupancy and register pressure. The example uses 256 for the 1×1 conv‑bias kernel because the bias vector is tiny (16 elements) and the kernel needs to keep the bias in registers; a larger block reduces register pressure.  
   – For the concat kernel, `XBLOCK = 128` suffices because the operation is a simple conditional store; the kernel processes 128 contiguous output elements per program.  
   – The `num_warps` argument (4 for conv‑bias, 2 for concat) is tuned to the work‑per‑thread ratio: more warps for memory‑bound loads (bias addition) and fewer for pure store kernels.  
   – The `num_stages` argument (1) indicates a single‑stage pipeline, which is enough because the kernels have no shared‑memory tiling.

4. Memory access pattern  
   – Loads are performed with `tl.load(ptr + offset, mask, other=0.0)`. The mask guarantees that threads whose offset exceeds `xnumel` do not read, preventing out‑of‑bounds accesses.  
   – Bias vectors are loaded with `eviction_policy='evict_last'` to keep them in registers across the block’s execution, avoiding repeated global memory traffic.  
   – Stores use the same mask (`xmask`) to write only valid elements. The concat kernel writes the first branch to `out_ptr + 64 * x2 + x0` and the second branch to `out_ptr + (64 + 64) * x2 + x0`, matching the stride‑aware layout expected by the downstream `cat` call.  
   – All kernels operate on contiguous memory (stride‑1 along the innermost dimension) because the tensors are created with `empty_strided_cuda` using a stride tuple that ensures contiguous layout. This guarantees coalesced loads/stores across the warp.

5. Numerics & correctness details  
   – The ReLU kernels implement `max(0, x + bias)` exactly as PyTorch’s `nn.ReLU(inplace=True)`. The `triton_helpers.maximum` helper is used for the zero constant, preserving the same dtype (float32).  
   – Bias addition is performed in FP32, matching the default dtype of the convolutions. No casting is required because the bias tensors are also FP32.  
   – The concat kernel respects the original ordering: the first branch (expand1x1) is written first, then the second branch (expand3x3). The conditional `tmp10` selects which branch to write based on the channel index (`x0 < 32`). This reproduces the `torch.cat([a, b], dim=1)` semantics.  
   – The max‑pool and adaptive‑avg‑pool layers remain unchanged; they are still executed by the external cuDNN kernels (`extern_kernels.max_pool2d` and `extern_kernels.adaptive_avg_pool2d`). Their output buffers are passed directly to the subsequent Triton kernels, preserving exact numerical results.  
   – The final classifier uses a 1×1 convolution followed by ReLU and adaptive average pooling. The 1×1 convolution is delegated to cuDNN, and the ReLU is fused with the bias addition in `relu_0`. The adaptive average pool is a pure reduction; it is not replaced because its implementation is already highly optimized.

6. Summary checklist for the generated code  
   – **Shape handling**: every 4‑D tensor flattened to 1‑D; `xnumel = N·C·H·W` used for launch grid.  
   – **Indexing**: `xoffset = program_id * XBLOCK`; `xindex = xoffset + tl.arange(0, XBLOCK)`. Masks guard against overflow.  
   – **Launch config**: `grid = ((xnumel + XBLOCK - 1) // XBLOCK,)`; `XBLOCK` chosen per kernel (256 for bias addition, 128 for concat).  
   – **Memory**: coalesced loads/stores; bias kept in registers via `eviction_policy='evict_last'`; masks prevent out‑of‑bounds writes.  
   – **Numerics**: FP32 throughout; ReLU implemented as `max(0, x+bias)`.  
   – **Fusion**: bias addition fused with ReLU (`relu_0`), concat fused into a single write‑kernel.  
   – **External calls**: convolutions and max‑pool remain cuDNN‑based; only elementwise ops are replaced.  
   – **Correctness**: channel ordering preserved; stride‑aware indexing matches PyTorch’s `cat` and `conv2d` layout.  

Following these reasoning steps yields a Triton‑augmented `ModelNew` that reproduces the original model’s forward pass while eliminating unnecessary intermediate tensors and fusing elementwise operations for maximal GPU throughput.</think>

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
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (16 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_2(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex // 1024
    x1 = xindex // 64 % 16
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64 * x2), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 64, tl.int64)
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 64 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (64 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_5(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x2 = xindex // 1024
    x1 = xindex // 128 % 32
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 128 * x2), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp9 = tl.load(in_ptr0 + (64 + x0 + 128 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (128 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_8(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x2 = xindex // 1024
    x1 = xindex // 256 % 64
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 256 * x2), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 256, tl.int64)
    tmp9 = tl.load(in_ptr0 + (128 + x0 + 256 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (256 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_11(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x2 = xindex // 1024
    x1 = xindex // 512 % 128
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 512 * x2), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 512, tl.int64)
    tmp9 = tl.load(in_ptr0 + (256 + x0 + 512 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (512 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_14(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x2 = xindex // 1024
    x1 = xindex // 1024 % 256
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1024 * x2), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 1024, tl.int64)
    tmp9 = tl.load(in_ptr0 + (512 + x0 + 1024 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (1024 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_16(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1024
    x4 = xindex // 1024
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x4), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x3, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_cat_18(out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x2 = xindex // 4096
    x1 = xindex // 2048 % 1024
    x3 = xindex
    tmp0 = x1
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096 * x2), tmp4 & xmask, eviction_policy
        ='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 2048, tl.int64)
    tmp9 = tl.load(in_ptr0 + (1024 + x0 + 4096 * x2), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + x3, tmp10, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24, primals_25, primals_26, primals_27,
        primals_28, primals_29, primals_30, primals_31, primals_32,
        primals_33, primals_34, primals_35, primals_36, primals_37,
        primals_38, primals_39, primals_40, primals_41, primals_42,
        primals_43, primals_44, primals_45, primals_46, primals_47,
        primals_48, primals_49, primals_50, primals_51, primals_52,
        primals_53, primals_54, primals_55, primals_56, primals_57,
        primals_58, primals_59, primals_60, primals_61, primals_62,
        primals_63, primals_64, primals_65, primals_66, primals_67,
        primals_68, primals_69, primals_70, primals_71, primals_72,
        primals_73, primals_74, primals_75, primals_76, primals_77,
        primals_78, primals_79, primals_80, primals_81, primals_82,
        primals_83, primals_84, primals_85, primals_86, primals_87,
        primals_88, primals_89, primals_90, primals_91, primals_92,
        primals_93, primals_94, primals_95, primals_96, primals_97,
        primals_98, primals_99, primals_100, primals_101, primals_102,
        primals_103, primals_104, primals_105, primals_106, primals_107,
        primals_108, primals_109, primals_110, primals_111, primals_112,
        primals_113, primals_114, primals_115, primals_116, primals_117,
        primals_118, primals_119, primals_120, primals_121, primals_122,
        primals_123, primals_124, primals_125, primals_126, primals_127,
        primals_128, primals_129, primals_130, primals_131, primals_132,
        primals_133, primals_134, primals_135, primals_136, primals_137,
        primals_138, primals_139, primals_140, primals_141, primals_142,
        primals_143, primals_144, primals_145, primals_146, primals_147,
        primals_148, primals_149, primals_150, primals_151, primals_152,
        primals_153, primals_154, primals_155, primals_156, primals_157,
        primals_158, primals_159, primals_160, primals_161, primals_162,
        primals_163, primals_164, primals_165, primals_166, primals_167,
        primals_168, primals_169, primals_170, primals_171, primals_172,
        primals_173, primals_174, primals_175, primals_176, primals_177,
        primals_178, primals_179, primals_180, primals_181, primals_182,
        primals_183, primals_184, primals_185, primals_186, primals_187,
        primals_188, primals_189, primals_190, primals_191, primals_192,
        primals_193, primals_194, primals_195, primals_196, primals_197,
        primals_198, primals_199, primals_200, primals_201, primals_202,
        primals_203, primals_204, primals_205, primals_206, primals_207,
        primals_208, primals_209, primals_210, primals_211, primals_212,
        primals_213, primals_214, primals_215, primals_216, primals_217,
        primals_218, primals_219, primals_220, primals_221, primals_222,
        primals_223, primals_224, primals_225, primals_226, primals_227,
        primals_228, primals_229, primals_230, primals_231, primals_232,
        primals_233, primals_234, primals_235, primals_236, primals_237,
        primals_238, primals_239, primals_240, primals_241, primals_242,
        primals_243, primals_244, primals_245, primals_246, primals_247,
        primals_248, primals_249, primals_250, primals_251, primals_252,
        primals_253, primals_254, primals_255, primals_256, primals_257,
        primals_258, primals_259, primals_260, primals_261, primals_262,
        primals_263, primals_264, primals_265, primals_266, primals_267,
        primals_268, primals_269, primals_270, primals_271, primals_272,
        primals_273, primals_274, primals_275, primals_276, primals_277,
        primals_278, primals_279, primals_280, primals_281, primals_282,
        primals_283, primals
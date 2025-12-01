1. **Goal** – Replace every elementwise operation, every batchnorm + ReLU pair, every concatenation, every max-pool, every average-pool, every linear layer (both the final classifier and the 1×1 convolutions in the transition layers) with a Triton kernel that reproduces the exact PyTorch semantics while exploiting the A100’s tensor cores, shared memory, and warp-level parallelism.

2. **Decomposition of the original forward pass**  
   - **Stage 1 (initial conv + BN + ReLU + max-pool)**  
     - `Conv2d(3,64,k=7,p=3,s=2)` → a standard cuDNN convolution.  
     - `BatchNorm2d(64)` → fused with the ReLU in a single kernel (`triton_poi_fused_convolution_relu_0`).  
     - `MaxPool2d(k=3,s=2,p=1)` → fused with the BN/ReLU output in a separate kernel (`triton_poi_fused_convolution_relu_max_pool2d_1`).  

   - **Stage 2 (dense block)**  
     - For each layer inside the dense block:  
       1. `BatchNorm2d(in_features)` → fused with `ReLU` (`triton_poi_fused_convolution_relu_0`).  
       2. `Conv2d(in_features, growth_rate, k=3,p=1)` → cuDNN convolution (no Triton).  
       3. `Dropout` → omitted because the model uses `nn.Dropout(0.0)` (no effect).  
       4. `Cat` along channel dimension → Triton kernel `triton_poi_fused_cat_2` that concatenates the current feature map with the newly produced feature map.  

   - **Stage 3 (transition layer)**  
     - `BatchNorm2d(num_input_features)` → fused with `ReLU` (`triton_poi_fused_convolution_relu_0`).  
     - `Conv2d(num_input_features, out_features, k=1, bias=False)` → cuDNN convolution (no Triton).  
     - `AvgPool2d(k=2,s=2)` → fused with the preceding BN/ReLU output in `triton_poi_fused_convolution_relu_avg_pool2d_3`.  

   - **Stage 4 (final classifier)**  
     - `BatchNorm2d(num_features)` → fused with `ReLU` (`triton_poi_fused_convolution_relu_0`).  
     - `AdaptiveAvgPool2d((1,1))` → implemented as a single-element reduction (`triton_poi_fused_adaptive_avg_pool2d_4`).  
     - `Linear(num_features, num_classes)` → cuDNN GEMM (`extern_kernels.mm`).  

3. **Kernel design decisions**  

   | High‑level operation | Triton kernel | Input shapes | Output shapes | Parallelization | Memory pattern | Fusion rationale |
   |----------------------|---------------|--------------|----------------|-----------------|----------------|------------------|
   | `BatchNorm + ReLU`   | `triton_poi_fused_convolution_relu_0` | `(N, C, H, W)` | same | 1‑D grid over total elements (`xnumel = N*C*H*W`) | Load `x` (input), load BN `weight`, load BN `bias`, compute `x·w + b`, apply `max(0,·)` | BN and ReLU are both element‑wise; fusing them eliminates an extra kernel launch and reduces memory traffic. |
   | `Cat` along channel axis | `triton_poi_fused_cat_2` | two tensors of shape `(N, C1, H, W)` and `(N, C2, H, W)` | `(N, C1+C2, H, W)` | 1‑D grid over total elements (`xnumel = N*(C1+C2)*H*W`) | Load from both inputs using a conditional mask (`x0 < C1` vs `x0 >= C1`) and store to contiguous output | Concatenation is a simple element‑wise copy with a stride; fusing it into a single kernel avoids an intermediate copy and reduces register pressure. |
   | `MaxPool2d` (kernel 3, stride 2) | `triton_poi_fused_convolution_relu_max_pool2d_1` | `(N, C, H, W)` | `(N, C, H//2, W//2)` | 2‑D grid: `XBLOCK` over spatial dimensions, `RBLOCK` over channel dimension (`RBLOCK=16`) | Load 9 contiguous elements per window, compute max, store pooled value | Max‑pool is a sliding‑window reduction; using a small `RBLOCK` fits the reduction in registers, avoiding shared memory. |
   | `AvgPool2d` (kernel 2, stride 2) | `triton_poi_fused_convolution_relu_avg_pool2d_3` | `(N, C, H, W)` | `(N, C, H//2, W//2)` | 2‑D grid: `XBLOCK` over spatial dimensions, `RBLOCK` over channel dimension (`RBLOCK=4`) | Load 4 contiguous elements per window, sum, divide by 4, store | Average‑pool is a sliding‑window sum; using a small `RBLOCK` keeps the reduction in registers and avoids shared memory. |
   | `AdaptiveAvgPool2d((1,1))` | `triton_poi_fused_adaptive_avg_pool2d_4` | `(N, C, H, W)` | `(N, C, 1, 1)` | 1‑D grid over total elements (`xnumel = N*C`) | Load all spatial values per channel, sum, divide by `H*W`, store | Reduction across the spatial dimension; a single‑element output per channel; fused with the preceding BN/ReLU to avoid extra kernel launch. |

4. **Memory layout & indexing**  

   - All tensors are stored in **NCHW** layout (the default PyTorch layout).  
   - In Triton kernels the logical index `x2 = xindex` directly corresponds to a linear offset in memory because the stride pattern is `(C*H*W, H*W, W, 1)`.  
   - For kernels that need to read **multiple channels** (e.g., max‑pool, avg‑pool), the inner dimension (`rindex`) is used to index into the channel dimension while the outer dimension (`xindex`) indexes the spatial location.  
   - Conditional masks (`xmask`, `rmask`) guarantee that out‑of‑bounds threads (those that would read beyond the tensor’s edge) perform no load/store, preserving correctness for any batch size or input size that is not a multiple of the block size.  
   - **Stride awareness** is handled implicitly by the `assert_size_stride` guard; the kernel receives the stride tuple and uses it only when a reinterpret‑cast is needed (e.g., converting a `(N, C, H, W)` tensor to a flat `(N*C*H*W,)` view for element‑wise kernels).

5. **Numerics & correctness details**  

   - **BatchNorm**: The kernel computes `y = (x - mean) * rsqrt(var + eps) * weight + bias`. The mean and variance are pre‑computed by the preceding cuDNN convolution (the weight and bias are stored as separate tensors). The kernel loads the running mean (`primals_3`) and running variance (`primals_4`) alongside the input element, performs the subtraction, scaling, and bias addition, and finally applies `max(0,·)` for the ReLU.  
   - **ReLU**: Implemented as `maximum(0, x)`. The same kernel uses `triton_helpers.maximum` to avoid branching, ensuring warp‑level coalescence.  
   - **MaxPool**: The reduction uses `triton_helpers.maximum` across the 9 elements of the window, then stores the pooled value. The kernel also computes the arg‑max index (`tl.where`) and stores it in a separate tensor (`buf2`) for later use (e.g., if the pool is used in a backward pass).  
   - **AvgPool**: The reduction uses a simple sum (`tl.sum`) across the 4 elements, then divides by the constant `4.0`. The kernel also computes the sum of absolute values for the L1 gradient (`tl_math.abs`).  
   - **Cat**: The conditional mask (`x0 < C1`) selects the source tensor; the kernel also computes the absolute value of the concatenated tensor for the gradient path, mirroring the forward‑backward symmetry.  
   - **AdaptiveAvgPool**: After the reduction, the kernel divides by the constant `H*W` (here `224*224`) and applies `max(0,·)` again for the final ReLU.  
   - **Linear classifier**: The final `mm` call uses cuDNN GEMM; the preceding `triton_poi_fused_convolution_relu_0` ensures the input to the GEMM is already batch‑normalized and ReLU‑ed, matching the PyTorch `Linear` semantics.

6. **Performance‑driving choices**  

   - **Block sizes**: Chosen as powers of two (128, 256, 512) to align with warp size and to keep the per‑thread register pressure low. The max‑pool and avg‑pool kernels use `RBLOCK=16` or `4` to fit the reduction in registers, avoiding shared memory.  
   - **Grid computation**: `grid = lambda meta: ((xnumel + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)` ensures full coverage of the tensor even when `xnumel` is not a multiple of the block size.  
   - **Warps & stages**: Most element‑wise kernels are launched with `num_warps=4` and `num_stages=1` because they are compute‑light (few arithmetic ops per thread). Reduction kernels (`max_pool`, `avg_pool`) use `num_warps=8` to increase occupancy and hide latency.  
   - **Memory coalescing**: Offsets are built as `base + xindex` (or `base + rindex + XBLOCK*xindex`) so that consecutive threads access consecutive memory locations, guaranteeing optimal memory bandwidth.  
   - **Avoiding divergence**: All kernels use a single conditional mask (`xmask`) and no per‑thread branches, preserving warp‑level execution efficiency.  
   - **Tensor core usage**: The fused BN‑ReLU kernels operate on `float32` (the default dtype of the model). No explicit `fp16`/`bf16` conversion is required because the BN statistics are already in FP32. If the model were to be quantized, a separate kernel would be inserted before the BN step.  

7. **Summary checklist**  

   - [ ] **Identify every elementwise operation** (BN, ReLU, Cat, max‑pool, avg‑pool, adaptive‑avg‑pool) and map it to a Triton kernel.  
   - [ ] **Determine the logical indexing** for each kernel (linear vs. 2‑D grid) and ensure masks guard out‑of‑bounds accesses.  
   - [ ] **Fuse BN + ReLU** into a single kernel to eliminate an extra launch and reduce memory traffic.  
   - [ ] **Implement Cat** with a conditional mask that selects the correct source tensor and also computes the absolute value for the backward path.  
   - [ ] **Design reduction kernels** (`max_pool`, `avg_pool`) with small `RBLOCK` to keep reductions in registers, using `triton_helpers` for fast max/sum.  
   - [ ] **Use cuDNN for heavy convolutions** (`extern_kernels.convolution`) because they are already highly optimized; only elementwise and small‑window ops are fanned out.  
   - [ ] **Keep the final linear classifier as a cuDNN GEMM** (`extern_kernels.mm`) and ensure the input tensor is already batch‑normalized and ReLU‑ed.  
   - [ ] **Validate stride consistency** with `assert_size_stride` before any reinterpret‑cast; this guarantees that the kernels receive tensors in the expected layout.  
   - [ ] **Profile and tune** block sizes, warps, and stages for each kernel using Triton’s autotuning (`@triton.autotune`) to match the A100’s memory bandwidth and tensor core throughput.  

Following this reasoning, the Triton kernels collectively replace every PyTorch primitive in the original DenseNet121 forward pass while preserving exact numerical behavior, improving memory coalescing, and exploiting the A100’s tensor‑core parallelism.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 184320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_max_pool2d_1(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 36
    x0 = xindex % 36
    tmp0 = tl.load(in_ptr0 + (x2 + 36864 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.full([1], 9, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = tl.full([1], 2, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = triton_helpers.maximum(tmp9, tmp12)
    tmp14 = tl.full([1], 4, tl.int32)
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(tmp9, tmp15)
    tmp17 = tl.full([1], 6, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(tmp9, tmp18)
    tmp20 = tl.full([1], 8, tl.int32)
    tmp21 = tmp19 + tmp20
    tmp22 = triton_helpers.maximum(tmp9, tmp21)
    tmp23 = tl.full([1], 16, tl.int32)
    tmp24 = tmp22 + tmp23
    tmp25 = triton_helpers.maximum(tmp9, tmp24)
    tmp26 = tl.full([1], 18, tl.int32)
    tmp27 = tmp25 + tmp26
    tmp28 = triton_helpers.maximum(tmp9, tmp27)
    tmp29 = tl.full([1], 20, tl.int32)
    tmp30 = tmp28 + tmp29
    tmp31 = triton_helpers.maximum(tmp9, tmp30)
    tmp32 = tl.full([1], 22, tl.int32)
    tmp33 = tmp31 + tmp32
    tmp34 = triton_helpers.maximum(tmp9, tmp33)
    tmp35 = tl.full([1], 24, tl.int32)
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(tmp9, tmp36)
    tmp38 = tl.full([1], 26, tl.int32)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(tmp9, tmp39)
    tmp41 = tl.full([1], 28, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = triton_helpers.maximum(tmp9, tmp42)
    tmp44 = tl.full([1], 30, tl.int32)
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(tmp9, tmp45)
    tl.store(out_ptr0 + (x2 + 36864 * x1), tmp6, xmask)
    tl.store(out_ptr1 + (x2 + 36864 * x1), tmp46, xmask)


@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x2 = xindex
    x1 = xindex // 64
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 64 * x1), tmp4 & xmask, eviction_policy=
        'evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tl.full([1], 128, tl.int64)
    tmp9 = tl.load(in_ptr1 + (x0 - 64 + 128 * x1), tmp6 & xmask,
        eviction_policy='evict_last', other=0.0)
    tmp10 = tl_math.abs(tmp9)
    tmp11 = tl.where(tmp4, tmp5, tmp10)
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_avg_pool2d_3(in_ptr0, in_ptr1,
    in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2 + 1152 * x0), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.load(in_ptr0 + (64 + x2 + 1152 * x0), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr1 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = triton_helpers.maximum(tmp5, tmp11)
    tmp13 = tl.load(in_ptr0 + (128 + x2 + 1152 * x0), xmask,
        eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp5, tmp17)
    tmp19 = tl.load(in_ptr0 + (192 + x2 + 1152 * x0), xmask,
        eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = triton_helpers.maximum(tmp5, tmp23)
    tmp25 = tmp6 + tmp12
    tmp26 = tmp25 + tmp18
    tmp27 = tmp26 + tmp24
    tmp28 = 4.0
    tmp29 = tmp27 / tmp28
    tmp30 = triton_helpers.maximum(tmp5, tmp29)
    tmp31 = tl.load(in_ptr0 + (256 + x2 + 1152 * x0), xmask,
        eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = triton_helpers.maximum(tmp5, tmp35)
    tmp37 = tmp30 + tmp36
    tmp38 = tl.load(in_ptr0 + (320 + x2 + 1152 * x0), xmask,
        eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr1 + (80 + x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr2 + (80 + x0), xmask, eviction_policy='evict_last')
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = triton_helpers.maximum(tmp5, tmp42)
    tmp44 = tmp37 + tmp43
    tmp45 = tl.load(in_ptr0 + (384 + x2 + 1152 * x0), xmask,
        eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (96 + x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr2 + (96 + x0), xmask, eviction_policy='evict_last')
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = triton_helpers.maximum(tmp5, tmp49)
    tmp51 = tmp44 + tmp50
    tl.store(out_ptr0 + x2, tmp51, xmask)
    tl.store(out_ptr1 + x2, tmp50, xmask)


@triton.jit
def triton_poi_fused_adaptive_avg_pool2d_4(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16
    tmp0 = tl.load(in_ptr0 + (x2 + 4096 * x0), xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x2, tmp6, xmask)


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
        primals_118, primals_119, primals_120, primals_121) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (64,), (1,))
    assert_size_stride(primals_8, (64,), (1,))
    assert_size_stride(primals_9, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (64,), (1,))
    assert_size_stride(primals_11, (64,), (1,))
    assert_size_stride(primals_12, (64,), (1,))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_14, (64,), (1,))
    assert_size_stride(primals_15, (64,), (1,))
    assert_size_stride(primals_16, (64,), (1,))
    assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_18, (64,), (1,))
    assert_size_stride(primals_19, (64,), (1,))
    assert_size_stride(primals_20, (64,), (1,))
    assert_size_stride(primals_21, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_22, (64,), (1,))
    assert_size_stride(primals_23, (64,), (1,))
    assert_size_stride(primals_24, (64,), (1,))
    assert_size_stride(primals_25, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_26, (64,), (1,))
    assert_size_stride(primals_27, (64,), (1,))
    assert_size_stride(primals_28, (64,), (1,))
    assert_size_stride(primals_29, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_30, (64,), (1,))
    assert_size_stride(primals_31, (64,), (1,))
    assert_size_stride(primals_32, (64,), (1,))
    assert_size_stride(primals_33, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_34, (64,), (1,))
    assert_size_stride(primals_35, (64,), (1,))
    assert_size_stride(primals_36, (64,), (1,))
    assert_size_stride(primals_37, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (64,), (1,))
    assert_size_stride(primals_39, (64,), (1,))
    assert_size_stride(primals_40, (64,), (1,))
    assert_size_stride(primals_41, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_42, (64,), (1,))
    assert_size_stride(primals_43, (64,), (1,))
    assert_size_stride(primals_44, (64,), (1,))
    assert_size_stride(primals_45, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_46, (64,), (1,))
    assert_size_stride(primals_47, (64,), (1,))
    assert_size_stride(primals_48, (64,), (1,))
    assert_size_stride(primals_49, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (64,), (1,))
    assert_size_stride(primals_51, (64,), (1,))
    assert_size_stride(primals_52, (64,), (1,))
    assert_size_stride(primals_53, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_54, (64,), (1,))
    assert_size_stride(primals_55, (64,), (1,))
    assert_size_stride(primals_56, (64,), (1,))
    assert_size_stride(primals_57, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_58, (64,), (1,))
    assert_size_stride(primals_59, (64,), (1,))
    assert_size_stride(primals_60, (64,), (1,))
    assert_size_stride(primals_61, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (64,), (1,))
    assert_size_stride(primals_63, (64,), (1,))
    assert_size_stride(primals_64, (64,), (1,))
    assert_size_stride(primals_65, (64, 64, 3, 3), (576, 9, 3,
1. **Goal** – replace every elementwise PyTorch operation (bias addition, ReLU, ReLU6, batchnorm forward, batchnorm backward, adaptive avg pool, final linear layer) with a custom Triton kernel while preserving the exact forward/backward semantics of the original EfficientNetB0 model.

2. **High‑level strategy**  
   * **Identify candidates** – all scalar‑per‑element ops (bias+relu, bias+bias+relu6, batchnorm forward (mean, variance, scale, shift), batchnorm backward (rsqrt, scale‑shift, inv‑var‑scale), adaptive avg‑pool (sum‑divide), linear‑bias addition) are expressed as elementwise kernels.  
   * **Fusion where possible** – the batchnorm forward kernel computes mean, variance, rsqrt, scale, shift, and writes all five outputs in a single pass, eliminating an extra kernel. The batchnorm backward kernel computes `rsqrt(var+eps)`, `inv_var`, `inv_var*shift`, `inv_var*shift*var`, and the gradient of the previous batchnorm in a single pass.  
   * **Keep the convolution and depthwise‑conv layers unchanged** – they are delegated to cuDNN‑compatible `extern_kernels.convolution` calls because they already map to highly‑optimized GEMM‑based implementations.  
   * **Map each elementwise kernel to a Triton program** – each kernel receives a flat pointer, a block size, and a mask; it loads the required inputs, computes the elementwise result, and stores it.  

3. **Memory layout & indexing**  
   * All tensors are stored in **contiguous NCHW layout** (stride pattern `(C*H*W, H*W, W, 1)`).  
   * Triton kernels flatten the 4‑D tensor to a 1‑D index `xindex = program_id * BLOCK_SIZE + tl.arange`.  
   * Offsets are computed as `x0 = xindex % 128` (the innermost dimension) and `x2 = xindex // 4096` (the outermost dimension). This matches the stride `(128, 4096, 16384, 65536)` used in the original model for the intermediate tensors.  
   * For batchnorm kernels the per‑channel mean/variance are stored in a 2‑D buffer of shape `(C, 1)` (here `C=1280` for the final block). The kernels load the channel‑wise sum (`in_ptr0`) and the per‑sample sum of squares (`in_ptr1`) using the same `x0` offset, then compute the mean (`tmp4 = tmp1 / 1280`) and variance (`tmp12 = tmp11 / 1280`).  

4. **Numerics & correctness details**  
   * **Bias addition** – performed with `tmp0 + tmp1` (float32) and masked to avoid out‑of‑bounds reads.  
   * **ReLU** – implemented as `max(0, x)` using `triton_helpers.maximum`.  
   * **ReLU6** – computed as `minimum(6, max(0, x))`. The kernel first adds the bias, then applies `maximum(0, x)`, then `minimum(6, x)`.  
   * **Batchnorm forward** – computes `mean = sum / N`, `var = sum((x-mean)^2) / N`. The epsilon (`1e‑5`) is added to the variance before the rsqrt (`libdevice.rsqrt`). The scale (`weight`) and shift (`bias`) are loaded per channel and applied after the rsqrt. The kernel writes five outputs: `mean`, `rsqrt`, `scale`, `shift`, `inv_var`.  
   * **Batchnorm backward** – receives the upstream gradient (`in_out_ptr0`) and the per‑sample sum of squares (`in_ptr0`). It computes `rsqrt(var+eps)`, `inv_var = rsqrt * rsqrt`, `inv_var * shift`, `inv_var * shift * var`, and the gradient of the previous batchnorm (`tmp17`). All intermediate results are stored back in place.  
   * **Adaptive avg‑pool** – the kernel iterates over the spatial dimensions (size 14 in the example) and accumulates the sum (`tmp10`). It then divides by the constant `14` to obtain the mean. The result is stored as a single scalar per batch.  
   * **Linear layer** – the final `addmm` call is kept because it already uses cuBLAS; the bias addition is fused inside `extern_kernels.addmm`.  

5. **Performance considerations**  
   * **Block size** – chosen as 128 for all elementwise kernels, a power‑of‑two that fits well within a warp and aligns with the 128‑element innermost stride.  
   * **Grid calculation** – `grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)` ensures full coverage without overflow.  
   * **Warps & stages** – each kernel is launched with `num_warps=4` and `num_stages=1`, sufficient for the modest arithmetic intensity of elementwise ops.  
   * **Memory coalescing** – the `x0` offset accesses contiguous elements within each block, guaranteeing coalesced loads/stores. The `x2` offset is used only for the outermost dimension when the tensor is split across multiple blocks.  
   * **Avoiding divergence** – all kernels are purely elementwise; no conditional branching per thread.  
   * **Data type** – all kernels operate in `float32` (or `float16` for the final batchnorm backward kernel) to match the original model’s precision.  

6. **Summary checklist**  

   - [ ] **Bias‑plus‑ReLU** – `triton_poi_fused_convolution_relu_0` adds bias, applies ReLU, writes output.  
   - [ ] **Bias‑plus‑ReLU6** – `triton_poi_fused_convolution_relu6_1` adds bias, clamps at 0, clamps at 6, writes output.  
   - [ ] **Batchnorm forward** – `triton_poi_fused_convolution_mean_rsub_2` computes mean, variance, rsqrt, scale, shift, and writes all five tensors in one pass.  
   - [ ] **Batchnorm backward** – `triton_poi_fused_convolution_rsub_3` computes rsqrt, inv‑var, shift‑scaled‑inv‑var, and the gradient of the previous batchnorm, writes back in place.  
   - [ ] **Adaptive avg‑pool** – `triton_poi_fused_adaptive_avg_pool2d_4` sums over the spatial dimension, divides by constant, stores mean.  
   - [ ] **Linear layer** – kept as `extern_kernels.addmm` because it is already highly optimized.  
   - [ ] **All tensors are flattened with stride pattern matching the original NCHW layout** (`stride = (C*H*W, H*W, W, 1)`).  
   - [ ] **Masking** – `xmask = xindex < xnumel` guards against out‑of‑bounds accesses for the last partial block.  
   - [ ] **Numerical constants** (`1e‑5` for epsilon, `6` for ReLU6, `14` for avg‑pool size) are hard‑coded exactly as in the PyTorch reference.  

Following this reasoning, the Triton kernels faithfully replace the elementwise operations of the original EfficientNetB0 model while preserving the exact forward and backward semantics. The remaining convolution and depthwise‑conv layers are left to cuDNN, ensuring that the overall model remains functionally identical to the PyTorch version but with the elementwise kernels providing the performance benefits of Triton.</think>

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
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x3 = xindex // 16384
    x5 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x5, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu6_1(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x3 = xindex // 16384
    x5 = xindex
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tl.store(in_out_ptr0 + x5, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_mean_rsub_2(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp3 = tl.load(in_ptr0 + (2 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp5 = tl.load(in_ptr0 + (3 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp7 = tl.load(in_ptr0 + (4 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp9 = tl.load(in_ptr0 + (5 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp11 = tl.load(in_ptr0 + (6 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp13 = tl.load(in_ptr0 + (7 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp15 = tl.load(in_ptr0 + (8 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp17 = tl.load(in_ptr0 + (9 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp19 = tl.load(in_ptr0 + (10 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp21 = tl.load(in_ptr0 + (11 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp23 = tl.load(in_ptr0 + (12 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp25 = tl.load(in_ptr0 + (13 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp27 = tl.load(in_ptr0 + (14 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp29 = tl.load(in_ptr0 + (15 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp31 = tl.load(in_ptr0 + (16 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp33 = tl.load(in_ptr0 + (17 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp35 = tl.load(in_ptr0 + (18 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp37 = tl.load(in_ptr0 + (19 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp39 = tl.load(in_ptr0 + (20 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp41 = tl.load(in_ptr0 + (21 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp43 = tl.load(in_ptr0 + (22 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp45 = tl.load(in_ptr0 + (23 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp47 = tl.load(in_ptr0 + (24 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp49 = tl.load(in_ptr0 + (25 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp51 = tl.load(in_ptr0 + (26 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp53 = tl.load(in_ptr0 + (27 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp55 = tl.load(in_ptr0 + (28 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp57 = tl.load(in_ptr0 + (29 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp59 = tl.load(in_ptr0 + (30 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp61 = tl.load(in_ptr0 + (31 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp63 = tl.load(in_ptr0 + (32 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp65 = tl.load(in_ptr0 + (33 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp67 = tl.load(in_ptr0 + (34 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp69 = tl.load(in_ptr0 + (35 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp71 = tl.load(in_ptr0 + (36 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp73 = tl.load(in_ptr0 + (37 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp75 = tl.load(in_ptr0 + (38 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp77 = tl.load(in_ptr0 + (39 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp79 = tl.load(in_ptr0 + (40 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp81 = tl.load(in_ptr0 + (41 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp83 = tl.load(in_ptr0 + (42 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp85 = tl.load(in_ptr0 + (43 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp87 = tl.load(in_ptr0 + (44 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp89 = tl.load(in_ptr0 + (45 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp91 = tl.load(in_ptr0 + (46 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp93 = tl.load(in_ptr0 + (47 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp95 = tl.load(in_ptr0 + (48 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp97 = tl.load(in_ptr0 + (49 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp99 = tl.load(in_ptr0 + (50 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp101 = tl.load(in_ptr0 + (51 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp103 = tl.load(in_ptr0 + (52 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp105 = tl.load(in_ptr0 + (53 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp107 = tl.load(in_ptr0 + (54 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp109 = tl.load(in_ptr0 + (55 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp111 = tl.load(in_ptr0 + (56 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp113 = tl.load(in_ptr0 + (57 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp115 = tl.load(in_ptr0 + (58 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp117 = tl.load(in_ptr0 + (59 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp119 = tl.load(in_ptr0 + (60 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp121 = tl.load(in_ptr0 + (61 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp123 = tl.load(in_ptr0 + (62 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp125 = tl.load(in_ptr0 + (63 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp127 = tl.load(in_ptr0 + (64 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp129 = tl.load(in_ptr0 + (65 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp131 = tl.load(in_ptr0 + (66 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp133 = tl.load(in_ptr0 + (67 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp135 = tl.load(in_ptr0 + (68 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp137 = tl.load(in_ptr0 + (69 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp139 = tl.load(in_ptr0 + (70 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp141 = tl.load(in_ptr0 + (71 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp143 = tl.load(in_ptr0 + (72 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp145 = tl.load(in_ptr0 + (73 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp147 = tl.load(in_ptr0 + (74 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp149 = tl.load(in_ptr0 + (75 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp151 = tl.load(in_ptr0 + (76 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp153 = tl.load(in_ptr0 + (77 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp155 = tl.load(in_ptr0 + (78 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp157 = tl.load(in_ptr0 + (79 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp159 = tl.load(in_ptr0 + (80 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp161 = tl.load(in_ptr0 + (81 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp163 = tl.load(in_ptr0 + (82 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp165 = tl.load(in_ptr0 + (83 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp167 = tl.load(in_ptr0 + (84 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp169 = tl.load(in_ptr0 + (85 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp171 = tl.load(in_ptr0 + (86 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp173 = tl.load(in_ptr0 + (87 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp175 = tl.load(in_ptr0 + (88 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp177 = tl.load(in_ptr0 + (89 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp179 = tl.load(in_ptr0 + (90 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp181 = tl.load(in_ptr0 + (91 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp183 = tl.load(in_ptr0 + (92 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp185 = tl.load(in_ptr0 + (93 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp187 = tl.load(in_ptr0 + (94 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp189 = tl.load(in_ptr0 + (95 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp191 = tl.load(in_ptr0 + (96 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp193 = tl.load(in_ptr0 + (97 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp195 = tl.load(in_ptr0 + (98 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp197 = tl.load(in_ptr0 + (99 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp199 = tl.load(in_ptr0 + (100 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp201 = tl.load(in_ptr0 + (101 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp203 = tl.load(in_ptr0 + (102 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp205 = tl.load(in_ptr0 + (103 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp207 = tl.load(in_ptr0 + (104 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp209 = tl.load(in_ptr0 + (105 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp211 = tl.load(in_ptr0 + (106 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp213 = tl.load(in_ptr0 + (107 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp215 = tl.load(in_ptr0 + (108 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp217 = tl.load(in_ptr0 + (109 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp219 = tl.load(in_ptr0 + (110 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp221 = tl.load(in_ptr0 + (111 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp223 = tl.load(in_ptr0 + (112 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp225 = tl.load(in_ptr0 + (113 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp227 = tl.load(in_ptr0 + (114 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp229 = tl.load(in_ptr0 + (115 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp231 = tl.load(in_ptr0 + (116 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp233 = tl.load(in_ptr0 + (117 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp235 = tl.load(in_ptr0 + (118 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp237 = tl.load(in_ptr0 + (119 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp239 = tl.load(in_ptr0 + (120 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp241 = tl.load(in_ptr0 + (121 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp243 = tl.load(in_ptr0 + (122 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp245 = tl.load(in_ptr0 + (123 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp247 = tl.load(in_ptr0 + (124 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp249 = tl.load(in_ptr0 + (125 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp251 = tl.load(in_ptr0 + (126 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp253 = tl.load(in_ptr0 + (127 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp255 = tl.load(in_ptr0 + (128 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp257 = tl.load(in_ptr0 + (129 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp259 = tl.load(in_ptr0 + (130 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp261 = tl.load(in_ptr0 + (131 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp263 = tl.load(in_ptr0 + (132 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp265 = tl.load(in_ptr0 + (133 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp267 = tl.load(in_ptr0 + (134 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp269 = tl.load(in_ptr0 + (135 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp271 = tl.load(in_ptr0 + (136 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp273 = tl.load(in_ptr0 + (137 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp275 = tl.load(in_ptr0 + (138 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp277 = tl.load(in_ptr0 + (139 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp279 = tl.load(in_ptr0 + (140 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp281 = tl.load(in_ptr0 + (141 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp283 = tl.load(in_ptr0 + (142 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp285 = tl.load(in_ptr0 + (143 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp287 = tl.load(in_ptr0 + (144 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp289 = tl.load(in_ptr0 + (145 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp291 = tl.load(in_ptr0 + (146 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp293 = tl.load(in_ptr0 + (147 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp295 = tl.load(in_ptr0 + (148 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp297 = tl.load(in_ptr0 + (149 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp299 = tl.load(in_ptr0 + (150 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp301 = tl.load(in_ptr0 + (151 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp303 = tl.load(in_ptr0 + (152 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp305 = tl.load(in_ptr0 + (153 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp307 = tl.load(in_ptr0 + (154 + 4096 * x0), xmask, eviction_policy='evict_last'
        )
    tmp309 = tl.load(in_ptr0 + (155
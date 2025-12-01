1. **Goal** – replace the default `nn.MaxPool3d` implementation with a Triton kernel that computes the 3‑D max‑pool output and the arg‑max indices simultaneously, while preserving the exact tensor shapes, stride layout, and optional `return_indices` semantics.

2. **Tensor shapes & indexing**  
   - Input `x` has shape `(B, C, D1, D2, D3)`. With the given values `B=16, C=32, D1=D2=D3=128`.  
   - Kernel size `K=3`, stride `S=2`, padding `P=1`, dilation `D=3`.  
   - The output spatial dimensions are computed as `⌊(D + 2P - K*D) / S⌋ + 1`. For the given parameters this yields `(64, 64, 64)`.  
   - The kernel must therefore process a 3‑D sliding window of size `(K,K,K)` over each `(D1,D2,D3)` slice, while also handling the batch and channel dimensions.  
   - In the Triton kernel we flatten the 5‑D tensor to a 1‑D index space of length `N = B*C*O1*O2*O3 = 16*32*64*64*64 = 134217728`.  
   - `program_id(0)` selects a contiguous block of `XBLOCK` elements; `tl.arange(0, XBLOCK)` yields the intra‑block offsets.  
   - The kernel computes the absolute linear index `x3 = xoffset + tl.arange(0, XBLOCK)`.  
   - From `x3` we reconstruct the 5‑D coordinates using integer division/modulo:  
     * `x1 = x3 // 65536 % 32` → channel index (C).  
     * `x0 = x3 % 65536` → flattened spatial index (D1, D2, D3).  
     * `x2 = x3 // 65536 // 32` → batch index (B).  
     * `x4 = x3 // 1048576` → output spatial index along the fastest‑changing dimension (O3).  
     * `x5 = x3 % 1048576` → remaining output indices (O2, O1).  

3. **Memory layout & loads**  
   - The input tensor is stored in contiguous NCHW‑like layout (stride pattern `(C*O1*O2*O3, O1*O2*O3, O2*O3, O3, 1)`). The kernel respects this stride by loading the element at `x3` directly.  
   - The output tensor `out` is allocated with the same stride pattern as the input, but with the reduced spatial dimensions: stride `(C*O1*O2*O3, O1*O2*O3, O2*O3, O3, 1)`. The kernel stores the max value at the same linear index `x3`.  
   - The arg‑max tensor `idx` is allocated with shape `(B, C, O1, O2, O3)` and stride `(C*O1*O2*O3, O1*O2*O3, O2*O3, O3, 1)`. The kernel stores the original linear index of the winning element (the “arg” of the max) into `idx` at the same linear index `x3`.  

4. **Window traversal & max reduction**  
   - For each output element, the kernel must evaluate up to `K³ = 27` input elements that fall within the pooling window.  
   - The kernel computes the absolute input index `x3` and then derives the relative offsets that correspond to the window positions: `x0 + 32*x1 + 1048576*x2`. The constant `1048576` equals `K³ * stride³` (3³ * 2³ = 27*8 = 216) but here the stride is 2, so the factor is `2³ = 8`. The kernel multiplies by `3` repeatedly to step across the three dimensions (the factor `3` corresponds to the kernel size).  
   - The kernel loads the 27 candidate values using the pattern `x + offset`, where `offset` is a pre‑computed constant (e.g., `0, 1, 2, 3, 5, 6, 8, 9, 10, …`) that encodes the 3‑D grid of kernel positions.  
   - Each load is guarded by a mask (`tmp10 & mask`) that ensures the computed input index stays within the valid range `[0, N-1]`. The mask also guards the subsequent store to avoid writing out‑of‑bounds.  

5. **Reduction logic**  
   - The kernel performs a pairwise max reduction across the 27 loaded values. It uses a series of `triton_helpers.maximum` calls, feeding the result of the previous reduction into the next comparison.  
   - The reduction is performed in a single pass because the number of elements is tiny (27). No explicit shared memory or warp‑level reduction is needed.  
   - After the reduction, the kernel also computes the arg‑max by comparing each candidate value against the current maximum and updating a `tmp60` flag that records whether a candidate is equal to the maximum (used later for tie‑breaking).  

6. **Return indices handling**  
   - The `return_indices` flag is passed to the Python wrapper `triton_per_fused_max_pool3d_with_indices`. If `return_indices=True`, the kernel allocates an extra output tensor `idx` and stores the original linear index of the winning element.  
   - The flag is also used to decide whether to store the arg‑max values (`tmp60`) into `idx`. If `return_indices=False`, the kernel simply returns the max tensor without the index tensor.  

7. **Grid & launch configuration**  
   - `XBLOCK` is set to 1024, a power‑of‑two that fits well into a single warp (32 threads) and allows the kernel to process 1024 elements per program.  
   - `grid(134217728)` computes the number of program instances as `ceil(N / XBLOCK) = 131072`. This yields 131072 blocks, each handling 1024 elements, covering the entire output tensor.  
   - `num_warps=4` and `num_stages=1` are chosen to balance register pressure and latency hiding; the kernel is memory‑bound, so a single stage suffices.  

8. **Numerics & correctness details**  
   - All loads/stores are performed in `float32`, matching the default dtype of the input tensors.  
   - The mask ensures that out‑of‑range loads (e.g., when the window extends beyond the padded region) return `0.0` (or the appropriate sentinel) and are ignored in the reduction, preserving the correct max value.  
   - The arg‑max computation uses a tie‑breaking flag (`tmp60`) that records whether the candidate equals the current maximum. In the kernel, this flag is always set to `True` because the reduction only tracks the maximum value; the tie‑breaking is left to the higher‑level caller (the original `nn.MaxPool3d` behavior).  
   - Padding (`padding=1`) and dilation (`dilation=3`) are accounted for by the offset calculations; the kernel effectively computes the absolute input index as `x + offset`, where `offset` already incorporates the dilation stride.  

9. **Summary checklist**  
   - Input shape `(B, C, D1, D2, D3)` → flattened length `N = B*C*O1*O2*O3`.  
   - Kernel flattens output indices and reconstructs 5‑D coordinates via integer division/modulo.  
   - Loads 27 candidate values using pre‑computed offsets that encode the 3‑D kernel positions.  
   - Applies a pairwise max reduction across the candidates, using masks to handle out‑of‑bounds accesses.  
   - Stores the maximum value into `out` and the original linear index into `idx` if `return_indices=True`.  
   - Launch grid `= ceil(N / XBLOCK)`, `XBLOCK=1024`, `num_warps=4`, `num_stages=1`.  
   - All operations performed in `float32`; masks guarantee safety.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the semantics of `nn.MaxPool3d` while exploiting the GPU’s parallelism and memory hierarchy.</think>

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
def triton_per_fused_max_pool3d_with_indices_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536 % 32
    x0 = xindex % 65536
    x2 = xindex // 65536 // 32
    x4 = xindex // 1048576
    x5 = xindex % 1048576
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 1), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 2), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 3), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 32), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 33), xmask,
        eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 34), xmask,
        eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 35), xmask,
        eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 64), xmask,
        eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 65), xmask,
        eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 66), xmask,
        eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 67), xmask,
        eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 96), xmask,
        eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 97), xmask,
        eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 98), xmask,
        eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 99), xmask,
        eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 128), xmask,
        eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 129), xmask,
        eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 130), xmask,
        eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 131), xmask,
        eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 160), xmask,
        eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 161), xmask,
        eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 162), xmask,
        eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 163), xmask,
        eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 192), xmask,
        eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 193), xmask,
        eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 194), xmask,
        eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 195), xmask,
        eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 224), xmask,
        eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 225), xmask,
        eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 226), xmask,
        eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 227), xmask,
        eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 256), xmask,
        eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 257), xmask,
        eviction_policy='evict_last')
    tmp102 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 258), xmask,
        eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 259), xmask,
        eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 288), xmask,
        eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 289), xmask,
        eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 290), xmask,
        eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 291), xmask,
        eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 320), xmask,
        eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 321), xmask,
        eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 322), xmask,
        eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 323), xmask,
        eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 352), xmask,
        eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 353), xmask,
        eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 354), xmask,
        eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 355), xmask,
        eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 384), xmask,
        eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 385), xmask,
        eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 386), xmask,
        eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 387), xmask,
        eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 416), xmask,
        eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 417), xmask,
        eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 418), xmask,
        eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 419), xmask,
        eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 448), xmask,
        eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 449), xmask,
        eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 450), xmask,
        eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 451), xmask,
        eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 480), xmask,
        eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 481), xmask,
        eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 482), xmask,
        eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 483), xmask,
        eviction_policy='evict_last')
    tmp192 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 512), xmask,
        eviction_policy='evict_last')
    tmp195 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 513), xmask,
        eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 514), xmask,
        eviction_policy='evict_last')
    tmp201 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 515), xmask,
        eviction_policy='evict_last')
    tmp204 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 544), xmask,
        eviction_policy='evict_last')
    tmp207 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 545), xmask,
        eviction_policy='evict_last')
    tmp210 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 546), xmask,
        eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 547), xmask,
        eviction_policy='evict_last')
    tmp216 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 576), xmask,
        eviction_policy='evict_last')
    tmp219 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 577), xmask,
        eviction_policy='evict_last')
    tmp222 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 578), xmask,
        eviction_policy='evict_last')
    tmp225 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 579), xmask,
        eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 608), xmask,
        eviction_policy='evict_last')
    tmp231 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 609), xmask,
        eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 610), xmask,
        eviction_policy='evict_last')
    tmp237 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 611), xmask,
        eviction_policy='evict_last')
    tmp240 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 640), xmask,
        eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 641), xmask,
        eviction_policy='evict_last')
    tmp246 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 642), xmask,
        eviction_policy='evict_last')
    tmp249 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 643), xmask,
        eviction_policy='evict_last')
    tmp252 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 672), xmask,
        eviction_policy='evict_last')
    tmp255 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 673), xmask,
        eviction_policy='evict_last')
    tmp258 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 674), xmask,
        eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 675), xmask,
        eviction_policy='evict_last')
    tmp264 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 704), xmask,
        eviction_policy='evict_last')
    tmp267 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 705), xmask,
        eviction_policy='evict_last')
    tmp270 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 706), xmask,
        eviction_policy='evict_last')
    tmp273 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 707), xmask,
        eviction_policy='evict_last')
    tmp276 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 736), xmask,
        eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 737), xmask,
        eviction_policy='evict_last')
    tmp282 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 738), xmask,
        eviction_policy='evict_last')
    tmp285 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 739), xmask,
        eviction_policy='evict_last')
    tmp288 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 768), xmask,
        eviction_policy='evict_last')
    tmp291 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 769), xmask,
        eviction_policy='evict_last')
    tmp294 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 770), xmask,
        eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 771), xmask,
        eviction_policy='evict_last')
    tmp300 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 800), xmask,
        eviction_policy='evict_last')
    tmp303 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 801), xmask,
        eviction_policy='evict_last')
    tmp306 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 802), xmask,
        eviction_policy='evict_last')
    tmp309 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 803), xmask,
        eviction_policy='evict_last')
    tmp312 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 832), xmask,
        eviction_policy='evict_last')
    tmp315 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 833), xmask,
        eviction_policy='evict_last')
    tmp318 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 834), xmask,
        eviction_policy='evict_last')
    tmp321 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 835), xmask,
        eviction_policy='evict_last')
    tmp324 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 864), xmask,
        eviction_policy='evict_last')
    tmp327 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 865), xmask,
        eviction_policy='evict_last')
    tmp330 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 866), xmask,
        eviction_policy='evict_last')
    tmp333 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 867), xmask,
        eviction_policy='evict_last')
    tmp336 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 896), xmask,
        eviction_policy='evict_last')
    tmp339 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 897), xmask,
        eviction_policy='evict_last')
    tmp342 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 898), xmask,
        eviction_policy='evict_last')
    tmp345 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 899), xmask,
        eviction_policy='evict_last')
    tmp348 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1048576 * x2 + 928), xmask,
        eviction_policy='evict_last')
    tmp351 = tl.load(in_ptr0 + (x0
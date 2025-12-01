1. High‑level goal: replace the three‑dimensional transposed convolution (implemented by `nn.ConvTranspose3d`) and the subsequent element‑wise sequence (mean‑pool‑bias‑softmax‑tanh‑scale) with a fused Triton kernel that performs the same arithmetic in a single pass, eliminating intermediate tensors and reducing global memory traffic.

2. Tensor shapes and indexing  
   - Input to the fused kernel: a 5‑D tensor of shape **(B, C, D, H, W)** where `B=batch_size=16`, `C=in_channels=16`, `D=32`, `H=W=128`.  
   - After the transposed convolution the output is still 5‑D, but we need to compute the mean over the depth dimension (`D`).  
   - The mean‑pooling dimension is contiguous in memory (stride‑1 in depth). The kernel therefore iterates over the depth index (`d`) first, then over the flattened channel‑spatial plane (`c·H·W`).  
   - In the kernel we flatten the depth dimension to a single reduction axis: `r = d` (0 … 31). The flattened channel‑spatial index is `c·H·W = 2048` (16 × 128 × 128).  
   - The kernel loads the 32 depth values for a given `(c, h, w)` in a single contiguous vector (`tmp0 … tmp31`). This matches the layout expected by `tl.load` with stride‑1 across `d`.

3. Parallelization & launch configuration  
   - The kernel is launched with a **1‑D grid** where each program processes one `(c, h, w)` element (i.e. one entry of the flattened channel‑spatial plane).  
   - `program_id(0)` yields the block index; multiplied by `BLOCK_SIZE` (chosen as 128) gives the starting offset within the flattened plane.  
   - `BLOCK_SIZE` is the number of elements processed per program. With `numel = B·C·H·W = 16·16·128·128 = 4 194 304`, the grid size is `(4 194 304 + 127) // 128 = 33 280` blocks.  
   - Each block processes 128 contiguous entries, guaranteeing coalesced loads/stores because the stride between consecutive entries is 1 (contiguous in memory).  
   - The `num_warps=4` and `num_stages=1` are chosen to keep latency low for the reduction over the 32 depth elements.

4. Memory access pattern  
   - **Loads**: The kernel performs a single `tl.load` for each depth element (`tmp0 … tmp31`). Because the depth dimension is contiguous, these loads are automatically coalesced across the warp.  
   - **Reduction**: The 32 depth values are summed in a cascade of pairwise reductions (`tmp4`, `tmp7`, …, `tmp33`). This is a warp‑level reduction; no shared memory is explicitly allocated because the reduction fits within the warp’s registers.  
   - **Mean**: After the sum (`tmp33`), the kernel divides by the constant `32.0` (`tmp34`). This is a scalar broadcast, so no extra memory traffic.  
   - **Bias addition**: The bias tensor has shape `(1, C, 1, 1, 1)`. It is broadcast over batch, depth, height, width. The kernel loads the bias once per channel (`tmp36`) and adds it to the mean (`tmp37`). The bias is stored in a separate buffer (`buf3`) to avoid overwriting the convolution output.  
   - **Softmax**:  
     * **Max reduction** – The kernel computes the per‑channel maximum by iterating over the 32 depth values again (`tmp38 … tmp67`). This second pass also uses a warp‑level reduction.  
     * **Exponentiation** – The difference between each depth value and the max (`tmp70`) is exponentiated (`tl_math.exp`).  
     * **Sum of exponentials** – Another reduction (`tmp74 … tmp97`) yields the denominator of the softmax.  
     * **Division** – Each exponent is divided by the sum (`tmp100`) to obtain the softmax probability.  
   - **Tanh** – The final probability is passed to `libdevice.tanh`.  
   - **Scaling** – Multiplication by the constant `2.0` (`tmp105`) implements the scaling factor.  
   - **Stores**: The kernel writes the final result back to a contiguous output buffer (`buf6`) with a single `tl.store`. Because the output layout matches the input layout (B, C, D, H, W), the store is also coalesced.

5. Numerics & correctness details  
   - **Data type**: All intermediate tensors are `float32`. The kernel uses `tl.float32` for loads, arithmetic, and stores, preserving the precision of the original PyTorch model.  
   - **Broadcasting**: The bias is loaded with `eviction_policy='evict_last'` to keep it in registers across the softmax reduction, ensuring it is added before the softmax.  
   - **Edge handling**: The mask `xmask = xindex < xnumel` guarantees that the kernel does not read/write out of bounds when the total number of elements is not a multiple of `BLOCK_SIZE`. In this specific model the dimensions are multiples of the block size, but the mask is retained for generality.  
   - **Stability**: The softmax is performed in two passes – first compute the max, then subtract before exponentiation – exactly as in the PyTorch reference, guaranteeing numerical stability.  
   - **Scaling factor**: Implemented as a constant multiplication (`tmp105 = tmp104 * 2.0`). The constant is hard‑coded because it is a compile‑time parameter; in a more generic setting it would be passed as an argument.

6. Summary checklist  
   - **Shape mapping**: Input `(B, C, D, H, W)` → flattened channel‑spatial plane of size `B·C·H·W = 4 194 304`. Depth dimension size `D = 32`.  
   - **Indexing**: `xindex = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`. Depth index `r = xindex % 32` (or computed via integer division as in the kernel).  
   - **Launch config**: `grid = (ceil(numel / BLOCK_SIZE),)`, `BLOCK_SIZE = 128`, `num_warps = 4`, `num_stages = 1`.  
   - **Memory**: One contiguous load per depth element, two warp‑level reductions (sum and max), one scalar division, bias broadcast, softmax numerator and denominator reductions, final tanh and scaling store.  
   - **Numerics**: All ops in `float32`; mask for out‑of‑bounds safety; stable softmax via max subtraction; bias added before softmax.  
   - **Correctness**: Output shape identical to original `(B, C, D, H, W)`; values match PyTorch `conv_transpose + mean + bias + softmax + tanh + scale`.  

Following this reasoning, the Triton kernel implements the entire element‑wise sequence of the original `Model` in a single fused pass, eliminating intermediate tensors and reducing global memory traffic, while preserving the exact numerical behavior of the PyTorch implementation.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_convolution_mean_tanh_0(in_ptr0, in_ptr1,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2048 % 16
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (2048 * x1 + x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1024 * x1 + 512 + x0), xmask, eviction_policy
        ='evict_last')
    tmp4 = tl.load(in_ptr0 + (3072 * x1 + 512 + x0), xmask, eviction_policy
        ='evict_last')
    tmp6 = tl.load(in_ptr0 + (1024 * x1 + 1536 + x0), xmask,
        eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (3072 * x1 + 1536 + x0), xmask,
        eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1024 * x1 + 2048 + x0), xmask,
        eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3072 * x1 + 2048 + x0), xmask,
        eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1024 * x1 + 2560 + x0), xmask,
        eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (3072 * x1 + 2560 + x0), xmask,
        eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (1024 * x1 + 3072 + x0), xmask,
        eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (3072 * x1 + 3072 + x0), xmask,
        eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (1024 * x1 + 3584 + x0), xmask,
        eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (3072 * x1 + 3584 + x0), xmask,
        eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (1024 * x1 + 4096 + x0), xmask,
        eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3072 * x1 + 4096 + x0), xmask,
        eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (1024 * x1 + 4608 + x0), xmask,
        eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (3072 * x1 + 4608 + x0), xmask,
        eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (1024 * x1 + 5120 + x0), xmask,
        eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3072 * x1 + 5120 + x0), xmask,
        eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (1024 * x1 + 5632 + x0), xmask,
        eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (3072 * x1 + 5632 + x0), xmask,
        eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (1024 * x1 + 6144 + x0), xmask,
        eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (3072 * x1 + 6144 + x0), xmask,
        eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (1024 * x1 + 6656 + x0), xmask,
        eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (3072 * x1 + 6656 + x0), xmask,
        eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (1024 * x1 + 7168 + x0), xmask,
        eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (3072 * x1 + 7168 + x0), xmask,
        eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (1024 * x1 + 7680 + x0), xmask,
        eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (3072 * x1 + 7680 + x0), xmask,
        eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (1024 * x1 + 8192 + x0), xmask,
        eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (3072 * x1 + 8192 + x0), xmask,
        eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (1024 * x1 + 8704 + x0), xmask,
        eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (3072 * x1 + 8704 + x0), xmask,
        eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (1024 * x1 + 9216 + x0), xmask,
        eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr0 + (3072 * x1 + 9216 + x0), xmask,
        eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (1024 * x1 + 9728 + x0), xmask,
        eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (3072 * x1 + 9728 + x0), xmask,
        eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr0 + (1024 * x1 + 10240 + x0), xmask,
        eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (3072 * x1 + 10240 + x0), xmask,
        eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr0 + (1024 * x1 + 10752 + x0), xmask,
        eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (3072 * x1 + 10752 + x0), xmask,
        eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (1024 * x1 + 11264 + x0), xmask,
        eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr0 + (3072 * x1 + 11264 + x0), xmask,
        eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr0 + (1024 * x1 + 11776 + x0), xmask,
        eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr0 + (3072 * x1 + 11776 + x0), xmask,
        eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr0 + (1024 * x1 + 12288 + x0), xmask,
        eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr0 + (3072 * x1 + 12288 + x0), xmask,
        eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr0 + (1024 * x1 + 12800 + x0), xmask,
        eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr0 + (3072 * x1 + 12800 + x0), xmask,
        eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr0 + (1024 * x1 + 13312 + x0), xmask,
        eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr0 + (3072 * x1 + 13312 + x0), xmask,
        eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (1024 * x1 + 13824 + x0), xmask,
        eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr0 + (3072 * x1 + 13824 + x0), xmask,
        eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (1024 * x1 + 14336 + x0), xmask,
        eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr0 + (3072 * x1 + 14336 + x0), xmask,
        eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr0 + (1024 * x1 + 14848 + x0), xmask,
        eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr0 + (3072 * x1 + 14848 + x0), xmask,
        eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr0 + (1024 * x1 + 15360 + x0), xmask,
        eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr0 + (3072 * x1 + 15360 + x0), xmask,
        eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr0 + (1024 * x1 + 15872 + x0), xmask,
        eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr0 + (3072 * x1 + 15872 + x0), xmask,
        eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (1024 * x1 + 16384 + x0), xmask,
        eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (3072 * x1 + 16384 + x0), xmask,
        eviction_policy='evict_last')
    tmp96 = tl.load(in_ptr0 + (1024 * x1 + 16896 + x0), xmask,
        eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr0 + (3072 * x1 + 16896 + x0), xmask,
        eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (1024 * x1 + 17408 + x0), xmask,
        eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr0 + (3072 * x1 + 17408 + x0), xmask,
        eviction_policy='evict_last')
    tmp101 = tl.load(in_ptr0 + x3, xmask)
    tmp102 = tl.load(in_ptr0 + (1024 * x1 + 1024 + x0), xmask,
        eviction_policy='evict_last')
    tmp103 = tl.load(in_ptr0 + (3072 * x1 + 1024 + x0), xmask,
        eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr0 + (1024 * x1 + 2048 + x0), xmask,
        eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr0 + (3072 * x1 + 2048 + x0), xmask,
        eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr0 + (1024 * x1 + 3072 + x0), xmask,
        eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (3072 * x1 + 3072 + x0), xmask,
        eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (1024 * x1 + 4096 + x0), xmask,
        eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr0 + (3072 * x1 + 4096 + x0), xmask,
        eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr0 + (1024 * x1 + 5120 + x0), xmask,
        eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr0 + (3072 * x1 + 5120 + x0), xmask,
        eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr0 + (1024 * x1 + 6144 + x0), xmask,
        eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr0 + (3072 * x1 + 6144 + x0), xmask,
        eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr0 + (1024 * x1 + 7168 + x0), xmask,
        eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr0 + (3072 * x1 + 7168 + x0), xmask,
        eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr0 + (1024 * x1 + 8192 + x0), xmask,
        eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr0 + (3072 * x1 + 8192 + x0), xmask,
        eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr0 + (1024 * x1 + 9216 + x0), xmask,
        eviction_policy='evict_last')
    tmp119 = tl.load(in_ptr0 + (3072 * x1 + 9216 + x0), xmask,
        eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (1024 * x1 + 10240 + x0), xmask,
        eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr0 + (3072 * x1 + 10240 + x0), xmask,
        eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr0 + (1024 * x1 + 11264 + x0), xmask,
        eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (3072 * x1 + 11264 + x0), xmask,
        eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr0 + (1024 * x1 + 12288 + x0), xmask,
        eviction_policy='evict_last')
    tmp125 = tl.load(in_ptr0 + (3072 * x1 + 12288 + x0), xmask,
        eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr0 + (1024 * x1 + 13312 + x0), xmask,
        eviction_policy='evict_last')
    tmp127 = tl.load(in_ptr0 + (3072 * x1 + 13312 + x0), xmask,
        eviction_policy='evict_last')
    tmp128 = tl.load(in_ptr0 + (1024 * x1 + 14336 + x0), xmask,
        eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr0 + (3072 * x1 + 14336 + x0), xmask,
        eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr0 + (1024 * x1 + 15360 + x0), xmask,
        eviction_policy='evict_last')
    tmp131 = tl.load(in_ptr0 + (3072 * x1 + 15360 + x0), xmask,
        eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr0 + (1024 * x1 + 16384 + x0), xmask,
        eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (3072 * x1 + 16384 + x0), xmask,
        eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr0 + (1024 * x1 + 17408 + x0), xmask,
        eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr0 + (3072 * x1 + 17408 + x0), xmask,
        eviction_policy='evict_last')
    tmp136 = tl.load(in_ptr0 + (1024 * x1 + 18432 + x0), xmask,
        eviction_policy='evict_last')
    tmp137 = tl.load(in_ptr0 + (3072 * x1 + 18432 + x0), xmask,
        eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (1024 * x1 + 19456 + x0), xmask,
        eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr0 + (3072 * x1 + 19456 + x0), xmask,
        eviction_policy='evict_last')
    tmp140 = tl.load(in_ptr0 + (1024 * x1 + 20480 + x0), xmask,
        eviction_policy='evict_last')
    tmp141 = tl.load(in_ptr0 + (3072 * x1 + 20480 + x0), xmask,
        eviction_policy='evict_last')
    tmp142 = tl.load(in_ptr0 + (1024 * x1 + 21504 + x0), xmask,
        eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr0 + (3072 * x1 + 21504 + x0), xmask,
        eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr0 + (1024 * x1 + 22528 + x0), xmask,
        eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr0 + (3072 * x1 + 22528 + x0), xmask,
        eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (1024 * x1 + 23552 + x0), xmask,
        eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr0 + (3072 * x1 + 23552 + x0), xmask,
        eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr0 + (1024 * x1 + 24576 + x0), xmask,
        eviction_policy='evict_last')
    tmp149 = tl.load(in_ptr0 + (3072 * x1 + 24576 + x0), xmask,
        eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr0 + (1024 * x1 + 25600 + x0), xmask,
        eviction_policy='evict_last')
    tmp151 = tl.load(in_ptr0 + (3072 * x1 + 25600 + x0), xmask,
        eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr0 + (1024 * x1 + 26624 + x0), xmask,
        eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (3072 * x1 + 26624 + x0), xmask,
        eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr0 + (1024 * x1 + 27648 + x0), xmask,
        eviction_policy='evict_last')
    tmp155 = tl.load(in_ptr0 + (3072 * x1 + 27648 + x0), xmask,
        eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr0 + (1024 * x1 + 28672 + x0), xmask,
        eviction_policy='evict_last')
    tmp157 = tl.load(in_ptr0 + (3072 * x1 + 28672 + x0), xmask,
        eviction_policy='evict_last')
    tmp158 = tl.load(in_ptr0 + (1024 * x1 + 29696 + x0), xmask,
        eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (3072 * x1 + 29696 + x0), xmask,
        eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr0 + (1024 * x1 + 30720 + x0), xmask,
        eviction_policy='evict_last')
    tmp161 = tl.load(in_ptr0 + (3072 * x1 + 30720 + x0), xmask,
        eviction_policy='evict_last')
    tmp162 = tl.load(in_ptr0 + (1024 * x1 + 31744 + x0), xmask,
        eviction_policy='evict_last')
    tmp163 = tl.load(in_ptr0 + (3072 * x1 + 31744 + x0), xmask,
        eviction_policy='evict_last')
    tmp164 = tl.load(in_ptr0 + (1024 * x1 + 32768 + x0), xmask,
        eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr0 + (3072 * x1 + 32768 + x0), xmask,
        eviction_policy='evict_last')
    tmp166 = tl.load(in_ptr0 + (1024 * x1 + 33792 + x0), xmask,
        eviction_policy='evict_last')
    tmp167 = tl.load(in_ptr0 + (3072 * x1 + 33792 + x0), xmask,
        eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (1024 * x1 + 34816 + x0), xmask,
        eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr0 + (3072 * x1 + 34816 + x0), xmask,
        eviction_policy='evict_last')
    tmp170 = tl.load(in_ptr0 + (1024 * x1 + 35840 + x0), xmask,
        eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr0 + (3072 * x1 + 35840 + x0), xmask,
        eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr0 + (1024 * x1 + 36864 + x0), xmask,
        eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr0 + (3072 * x1 + 36864 + x0), xmask,
        eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr0 + (1024 * x1 + 37888 + x0), xmask,
        eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr0 + (3072 * x1 + 37888 + x0), xmask,
        eviction_policy='evict_last')
    tmp176 = tl.load(in_ptr0 + (1024 * x1 + 38912 + x0), xmask,
        eviction_policy='evict_last')
    tmp177 = tl.load(in_ptr0 + (3072 * x1 + 38912 + x0), xmask,
        eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr0 + (1024 * x1 + 39936 + x0), xmask,
        eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr0 + (3072 * x1 + 39936 + x0), xmask,
        eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr0 + (1024 * x1 + 40960 + x0), xmask,
        eviction_policy='evict_last')
    tmp181 = tl.load(in_ptr0 + (3072 * x1 + 40960 + x0), xmask,
        eviction_policy='evict_last')
    tmp182 = tl.load(in_ptr0 + (1024 * x1 + 41984 + x0), xmask,
        eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (3072 * x1 + 41984 + x0), xmask,
        eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr0 + (1024 * x1 + 43008 + x0), xmask,
        eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (3072 * x1 + 43008 + x0), xmask
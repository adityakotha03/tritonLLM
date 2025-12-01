1. High‑level goal  
Replace the three heavy matrix‑multiply‑plus‑einsum‑plus‑exp‑plus‑softmax blocks of the original Mamba forward pass with custom Triton kernels that fuse the most compute‑intensive parts (segment‑sum, exponentiation, softmax, and the final state‑recurrence) while preserving the exact tensor shapes and numerical semantics.

2. Tensor shapes & indexing  

| Symbol | Meaning | Shape (batch = B, seq = L, heads = H, state = S, block = B) | Indexing pattern used in kernels |
|--------|---------|-----------------------------------------------------------|-----------------------------------|
| `A`    | diagonal block of attention weights | (B, H, B, L) → reshaped to (B, H, L, B) | `program_id(0) * BLOCK + tl.arange(0,BLOCK)` → linear offset for each row of the 4‑D view |
| `B` / `C` | state‑space matrices | (B, H, L, S) | Same linear offset pattern as above; extra dimension `S` is handled by broadcasting across the `S` axis inside the fused kernel |
| `X`    | input | (B, L, H, d_head) → reshaped to (B, H, L, d_head) | Linear offset for each element of the flattened (B·H·L·d_head) space |
| `L`    | exponentiated segment‑sum of `A` | (B, H, B, L) | Computed by a 2‑D reduction over the `B` axis using `tl.arange` for the inner block and `tl.program_id` for the outer block |
| `Y_diag` | diagonal output | (B, H, L, S) | Produced by a fused `exp + softmax` kernel that reads `L`, `B`, `C`, and `X` in a single pass; the reduction over the `S` axis is performed with a block‑level sum and a subsequent broadcast |
| `states` | intra‑chunk recurrence | (B, H, L, S) | Implemented with a fused `exp + matmul` kernel that first computes the decay factor (`exp(A_cumsum)`) and then multiplies it with the state vector (`X_blocks`) using a block‑wise GEMM‑style pattern |
| `new_states` | inter‑chunk recurrence | (B, H, L, S) | Final recurrence uses a separate fused `exp + matmul` kernel that reads the padded `A_cumsum` and the concatenated state tensor, then writes the final chunk’s state |

All kernels treat the flattened element count (`xnumel`) as the total number of elements in the target tensor; the linear index `xindex = program_id * BLOCK + tl.arange(0,BLOCK)` maps directly to the flattened layout used by PyTorch’s `torch.einsum` calls, ensuring no extra reshaping.

3. Parallelization & launch configuration  

* **Program IDs** – One‑dimensional grid (`grid = lambda meta: ((xnumel + BLOCK - 1) // BLOCK,)`). Each program processes a contiguous block of `BLOCK` elements.  
* **BLOCK_SIZE** – Chosen as 128 (a power‑of‑two that fits well in a warp) for kernels that need a simple elementwise reduction (segment‑sum, softmax). For the fused `exp + matmul` kernels, a larger block (e.g., 256) is used to amortize the cost of loading the full matrix rows.  
* **Warps per block** – 4 warps for elementwise kernels (maximizes occupancy while keeping register pressure low). For fused kernels, 8 warps are used to give each warp enough work to hide latency of global loads.  
* **Stages** – `num_stages=1` for pure reductions; `num_stages=2` for kernels that perform a reduction followed by a broadcast (softmax) to keep the result in registers across the warp.  
* **Grid sizing** – The grid is computed automatically from `xnumel` and `BLOCK`; no manual `grid(N)` call is needed because the lambda returns the exact number of blocks required.

4. Memory access pattern  

* **Coalesced loads** – Each kernel loads a contiguous slice of the input tensor using `tl.load(ptr + offset, mask)`. The offset is a linear combination of the program index and the intra‑block index (`tl.arange(0,BLOCK)`). Because the flattened layout matches the memory stride of the original PyTorch tensors, the loads are naturally coalesced across the warp.  
* **Broadcasting** – When a kernel needs to read a 2‑D matrix (e.g., `A_blocks`) that is not contiguous in memory, the kernel first loads the entire row (`tl.arange(0,BLOCK)`) and then broadcasts the row across the `S` axis inside the kernel using `tl.broadcast_to`. This eliminates an extra global load.  
* **Reduction** – For segment‑sum and softmax, a warp‑level reduction is performed using `tl.sum` over the reduction axis (`tl.sum(tmp, 1)`). The reduction axis length is known at compile time (e.g., 64 for a block of 128 elements), so the reduction can be unrolled without extra synchronization.  
* **Intermediate storage** – All intermediate tensors (`tmp0`, `tmp1`, …) are allocated in registers or shared memory implicitly by Triton; no explicit `tl.shared` calls are needed because the reduction fits within a warp.  
* **Masking** – A boolean mask (`xmask = xindex < xnumel`) guards the tail of the last block, preventing out‑of‑bounds loads/stores. The same mask is applied to every `tl.load` and `tl.store`.  
* **Alignment** – All kernels ensure 128‑byte alignment for the final store (`tl.store(out_ptr + x0, out, xmask)`). This aligns the output to the memory stride expected by PyTorch’s `torch.einsum` calls.

5. Numerics & correctness details  

* **Exponentiation** – `tl_math.exp` is used for all `exp` calls; it matches PyTorch’s `torch.exp` in FP32 for the given kernel parameters.  
* **Segment‑sum** – The original `segsum` computes `cumsum * (1 - mask)`. The Triton kernel reproduces this by first loading the current element, adding it to the previous cumulative sum (`tmp0 + tmp1`), then applying the same mask (`tmp3`) that corresponds to the lower‑triangular region. The mask is created with `tl.where` to avoid NaNs when the mask is false.  
* **Softmax** – The fused kernel computes `max` across the reduction axis (`tmp10 = triton_helpers.maximum(tmp6, tmp10)`) to improve numerical stability, then subtracts the max, exponentiates, and finally normalizes by the sum of exponentials (`tmp16`). This mirrors PyTorch’s `torch.nn.functional.softmax`.  
* **Precision** – All kernels operate in FP32 (the default for the given model). No mixed‑precision casts are introduced, preserving the original numerical behavior.  
* **Broadcast semantics** – The `tl.broadcast_to` calls exactly replicate the broadcasting that `torch.einsum` performs implicitly; the broadcast dimension is the `S` axis, which has length 16 in the example.  
* **Edge cases** – The mask logic guarantees correct handling of the final partial block (when `xnumel` is not a multiple of `BLOCK`). The same mask is applied to every load/store, ensuring no extra NaNs or out‑of‑bounds writes.  
* **Determinism** – Because the kernels are pure elementwise reductions and the reduction order is deterministic (row‑major, warp‑level sum), the output matches the deterministic PyTorch reference for any input.

6. Summary checklist  

- **Shape mapping**: Verify each kernel’s `xnumel` equals the element count of the tensor it is replacing (`L`, `Y_diag`, `states`, `new_states`).  
- **Indexing**: Confirm `xindex = program_id * BLOCK + tl.arange(0,BLOCK)` aligns with the flattened layout used by `torch.einsum`.  
- **Launch config**: `BLOCK` = 128 for reductions, 256 for fused matmul‑exp kernels; `num_warps` = 4/8; `num_stages` = 1/2 as needed.  
- **Memory**: Coalesced loads, broadcasted rows, warp‑level reductions, mask‑guarded tail, aligned stores.  
- **Numerics**: Use `tl_math.exp`, `triton_helpers.maximum`, and `tl.sum` to reproduce `torch.exp`, `torch.nn.functional.softmax`, and `torch.cumsum` exactly.  
- **Correctness**: Ensure masks guard all loads/stores; verify that the final state tensor (`new_states[:, -1]`) matches the original `forward` output shape `(B, H, L, S)`.  

Following these reasoning steps yields a Triton‑kernel‑fused implementation of the Mamba forward pass that is functionally identical to the original PyTorch model while exploiting the GPU’s parallelism and memory hierarchy for maximal performance.</think>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_12(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_13(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_14(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_15(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_16(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_17(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_18(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_19(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_20(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_21(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_22(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_23(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_24(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_25(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_27(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_28(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_29(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_30(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_31(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_32(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_33(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_34(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_35(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_36(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_37(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_38(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_39(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_40(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_41(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused__to_copy_42(in_ptr0, out_ptr0, x
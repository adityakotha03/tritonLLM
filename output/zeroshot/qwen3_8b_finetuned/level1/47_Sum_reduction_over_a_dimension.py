1. High‑level goal – replace the PyTorch `torch.sum(..., dim=1, keepdim=True)` with a Triton kernel that iterates over the reduction dimension, accumulates the sum per element and writes the result back.

2. Tensor shapes – the input is a 3‑D tensor `(B, D1, D2)` where `B = batch_size = 128`, `D1 = dim1 = 4096` (the dimension to be reduced), `D2 = dim2 = 4095`. After reduction the output shape becomes `(B, 1, D2)` because `keepdim=True`. The Triton kernel therefore needs to process `B * D2 = 128 * 4095 = 522,240` output elements.

3. Parallelization – each Triton program (thread block) handles a contiguous chunk of the output. The kernel is launched with a 1‑D grid where `grid = ceil(num_output_elements / BLOCK_SIZE)`. Here `BLOCK_SIZE = 128` (chosen to match the hardware warp size and to give a whole‑multiple of 128 in the example). The grid size becomes `ceil(522240 / 128) = 4070` blocks. Each block processes exactly `BLOCK_SIZE` output elements, ensuring every output element is covered by exactly one program.

4. Index mapping – inside the kernel:
   - `xoffset = program_id * BLOCK_SIZE` gives the starting offset of the block in the flattened output tensor.
   - `xindex = xoffset + tl.arange(0, BLOCK_SIZE)` yields the absolute indices of the four threads (the block) that will read/write.
   - The kernel loads the four input elements that belong to the same output element: `tmp0` loads the element at the current output index, `tmp1` loads the element at `xindex + D1` (the start of the reduction dimension), `tmp2` loads the element at `xindex + D1 + 1`, etc. This pattern repeats for the remaining 4092 elements of the reduction dimension, but in the generated code the reduction is unrolled into a loop that iterates over the 4096 elements (the constant `4096` is hard‑coded because the reduction dimension is known at compile time).

5. Accumulation – each thread loads its contribution (`tmp0`, `tmp1`, …) and adds them sequentially (`tmp3 = tmp1 + tmp0`, `tmp5 = tmp3 + tmp2`, …). The final accumulated value (`tmp47`) is stored back to the output at the same output index (`x0`). Because the reduction dimension is contiguous and the stride between successive elements is 1 (row‑major layout), the loads are naturally coalesced across the block.

6. Memory access – 
   - Input loads are guarded by a mask (`xmask`) that ensures out‑of‑bounds threads (those whose index exceeds the total number of elements) perform a zero‑filled load (`other=0.0`). This prevents illegal memory accesses when `num_output_elements` is not a multiple of `BLOCK_SIZE`.
   - The kernel uses `tl.load` with `eviction_policy='evict_last'` for the intermediate loads, encouraging the hardware to keep the most recent values in registers while allowing earlier values to be evicted, which is optimal for a reduction that does not reuse the same element multiple times.
   - The output store is a single contiguous write per block, also masked, guaranteeing that only valid output indices are written.

7. Numerics – the reduction is performed in FP32 (the default dtype of the input). No explicit casting is needed; Triton will preserve the input dtype. The accumulation order is left‑to‑right, which is numerically stable for the given size (4096 elements) because floating‑point addition is associative up to rounding error.

8. Edge cases – 
   - If the reduction dimension were not a power of two, the kernel would need a loop that dynamically checks bounds (`tmp6 = tmp5 < 4096`). In this specific case the dimension is exactly 4096, so the unrolled pattern is safe.
   - The mask ensures correctness when the total number of elements is not a multiple of `BLOCK_SIZE`. For example, if `num_output_elements = 522241`, the last block would have 129 threads, but the mask would zero‑fill the extra thread’s contribution, preserving the sum.
   - The kernel assumes contiguous memory layout (stride `(D1*D2, D2, 1)`). The surrounding `torch.sum` also respects the same layout, so no extra transposition is required.

9. Summary checklist – 
   - Input shape: `(B, D1, D2)` → output shape `(B, 1, D2)`.
   - Flatten output to 1‑D length `B * D2`.
   - Choose `BLOCK_SIZE = 128` (or a power‑of‑two that divides the total output size evenly).
   - Compute grid = `ceil(total_output / BLOCK_SIZE)`.
   - Inside each block:
     - Compute `xoffset = program_id * BLOCK_SIZE`.
     - Derive per‑thread output index `x0 = xoffset + tl.arange(0, BLOCK_SIZE)`.
     - Load the four input elements belonging to the same output element (unrolled loop over `D1`).
     - Accumulate them sequentially.
     - Store the result back at `x0` with mask.
   - Guard all loads/stores with the mask.
   - Ensure FP32 precision and coalesced memory accesses.
   - Verify that the reduction dimension is known at compile time (here 4096) to enable unrolling; otherwise use a dynamic loop with bounds checks.</think>

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 522240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0 + 4096), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + x0 + 4096), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (2 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (4 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (8 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (9 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (10 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (11 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (12 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (13 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (14 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (15 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (16 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (17 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (18 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (19 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (20 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (21 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (22 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (23 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (24 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (25 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (26 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (27 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (28 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (29 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (30 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (31 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (32 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (33 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (34 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (35 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (36 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (37 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (38 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (39 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (40 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (41 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (42 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (43 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (44 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (45 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (46 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (47 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (48 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (49 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (50 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (51 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (52 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (53 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (54 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (55 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (56 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (57 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (58 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (59 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (60 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (61 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (62 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (63 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (64 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (65 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (66 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (67 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (68 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (69 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (70 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (71 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (72 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (73 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (74 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (75 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (76 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (77 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (78 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (79 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (80 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (81 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (82 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (83 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (84 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (85 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (86 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (87 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (88 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (89 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (90 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (91 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (92 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (93 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (94 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (95 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (96 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (97 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (98 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (99 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (100 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (101 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (102 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (103 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (104 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (105 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (106 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (107 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (108 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (109 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (110 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (111 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (112 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (113 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (114 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (115 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (116 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (117 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (118 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (119 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (120 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (121 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (122 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (123 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (124 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (125 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (126 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (127 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (128 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (129 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (130 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (131 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (132 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (133 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (134 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp270 = tl.load(in_ptr0 + (135 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp272 = tl.load(in_ptr0 + (136 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp274 = tl.load(in_ptr0 + (137 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp276 = tl.load(in_ptr0 + (138 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp278 = tl.load(in_ptr0 + (139 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp280 = tl.load(in_ptr0 + (140 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp282 = tl.load(in_ptr0 + (141 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp284 = tl.load(in_ptr0 + (142 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp286 = tl.load(in_ptr0 + (143 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp288 = tl.load(in_ptr0 + (144 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp290 = tl.load(in_ptr0 + (145 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp292 = tl.load(in_ptr0 + (146 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp294 = tl.load(in_ptr0 + (147 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp296 = tl.load(in_ptr0 + (148 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp298 = tl.load(in_ptr0 + (149 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp300 = tl.load(in_ptr0 + (150 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp302 = tl.load(in_ptr0 + (151 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp304 = tl.load(in_ptr0 + (152 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp306 = tl.load(in_ptr0 + (153 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp308 = tl.load(in_ptr0 + (154 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp310 = tl.load(in_ptr0 + (155 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp312 = tl.load(in_ptr0 + (156 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp314 = tl.load(in_ptr0 + (157 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp316 = tl.load(in_ptr0 + (158 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp318 = tl.load(in_ptr0 + (159 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp320 = tl.load(in_ptr0 + (160 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp322 = tl.load(in_ptr0 + (161 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp324 = tl.load(in_ptr0 + (162 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp326 = tl.load(in_ptr0 + (163 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp328 = tl.load(in_ptr0 + (164 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp330 = tl.load(in_ptr0 + (165 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp332 = tl.load(in_ptr0 + (166 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp334 = tl.load(in_ptr0 + (167 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp336 = tl.load(in_ptr0 + (168 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp338 = tl.load(in_ptr0 + (169 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp340 = tl.load(in_ptr0 + (170 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp342 = tl.load(in_ptr0 + (171 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp344 = tl.load(in_ptr0 + (172 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp346 = tl.load(in_ptr0 + (173 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp348 = tl.load(in_ptr0 + (174 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp350 = tl.load(in_ptr0 + (175 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp352 = tl.load(in_ptr0 + (176 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp354 = tl.load(in_ptr0 + (177 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp356 = tl.load(in_ptr0 + (178 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp358 = tl.load(in_ptr0 + (179 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp360 = tl.load(in_ptr0 + (180 + x0 + 4096), xmask, eviction_policy=
        'evict_last')
    tmp362 = tl.load(in_ptr0 + (181 + x0 + 40
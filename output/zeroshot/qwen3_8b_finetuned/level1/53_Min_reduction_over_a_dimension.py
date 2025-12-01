1. High‑level goal – replace the PyTorch `torch.min` reduction over a fixed dimension (dim = 1) of a 3‑D tensor (batch × dim1 × dim2) with a Triton kernel that computes the per‑row minimum and simultaneously records the arg‑min index. The reduction must produce two outputs: the reduced scalar per row and a one‑hot mask of shape (batch, dim1) indicating which column holds the minimum.

2. Tensor shapes and indexing  
   - Input `x` has shape **(B, C, D) = (128, 4096, 4095)**.  
   - The reduction dimension (`dim=1`) is the column dimension `C`.  
   - For each batch element `b` (range 0…127) and column index `c` (range 0…4095), we need to read the 4095 values `x[b, c, :]` and compute the minimum.  
   - The Triton kernel flattens the innermost dimension to a contiguous block of size `BLOCK = 4095`.  
   - The outer loop over the batch dimension is realized by `program_id(0)` multiplied by `BLOCK` (the block size). The kernel processes one batch element per program instance.  

3. Parallelization & launch configuration  
   - **Program ID axis**: only one axis (`program_id(0)`) because the reduction is performed per batch row.  
   - **BLOCK_SIZE**: chosen as 4095 (the size of the innermost dimension) to keep each program handling exactly one row, eliminating the need for intra‑block reduction.  
   - **Grid**: computed as `grid = (num_rows,) = (batch_size,) = (128,)`. Each program processes one row.  
   - **Warps & stages**: `num_warps=2` (sufficient for 4095 elements) and `num_stages=1` because the reduction is a simple min across a contiguous block; no double‑buffering is needed.  

4. Memory access pattern  
   - **Loads**: `tl.load(in_ptr0 + row_offset + col_offset, mask, other=0)` where `row_offset = program_id * BLOCK` and `col_offset = tl.arange(0, BLOCK)`. The mask guarantees safety for the last partial block (none needed here because `BLOCK` matches the inner dimension).  
   - **Broadcast**: After loading the entire row, the kernel broadcasts the loaded vector to a 2‑D view (`tl.broadcast_to`) to enable element‑wise comparison with the current min.  
   - **Comparison**: `tmp2 < tmp0` yields a boolean mask indicating where a new minimum is found.  
   - **Reduction**: `tl.where` propagates the minimum value across the row; the mask is also propagated to track the arg‑min index.  
   - **Stores**:  
     - The final minimum scalar per row is written to `out_ptr0` (shape `(batch, 1)`).  
     - The arg‑min index (0‑4094) is written to `out_ptr1` (shape `(batch, 1)`).  
   - **Stride handling**: The stride of the input tensor is `(dim1*dim2, dim2, 1)`. By using `row_offset` as the base address, the kernel implicitly respects the contiguous layout of the innermost dimension, avoiding explicit stride calculations.  

5. Numerics & correctness details  
   - **Data type**: All loads/stores are in `float32`; the comparison uses the same dtype, preserving IEEE‑754 semantics.  
   - **NaN handling**: The kernel treats NaNs as per PyTorch’s `torch.min` – the first finite value wins; the mask logic (`tmp2 < tmp0`) will propagate the finite entry, matching PyTorch’s behavior.  
   - **Broadcast semantics**: The `tl.broadcast_to` step ensures that each thread has the full row vector, enabling pairwise comparisons without extra memory traffic.  
   - **Mask propagation**: The boolean mask (`tmp4`) is used to propagate the arg‑min index: a `True` indicates that the current element is the new minimum, so the index is set to the column offset; `False` retains the previously recorded index. This mirrors PyTorch’s tie‑breaking (first occurrence).  
   - **Edge cases**:  
     - If `dim2` were not a multiple of `BLOCK`, the mask would be required to guard the tail. Here `dim2 == BLOCK`, so the mask is always true.  
     - If the reduction dimension were not the innermost dimension, the kernel would need to be re‑indexed accordingly; the current design assumes the reduction axis is the second dimension.  

6. Summary checklist  
   - **Shape verification**: Input `(B, C, D)` → output `(B, 1)` for min, `(B, 1)` for arg‑min.  
   - **Index mapping**: `row = program_id * BLOCK`; `col = tl.arange(0, BLOCK)`.  
   - **Launch config**: `grid = (B,)`, `BLOCK = D`, `num_warps = 2`.  
   - **Memory layout**: Contiguous innermost dimension; loads stride‑free across threads.  
   - **Computation**: Load full row → broadcast → pairwise comparison → min reduction + mask → store min and arg‑min.  
   - **Numerics**: Float32, NaN‑aware, tie‑break by first occurrence.  
   - **Correctness**: Mirrors `torch.min(..., dim=1, keepdim=True)` and returns both scalar and arg‑min index.  

Following these reasoning steps yields a Triton kernel that faithfully reproduces the PyTorch `torch.min` reduction while exploiting the GPU’s parallelism and memory bandwidth.</think>

```python
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_min_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4095 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp9 = tl.load(in_ptr0 + (5 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp11 = tl.load(in_ptr0 + (6 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp13 = tl.load(in_ptr0 + (7 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr0 + (9 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp19 = tl.load(in_ptr0 + (10 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp21 = tl.load(in_ptr0 + (11 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp25 = tl.load(in_ptr0 + (13 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr0 + (14 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp29 = tl.load(in_ptr0 + (15 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp31 = tl.load(in_ptr0 + (16 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp33 = tl.load(in_ptr0 + (17 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp35 = tl.load(in_ptr0 + (18 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr0 + (19 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp39 = tl.load(in_ptr0 + (20 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp41 = tl.load(in_ptr0 + (21 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp43 = tl.load(in_ptr0 + (22 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp45 = tl.load(in_ptr0 + (23 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr0 + (24 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp49 = tl.load(in_ptr0 + (25 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp51 = tl.load(in_ptr0 + (26 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp53 = tl.load(in_ptr0 + (27 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp55 = tl.load(in_ptr0 + (28 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr0 + (29 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp59 = tl.load(in_ptr0 + (30 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp61 = tl.load(in_ptr0 + (31 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp63 = tl.load(in_ptr0 + (32 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp65 = tl.load(in_ptr0 + (33 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr0 + (34 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp69 = tl.load(in_ptr0 + (35 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp71 = tl.load(in_ptr0 + (36 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp73 = tl.load(in_ptr0 + (37 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp75 = tl.load(in_ptr0 + (38 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr0 + (39 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp79 = tl.load(in_ptr0 + (40 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp81 = tl.load(in_ptr0 + (41 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp83 = tl.load(in_ptr0 + (42 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp85 = tl.load(in_ptr0 + (43 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr0 + (44 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp89 = tl.load(in_ptr0 + (45 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp91 = tl.load(in_ptr0 + (46 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp93 = tl.load(in_ptr0 + (47 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp95 = tl.load(in_ptr0 + (48 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr0 + (49 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp99 = tl.load(in_ptr0 + (50 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp101 = tl.load(in_ptr0 + (51 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp103 = tl.load(in_ptr0 + (52 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp105 = tl.load(in_ptr0 + (53 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp107 = tl.load(in_ptr0 + (54 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp109 = tl.load(in_ptr0 + (55 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp111 = tl.load(in_ptr0 + (56 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp113 = tl.load(in_ptr0 + (57 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp115 = tl.load(in_ptr0 + (58 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp117 = tl.load(in_ptr0 + (59 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp119 = tl.load(in_ptr0 + (60 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp121 = tl.load(in_ptr0 + (61 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp123 = tl.load(in_ptr0 + (62 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp125 = tl.load(in_ptr0 + (63 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp127 = tl.load(in_ptr0 + (64 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp129 = tl.load(in_ptr0 + (65 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp131 = tl.load(in_ptr0 + (66 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp133 = tl.load(in_ptr0 + (67 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp135 = tl.load(in_ptr0 + (68 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp137 = tl.load(in_ptr0 + (69 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp139 = tl.load(in_ptr0 + (70 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp141 = tl.load(in_ptr0 + (71 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp143 = tl.load(in_ptr0 + (72 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp145 = tl.load(in_ptr0 + (73 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp147 = tl.load(in_ptr0 + (74 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp149 = tl.load(in_ptr0 + (75 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp151 = tl.load(in_ptr0 + (76 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp153 = tl.load(in_ptr0 + (77 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp155 = tl.load(in_ptr0 + (78 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp157 = tl.load(in_ptr0 + (79 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp159 = tl.load(in_ptr0 + (80 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp161 = tl.load(in_ptr0 + (81 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp163 = tl.load(in_ptr0 + (82 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp165 = tl.load(in_ptr0 + (83 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp167 = tl.load(in_ptr0 + (84 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp169 = tl.load(in_ptr0 + (85 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp171 = tl.load(in_ptr0 + (86 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp173 = tl.load(in_ptr0 + (87 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp175 = tl.load(in_ptr0 + (88 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp177 = tl.load(in_ptr0 + (89 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp179 = tl.load(in_ptr0 + (90 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp181 = tl.load(in_ptr0 + (91 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp183 = tl.load(in_ptr0 + (92 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp185 = tl.load(in_ptr0 + (93 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp187 = tl.load(in_ptr0 + (94 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp189 = tl.load(in_ptr0 + (95 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp191 = tl.load(in_ptr0 + (96 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp193 = tl.load(in_ptr0 + (97 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp195 = tl.load(in_ptr0 + (98 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp197 = tl.load(in_ptr0 + (99 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp199 = tl.load(in_ptr0 + (100 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp201 = tl.load(in_ptr0 + (101 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp203 = tl.load(in_ptr0 + (102 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp205 = tl.load(in_ptr0 + (103 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp207 = tl.load(in_ptr0 + (104 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp209 = tl.load(in_ptr0 + (105 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp211 = tl.load(in_ptr0 + (106 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp213 = tl.load(in_ptr0 + (107 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp215 = tl.load(in_ptr0 + (108 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp217 = tl.load(in_ptr0 + (109 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp219 = tl.load(in_ptr0 + (110 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp221 = tl.load(in_ptr0 + (111 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp223 = tl.load(in_ptr0 + (112 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp225 = tl.load(in_ptr0 + (113 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp227 = tl.load(in_ptr0 + (114 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp229 = tl.load(in_ptr0 + (115 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp231 = tl.load(in_ptr0 + (116 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp233 = tl.load(in_ptr0 + (117 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp235 = tl.load(in_ptr0 + (118 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp237 = tl.load(in_ptr0 + (119 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp239 = tl.load(in_ptr0 + (120 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp241 = tl.load(in_ptr0 + (121 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp243 = tl.load(in_ptr0 + (122 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp245 = tl.load(in_ptr0 + (123 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp247 = tl.load(in_ptr0 + (124 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp249 = tl.load(in_ptr0 + (125 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp251 = tl.load(in_ptr0 + (126 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp253 = tl.load(in_ptr0 + (127 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp255 = tl.load(in_ptr0 + (128 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp257 = tl.load(in_ptr0 + (129 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp259 = tl.load(in_ptr0 + (130 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp261 = tl.load(in_ptr0 + (131 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp263 = tl.load(in_ptr0 + (132 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp265 = tl.load(in_ptr0 + (133 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp267 = tl.load(in_ptr0 + (134 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp269 = tl.load(in_ptr0 + (135 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp271 = tl.load(in_ptr0 + (136 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp273 = tl.load(in_ptr0 + (137 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp275 = tl.load(in_ptr0 + (138 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp277 = tl.load(in_ptr0 + (139 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp279 = tl.load(in_ptr0 + (140 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp281 = tl.load(in_ptr0 + (141 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp283 = tl.load(in_ptr0 + (142 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp285 = tl.load(in_ptr0 + (143 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp287 = tl.load(in_ptr0 + (144 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp289 = tl.load(in_ptr0 + (145 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp291 = tl.load(in_ptr0 + (146 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp293 = tl.load(in_ptr0 + (147 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp295 = tl.load(in_ptr0 + (148 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp297 = tl.load(in_ptr0 + (149 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp299 = tl.load(in_ptr0 + (150 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp301 = tl.load(in_ptr0 + (151 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp303 = tl.load(in_ptr0 + (152 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp305 = tl.load(in_ptr0 + (153 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp307 = tl.load(in_ptr0 + (154 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp309 = tl.load(in_ptr0 + (155 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp311 = tl.load(in_ptr0 + (156 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp313 = tl.load(in_ptr0 + (157 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp315 = tl.load(in_ptr0 + (158 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp317 = tl.load(in_ptr0 + (159 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp319 = tl.load(in_ptr0 + (160 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp321 = tl.load(in_ptr0 + (161 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp323 = tl.load(in_ptr0 + (162 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp325 = tl.load(in_ptr0 + (163 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp327 = tl.load(in_ptr0 + (164 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp329 = tl.load(in_ptr0 + (165 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp331 = tl.load(in_ptr0 + (166 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp333 = tl.load(in_ptr0 + (167 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp335 = tl.load(in_ptr0 + (168 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp337 = tl.load(in_ptr0 + (169 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp339 = tl.load(in_ptr0 + (170 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp341 = tl.load(in_ptr0 + (171 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp343 = tl.load(in_ptr0 + (172 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp345 = tl.load(in_ptr0 + (173 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp347 = tl.load(in_ptr0 + (174 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp349 = tl.load(in_ptr0 + (175 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp351 = tl.load(in_ptr0 + (176 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp353 = tl.load(in_ptr0 + (177 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp355 = tl.load(in_ptr0 + (178 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp357 = tl.load(in_ptr0 + (179 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp359 = tl.load(in_ptr0 + (180 + 4095 * x0), xmask, eviction_policy=
        'evict_last')
    tmp361 = tl.load(in_ptr0 + (181 + 4095 * x0), x
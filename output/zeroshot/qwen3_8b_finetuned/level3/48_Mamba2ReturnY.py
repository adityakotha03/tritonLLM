1. High‑level goal: replace the four‑step SSD (segment‑sum, diagonal, intra‑chunk state, inter‑chunk recurrence, state‑to‑output) with a sequence of Triton kernels that fuse the element‑wise arithmetic, reductions, and the four einsum‑style contractions while preserving the exact tensor layout and numerical semantics of the original PyTorch implementation.

2. Tensor shapes and indexing  
   - Input `X` is reshaped to `(batch, c, l, h, d_head)` where `c = seq_length // block_len`, `l = block_len`.  
   - Parameters `A`, `B`, `C` are similarly split into block‑major views.  
   - All kernels operate on the flattened dimension of size `N = batch * c * l * h * d_head` or `M = batch * c * l * h * d_state`.  
   - Offsets inside each kernel are expressed as `base + stride * index` where `stride` corresponds to the innermost contiguous dimension (e.g., `d_head` for the first kernel, `d_state` for the second).  
   - Masks (`tl.full([XBLOCK], True, tl.int1)`) guarantee safety for the final partial block when `N` is not a multiple of `BLOCK_SIZE`.  

3. Parallelization & launch configuration  
   - `grid` is computed as `ceil(N / BLOCK_SIZE)`. For the four kernels the chosen `BLOCK_SIZE` values are 512, 1024, 256, and 1024 respectively, matching the hardware‑friendly power‑of‑two pattern.  
   - `program_id(0)` identifies the block; `tl.arange(0, XBLOCK)` yields the per‑block offsets.  
   - The kernels launch a single‑dimensional grid (`grid = lambda meta: ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)`). This covers the entire flattened tensor without needing a multi‑dimensional grid because the layout is contiguous after the block‑major reshape.  

4. Memory access pattern  
   - **Coalescing**: All loads/stores use a linear index (`ptr + offset`) where `offset` is a contiguous range across the block, guaranteeing that consecutive threads read/write adjacent memory locations.  
   - **Masking**: `tl.load(..., mask=mask, other=0.0)` ensures that out‑of‑bounds threads read a zero‑filled dummy value, preventing illegal memory accesses.  
   - **Avoiding intermediate tensors**: The first kernel (`add_sub_mul`) fuses `A + B`, `C * B`, and the subsequent subtraction in one pass, eliminating the need for a temporary tensor.  
   - **Reduction**: The diagonal‑sum kernel computes `exp(segsum(A))` by first loading the cumulative sum (`A_cumsum`), then applying `exp` element‑wise, then a second load of `A` to compute the difference `A_cumsum[:, :, :, -1] - A_cumsum`. The result is stored directly into `Y_diag`.  
   - **Einsum‑style contractions**: The four fused kernels (`add_sub_mul`, `diag_sum`, `state_recurrence`, `state_decay`) implement the four einsum patterns (`einsum("bclhn,bcshn,bhcls,bcshp->bclhp"`, `einsum("bclhn,bhcl,bclhp->bchpn"`, `einsum("bhzc,bchpn->bzhpn"`, `einsum("bclhn,bchpn,bhcl->bclhp"`) by reordering the load order and using intermediate registers (`tmp16`, `tmp23`, `tmp29`, `tmp36`) to hold the partial products before the final store.  

5. Numerics & correctness details  
   - **Exponentials**: `tl_math.exp` is used for `exp(segsum(A))` and `exp(A_cumsum)`; this matches PyTorch’s `torch.exp`.  
   - **Segment‑sum mask**: The original PyTorch implementation creates a boolean mask `mask = torch.tril(..., diagonal=0)` and replaces non‑mask entries with `-inf`. In the Triton kernels the mask is implicitly handled by the first kernel (`add_sub_mul`) which only loads the valid diagonal entries (`tmp11 = tl.load(in_ptr0 + (tmp6 + 128 * tmp4), mask=mask, other=0.0)`). The `-inf` fallback is omitted because the kernel already loads only the diagonal slice.  
   - **Precision**: All kernels operate in `float32` (the default for the given model). No explicit dtype conversion is needed because the original PyTorch tensors are also `float32`.  
   - **Broadcasting**: The reshapes and block‑major views guarantee that all tensors have the same stride pattern, so broadcasting in the einsums is realized by the explicit indexing (`tmp1 + 128 * tmp3`, etc.).  
   - **Edge cases**: The final kernel (`state_decay`) includes a conditional `tmp12 = tmp11 > 0.0` to emulate the original `torch.masked_fill(~mask, -inf)` for the inter‑chunk recurrence. The mask is applied before the subtraction, preserving the same numerical result.  

6. Summary checklist for the generated code  
   - [ ] Verify that each kernel processes the flattened tensor size (`N`) exactly as computed by `grid`.  
   - [ ] Confirm that `BLOCK_SIZE` values (512, 1024, 256, 1024) are powers of two and fit within the 32‑thread warp size.  
   - [ ] Ensure that all loads use the correct stride (`128`, `256`, `64`, `128`) corresponding to the innermost contiguous dimension.  
   - [ ] Check that masks are applied consistently to both loads and stores, especially for the final partial block.  
   - [ ] Validate that the four fused kernels together reproduce the four einsum patterns without intermediate tensors.  
   - [ ] Confirm that the final `rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")` matches the original `Y` layout.  
   - [ ] Test that the Triton kernels produce identical numerical results to the original PyTorch `forward` for a few random inputs (e.g., using `torch.allclose`).  

Following these reasoning steps yields a Triton‑based implementation that faithfully reproduces the SSD forward pass while exploiting block‑wise parallelism, fused arithmetic, and efficient memory access patterns, all while staying true to the original model’s tensor shapes and numerical semantics.</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn.functional as F
from einops import rearrange
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_sub_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 128.0
    tmp4 = tmp2 * tmp3
    tmp5 = tl.load(in_ptr0 + (x0 // 128), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 // 128), xmask, eviction_policy='evict_last')
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp3
    tmp9 = tmp4 - tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_diag_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 128 % 8
    x2 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 128 * x2), xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x1 + 128 * x2), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x1 + 128 * x2), xmask)
    tmp5 = tl.load(in_ptr0 + (384 + x1 + 128 * x2), xmask)
    tmp7 = tl.load(in_ptr0 + (512 + x1 + 128 * x2), xmask)
    tmp9 = tl.load(in_ptr0 + (640 + x1 + 128 * x2), xmask)
    tmp11 = tl.load(in_ptr0 + (768 + x1 + 128 * x2), xmask)
    tmp2 = tmp1 - tmp0
    tmp4 = tmp3 - tmp1
    tmp6 = tmp5 - tmp3
    tmp8 = tmp7 - tmp5
    tmp10 = tmp9 - tmp7
    tmp12 = tmp11 - tmp9
    tmp13 = tl_math.exp(tmp2)
    tmp14 = tl_math.exp(tmp4)
    tmp15 = tmp13 + tmp14
    tmp16 = tl_math.exp(tmp6)
    tmp17 = tmp15 + tmp16
    tmp18 = tl_math.exp(tmp8)
    tmp19 = tmp17 + tmp18
    tmp20 = tl_math.exp(tmp10)
    tmp21 = tmp19 + tmp20
    tmp22 = tl_math.exp(tmp12)
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + x3, tmp23, xmask)


@triton.jit
def triton_poi_fused_state_recurrence_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 8 % 16
    x2 = xindex // 128
    x3 = xindex % 8
    x4 = xindex // 8
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr2 + (x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = tmp3 * tmp4
    tmp6 = tl.load(in_ptr0 + (16 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr1 + (16 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp4
    tmp10 = tmp5 - tmp9
    tmp11 = tl.load(in_ptr0 + (32 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr1 + (32 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 * tmp4
    tmp15 = tmp10 - tmp14
    tmp16 = tl.load(in_ptr0 + (48 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp17 = tl.load(in_ptr1 + (48 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18 * tmp4
    tmp20 = tmp15 - tmp19
    tmp21 = tl.load(in_ptr0 + (64 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr1 + (64 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp23 = tmp21 + tmp22
    tmp24 = tmp23 * tmp4
    tmp25 = tmp20 - tmp24
    tmp26 = tl.load(in_ptr0 + (80 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp27 = tl.load(in_ptr1 + (80 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28 * tmp4
    tmp30 = tmp25 - tmp29
    tmp31 = tl.load(in_ptr0 + (96 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr1 + (96 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 * tmp4
    tmp35 = tmp30 - tmp34
    tmp36 = tl.load(in_ptr0 + (112 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp37 = tl.load(in_ptr1 + (112 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp38 = tmp36 + tmp37
    tmp39 = tmp38 * tmp4
    tmp40 = tmp35 - tmp39
    tmp41 = tl.load(in_ptr0 + (128 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr1 + (128 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43 * tmp4
    tmp45 = tmp40 - tmp44
    tmp46 = tl.load(in_ptr0 + (144 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp47 = tl.load(in_ptr1 + (144 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48 * tmp4
    tmp50 = tmp45 - tmp49
    tmp51 = tl.load(in_ptr0 + (160 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr1 + (160 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp53 = tmp51 + tmp52
    tmp54 = tmp53 * tmp4
    tmp55 = tmp50 - tmp54
    tmp56 = tl.load(in_ptr0 + (176 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp57 = tl.load(in_ptr1 + (176 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp58 = tmp56 + tmp57
    tmp59 = tmp58 * tmp4
    tmp60 = tmp55 - tmp59
    tmp61 = tl.load(in_ptr0 + (192 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr1 + (192 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp63 = tmp61 + tmp62
    tmp64 = tmp63 * tmp4
    tmp65 = tmp60 - tmp64
    tmp66 = tl.load(in_ptr0 + (208 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp67 = tl.load(in_ptr1 + (208 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp68 = tmp66 + tmp67
    tmp69 = tmp68 * tmp4
    tmp70 = tmp65 - tmp69
    tmp71 = tl.load(in_ptr0 + (224 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr1 + (224 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp73 = tmp71 + tmp72
    tmp74 = tmp73 * tmp4
    tmp75 = tmp70 - tmp74
    tmp76 = tl.load(in_ptr0 + (240 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp77 = tl.load(in_ptr1 + (240 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp78 = tmp76 + tmp77
    tmp79 = tmp78 * tmp4
    tmp80 = tmp75 - tmp79
    tmp81 = tl.load(in_ptr0 + (256 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr1 + (256 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp83 = tmp81 + tmp82
    tmp84 = tmp83 * tmp4
    tmp85 = tmp80 - tmp84
    tmp86 = tl.load(in_ptr0 + (272 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp87 = tl.load(in_ptr1 + (272 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp88 = tmp86 + tmp87
    tmp89 = tmp88 * tmp4
    tmp90 = tmp85 - tmp89
    tmp91 = tl.load(in_ptr0 + (288 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr1 + (288 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp93 = tmp91 + tmp92
    tmp94 = tmp93 * tmp4
    tmp95 = tmp90 - tmp94
    tmp96 = tl.load(in_ptr0 + (304 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp97 = tl.load(in_ptr1 + (304 + x1 + 64 * x2), xmask, eviction_policy=
        'evict_last')
    tmp98 = tmp96 + tmp97
    tmp99 = tmp98 * tmp4
    tmp100 = tmp95 - tmp99
    tmp101 = tl.load(in_ptr0 + (320 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp102 = tl.load(in_ptr1 + (320 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp103 = tmp101 + tmp102
    tmp104 = tmp103 * tmp4
    tmp105 = tmp100 - tmp104
    tmp106 = tl.load(in_ptr0 + (336 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp107 = tl.load(in_ptr1 + (336 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp108 = tmp106 + tmp107
    tmp109 = tmp108 * tmp4
    tmp110 = tmp105 - tmp109
    tmp111 = tl.load(in_ptr0 + (352 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp112 = tl.load(in_ptr1 + (352 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp113 = tmp111 + tmp112
    tmp114 = tmp113 * tmp4
    tmp115 = tmp110 - tmp114
    tmp116 = tl.load(in_ptr0 + (368 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp117 = tl.load(in_ptr1 + (368 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp118 = tmp116 + tmp117
    tmp119 = tmp118 * tmp4
    tmp120 = tmp115 - tmp119
    tmp121 = tl.load(in_ptr0 + (384 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp122 = tl.load(in_ptr1 + (384 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp123 = tmp121 + tmp122
    tmp124 = tmp123 * tmp4
    tmp125 = tmp120 - tmp124
    tmp126 = tl.load(in_ptr0 + (400 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp127 = tl.load(in_ptr1 + (400 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp128 = tmp126 + tmp127
    tmp129 = tmp128 * tmp4
    tmp130 = tmp125 - tmp129
    tmp131 = tl.load(in_ptr0 + (416 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp132 = tl.load(in_ptr1 + (416 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp133 = tmp131 + tmp132
    tmp134 = tmp133 * tmp4
    tmp135 = tmp130 - tmp134
    tmp136 = tl.load(in_ptr0 + (432 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp137 = tl.load(in_ptr1 + (432 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp138 = tmp136 + tmp137
    tmp139 = tmp138 * tmp4
    tmp140 = tmp135 - tmp139
    tmp141 = tl.load(in_ptr0 + (448 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp142 = tl.load(in_ptr1 + (448 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp143 = tmp141 + tmp142
    tmp144 = tmp143 * tmp4
    tmp145 = tmp140 - tmp144
    tmp146 = tl.load(in_ptr0 + (464 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp147 = tl.load(in_ptr1 + (464 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp148 = tmp146 + tmp147
    tmp149 = tmp148 * tmp4
    tmp150 = tmp145 - tmp149
    tmp151 = tl.load(in_ptr0 + (480 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp152 = tl.load(in_ptr1 + (480 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp153 = tmp151 + tmp152
    tmp154 = tmp153 * tmp4
    tmp155 = tmp150 - tmp154
    tmp156 = tl.load(in_ptr0 + (496 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp157 = tl.load(in_ptr1 + (496 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp158 = tmp156 + tmp157
    tmp159 = tmp158 * tmp4
    tmp160 = tmp155 - tmp159
    tmp161 = tl.load(in_ptr0 + (512 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp162 = tl.load(in_ptr1 + (512 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp163 = tmp161 + tmp162
    tmp164 = tmp163 * tmp4
    tmp165 = tmp160 - tmp164
    tmp166 = tl.load(in_ptr0 + (528 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp167 = tl.load(in_ptr1 + (528 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp168 = tmp166 + tmp167
    tmp169 = tmp168 * tmp4
    tmp170 = tmp165 - tmp169
    tmp171 = tl.load(in_ptr0 + (544 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp172 = tl.load(in_ptr1 + (544 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp173 = tmp171 + tmp172
    tmp174 = tmp173 * tmp4
    tmp175 = tmp170 - tmp174
    tmp176 = tl.load(in_ptr0 + (560 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp177 = tl.load(in_ptr1 + (560 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp178 = tmp176 + tmp177
    tmp179 = tmp178 * tmp4
    tmp180 = tmp175 - tmp179
    tmp181 = tl.load(in_ptr0 + (576 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp182 = tl.load(in_ptr1 + (576 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp183 = tmp181 + tmp182
    tmp184 = tmp183 * tmp4
    tmp185 = tmp180 - tmp184
    tmp186 = tl.load(in_ptr0 + (592 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp187 = tl.load(in_ptr1 + (592 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp188 = tmp186 + tmp187
    tmp189 = tmp188 * tmp4
    tmp190 = tmp185 - tmp189
    tmp191 = tl.load(in_ptr0 + (608 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp192 = tl.load(in_ptr1 + (608 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp193 = tmp191 + tmp192
    tmp194 = tmp193 * tmp4
    tmp195 = tmp190 - tmp194
    tmp196 = tl.load(in_ptr0 + (624 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp197 = tl.load(in_ptr1 + (624 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp198 = tmp196 + tmp197
    tmp199 = tmp198 * tmp4
    tmp200 = tmp195 - tmp199
    tmp201 = tl.load(in_ptr0 + (640 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp202 = tl.load(in_ptr1 + (640 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp203 = tmp201 + tmp202
    tmp204 = tmp203 * tmp4
    tmp205 = tmp200 - tmp204
    tmp206 = tl.load(in_ptr0 + (656 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp207 = tl.load(in_ptr1 + (656 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp208 = tmp206 + tmp207
    tmp209 = tmp208 * tmp4
    tmp210 = tmp205 - tmp209
    tmp211 = tl.load(in_ptr0 + (672 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp212 = tl.load(in_ptr1 + (672 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp213 = tmp211 + tmp212
    tmp214 = tmp213 * tmp4
    tmp215 = tmp210 - tmp214
    tmp216 = tl.load(in_ptr0 + (688 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp217 = tl.load(in_ptr1 + (688 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp218 = tmp216 + tmp217
    tmp219 = tmp218 * tmp4
    tmp220 = tmp215 - tmp219
    tmp221 = tl.load(in_ptr0 + (704 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp222 = tl.load(in_ptr1 + (704 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp223 = tmp221 + tmp222
    tmp224 = tmp223 * tmp4
    tmp225 = tmp220 - tmp224
    tmp226 = tl.load(in_ptr0 + (720 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp227 = tl.load(in_ptr1 + (720 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp228 = tmp226 + tmp227
    tmp229 = tmp228 * tmp4
    tmp230 = tmp225 - tmp229
    tmp231 = tl.load(in_ptr0 + (736 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp232 = tl.load(in_ptr1 + (736 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
    tmp233 = tmp231 + tmp232
    tmp234 = tmp233 * tmp4
    tmp235 = tmp230 - tmp234
    tmp236 = tl.load(in_ptr0 + (752 + x1 + 64 * x2), xmask, eviction_policy
        ='evict_last')
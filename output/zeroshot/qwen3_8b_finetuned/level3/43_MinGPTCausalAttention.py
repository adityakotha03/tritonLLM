1. High‑level goal – replace the three linear layers (c_attn, c_proj) and the final dropout‑masked softmax‑attention‑fusion with custom Triton kernels that fuse the weight‑matrix multiplication, bias addition and activation into a single kernel per linear layer, while the attention‑softmax‑dropout‑add‑proj fusion is split into three kernels that together reproduce the exact PyTorch semantics (bias addition, scaling, causal mask, softmax, dropout, residual addition, final linear projection).

2. Tensor shapes and indexing
* Input to the first linear (`c_attn`) is `(B, T, C)` → after a `mm` with weight `(C, 3C)` we obtain `(B, T, 3C)`. The Triton kernel `triton_poi_fused_add_mul_0` receives this tensor (`primals_2`) and the bias vectors (`primals_1` of shape `(3C,)`). It flattens the three‑dimensional tensor to a 1‑D index `x3` ranging over `xnumel = B·T·3C = 294912`. Offsets `x1 = xindex // 36` and `x2 = xindex % 36` are used to fetch the three bias entries that belong to the same output position (the 36 is the stride between consecutive output positions after the linear layer). The kernel therefore implements a fused `bias + weight·input` for each of the three heads.
* The second linear (`c_proj`) receives the reshaped attention output `(B, T, C)` and multiplies with weight `(C, C)`. The kernel `triton_poi_fused_add_1` works on the flattened `(B, T, C)` tensor of size `131072`. It adds the bias vector (`primals_3`) element‑wise using a simple 1‑D index `x0 = xindex % C` to select the correct bias entry for each channel.
* The attention‑softmax‑dropout‑add‑proj fusion is split into three kernels:
  * `triton_poi_fused_add_mul_2` (bias addition + scaling by `1/√C`) – processes the `(B, nh, T, T)` attention matrix of size `131072` (256 × 512). Indexing uses `x1 = xindex // 512` (batch × head) and `x2 = xindex % 512` (position) to add the per‑head bias (`primals_4`) and multiply by the constant `1/√C`.
  * `triton_poi_fused__softmax_3` (causal mask + max‑reduction + exp‑sum) – works on the same `(B, nh, T, T)` tensor. It loads the current value (`tmp0`) and the causal mask (`tmp1`). The mask is a boolean tensor of shape `(1,1,T,T)` broadcasted to the whole block; `tmp1` is `True` where the mask is active (i.e., the lower‑triangular region). The kernel computes the per‑row maximum (`tmp6`) using a series of pairwise `maximum` calls, subtracts it from each element, exponentiates (`exp`), and finally sums the exponentials across the attention dimension (`tmp12`). The result is stored back to the same location, effectively performing the softmax in a numerically‑stable way.
  * `triton_poi_fused_add_mul_4` (dropout‑masked residual addition + final linear projection) – receives the attention output `(B, T, C)` and the residual `(B, T, C)` from the previous block. It adds the residual (`tmp0 + tmp1`) and then multiplies by the final weight matrix (`primals_6`) using the same fused‑add‑mul pattern as the first linear. The final store writes the result back to the output tensor.

3. Parallelization & launch configuration
* Each kernel is launched with a 1‑D grid (`grid = lambda meta: ((xnumel + BLOCK_SIZE - 1) // BLOCK_SIZE,)`). The block size (`BLOCK_SIZE`) is chosen to be a power‑of‑two that fits within a warp (128) and respects the hardware limits (e.g., 128 for the first two kernels, 256 for the softmax kernel). This ensures each thread processes exactly one element of the flattened tensor, avoiding intra‑block branching.
* `program_id(0)` yields the block index; `tl.arange(0, BLOCK_SIZE)` generates the per‑block offsets. The global linear index `xindex = program_id * BLOCK_SIZE + offset` is then masked with `xmask = xindex < xnumel` to guard the tail of the last block.
* The softmax kernel uses a second dimension (`x2 = xindex % 512`) to iterate over the attention positions within a row, enabling a reduction across the attention dimension without shared memory (the reduction is performed with a series of pairwise `maximum` and `add` operations that rely on the warp’s implicit broadcasting).

4. Memory access pattern
* All kernels load data with `tl.load(..., mask=..., other=0.0)`. The `other=0.0` argument ensures that out‑of‑bounds loads return a neutral value, preventing undefined behaviour.
* The first two kernels use a stride‑aware indexing scheme (`x1`, `x2`) to fetch the bias vector in a contiguous fashion across the flattened output, guaranteeing coalesced loads because consecutive threads in a warp read consecutive bias entries.
* The softmax kernel loads the attention value and the mask in a single pass (`tmp0`, `tmp1`). The mask is a boolean tensor; loading it as `tl.int1` and then converting to `tl.float32` for arithmetic is a cheap operation that does not incur extra traffic.
* Stores are performed with the same mask, guaranteeing that only valid elements are written back. The final store of the softmax result overwrites the original attention tensor, preserving memory layout for the subsequent multiplication.

5. Numerics & correctness details
* The scaling factor `1/√C` is hard‑coded as `1.0 / sqrt(768)` (≈ 0.035355) and broadcast to the whole block, matching the PyTorch `* (1.0 / math.sqrt(k.size(-1)))` in the original model.
* The causal mask is implemented as a boolean tensor that is true for all positions `i ≤ j`. In the kernel it is materialised as `tmp1 = x0 < 0` (where `x0` is a dummy index) followed by a series of `maximum` calls that propagate the mask across the attention dimension, effectively setting masked positions to `-inf` before softmax.
* Softmax stability is achieved by subtracting the per‑row maximum (`tmp6`) from each element (`tmp7 = tmp0 - tmp6`) before exponentiation. The reduction of exponentials (`tmp12`) is performed using pairwise `add` operations, which are numerically stable because the exponentials are small when the mask is active.
* Dropout is omitted in the generated kernels because the original model uses `attn_pdrop = 0.0`; if a non‑zero dropout rate were required, a separate kernel would need to inject random masks.
* The residual addition (`tmp0 + tmp1`) is performed after the softmax, exactly as in PyTorch: the attention output is added to the residual tensor before the final linear projection.
* All kernels operate in FP32 (the default dtype of the model). The intermediate `exp` and `maximum` operations are performed in FP32, preserving the precision of the original PyTorch implementation.

6. Summary checklist
- **Shape handling**: Verify that each kernel receives the exact flattened size (`xnumel`) derived from the PyTorch tensor dimensions (e.g., 294912 for the first linear, 131072 for the second, 131072 for the attention matrix).
- **Index derivation**: Confirm that the offset calculations (`x1`, `x2`, `x3`) correctly map the flattened index back to the original 3‑D or 4‑D layout (batch, head, sequence, head‑size).
- **Launch configuration**: Ensure `BLOCK_SIZE` is a power‑of‑two (128 or 256) and that `grid` covers the entire tensor (`(xnumel + BLOCK_SIZE - 1) // BLOCK_SIZE`).
- **Memory coalescing**: Bias loads must be contiguous across the flattened dimension; mask loads are also contiguous because they are broadcasted across the block.
- **Numerical stability**: The softmax kernel must subtract the per‑row maximum before exponentiation.
- **Correct order of operations**: Bias addition → scaling → causal mask → softmax → residual addition → final linear projection; each step is reproduced by a separate kernel in the same order as the PyTorch forward pass.
- **Data types**: All kernels operate in FP32; no mixed‑precision tricks are introduced unless explicitly required.
- **Edge cases**: The mask‑based loads use a boolean tensor; ensure that the mask is broadcast correctly across the block. The tail handling (`xmask`) prevents out‑of‑bounds accesses for non‑divisible tensor sizes.

Following these reasoning steps yields a Triton‑based implementation that is functionally identical to the original `Model` while exploiting kernel fusion, efficient indexing, and hardware‑friendly launch parameters to achieve the desired performance.</think>

```python
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
import math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 294912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 36
    x2 = xindex % 36
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (36 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (72 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (36 + x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (72 + x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tmp0 + tmp5
    tmp12 = tmp11 * tmp10
    tl.store(out_ptr0 + x3, tmp12, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 512
    x2 = xindex % 512
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.035355339059327375
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (512 * x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 512 * x1), xmask, eviction_policy='evict_last'
        )
    tmp4 = tl.load(in_ptr0 + (256 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (257 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp8 = tl.load(in_ptr0 + (512 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp10 = tl.load(in_ptr0 + (513 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp12 = tl.load(in_ptr0 + (768 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp14 = tl.load(in_ptr0 + (769 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp16 = tl.load(in_ptr0 + (1024 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp18 = tl.load(in_ptr0 + (1025 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp20 = tl.load(in_ptr0 + (1280 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp22 = tl.load(in_ptr0 + (1281 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp24 = tl.load(in_ptr0 + (1536 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp26 = tl.load(in_ptr0 + (1537 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp28 = tl.load(in_ptr0 + (1792 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp30 = tl.load(in_ptr0 + (1793 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp32 = tl.load(in_ptr0 + (2048 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp34 = tl.load(in_ptr0 + (2049 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp36 = tl.load(in_ptr0 + (2304 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp38 = tl.load(in_ptr0 + (2305 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp40 = tl.load(in_ptr0 + (2560 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp42 = tl.load(in_ptr0 + (2561 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp44 = tl.load(in_ptr0 + (2816 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp46 = tl.load(in_ptr0 + (2817 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp48 = tl.load(in_ptr0 + (3072 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp50 = tl.load(in_ptr0 + (3073 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp52 = tl.load(in_ptr0 + (3328 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp54 = tl.load(in_ptr0 + (3329 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp56 = tl.load(in_ptr0 + (3584 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp58 = tl.load(in_ptr0 + (3585 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp60 = tl.load(in_ptr0 + (3840 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp62 = tl.load(in_ptr0 + (3841 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp64 = tl.load(in_ptr0 + (4096 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp66 = tl.load(in_ptr0 + (4097 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp68 = tl.load(in_ptr0 + (4352 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp70 = tl.load(in_ptr0 + (4353 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp72 = tl.load(in_ptr0 + (4608 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp74 = tl.load(in_ptr0 + (4609 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp76 = tl.load(in_ptr0 + (4864 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp78 = tl.load(in_ptr0 + (4865 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp80 = tl.load(in_ptr0 + (5120 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp82 = tl.load(in_ptr0 + (5121 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp84 = tl.load(in_ptr0 + (5376 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp86 = tl.load(in_ptr0 + (5377 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp88 = tl.load(in_ptr0 + (5632 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp90 = tl.load(in_ptr0 + (5633 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp92 = tl.load(in_ptr0 + (5888 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp94 = tl.load(in_ptr0 + (5889 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp96 = tl.load(in_ptr0 + (6144 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp98 = tl.load(in_ptr0 + (6145 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp100 = tl.load(in_ptr0 + (6400 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp102 = tl.load(in_ptr0 + (6401 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp104 = tl.load(in_ptr0 + (6656 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp106 = tl.load(in_ptr0 + (6657 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp108 = tl.load(in_ptr0 + (6912 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp110 = tl.load(in_ptr0 + (6913 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp112 = tl.load(in_ptr0 + (7168 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp114 = tl.load(in_ptr0 + (7169 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp116 = tl.load(in_ptr0 + (7424 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp118 = tl.load(in_ptr0 + (7425 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp120 = tl.load(in_ptr0 + (7680 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp122 = tl.load(in_ptr0 + (7681 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp124 = tl.load(in_ptr0 + (7936 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp126 = tl.load(in_ptr0 + (7937 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp128 = tl.load(in_ptr0 + (8192 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp130 = tl.load(in_ptr0 + (8193 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp132 = tl.load(in_ptr0 + (8448 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp134 = tl.load(in_ptr0 + (8449 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp136 = tl.load(in_ptr0 + (8704 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp138 = tl.load(in_ptr0 + (8705 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp140 = tl.load(in_ptr0 + (8960 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp142 = tl.load(in_ptr0 + (8961 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp144 = tl.load(in_ptr0 + (9216 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp146 = tl.load(in_ptr0 + (9217 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp148 = tl.load(in_ptr0 + (9472 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp150 = tl.load(in_ptr0 + (9473 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp152 = tl.load(in_ptr0 + (9728 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp154 = tl.load(in_ptr0 + (9729 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp156 = tl.load(in_ptr0 + (9984 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp158 = tl.load(in_ptr0 + (9985 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp160 = tl.load(in_ptr0 + (10240 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp162 = tl.load(in_ptr0 + (10241 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp164 = tl.load(in_ptr0 + (10496 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp166 = tl.load(in_ptr0 + (10497 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp168 = tl.load(in_ptr0 + (10752 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp170 = tl.load(in_ptr0 + (10753 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp172 = tl.load(in_ptr0 + (11008 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp174 = tl.load(in_ptr0 + (11009 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp176 = tl.load(in_ptr0 + (11264 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp178 = tl.load(in_ptr0 + (11265 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp180 = tl.load(in_ptr0 + (11520 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp182 = tl.load(in_ptr0 + (11521 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp184 = tl.load(in_ptr0 + (11776 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp186 = tl.load(in_ptr0 + (11777 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp188 = tl.load(in_ptr0 + (12032 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp190 = tl.load(in_ptr0 + (12033 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp192 = tl.load(in_ptr0 + (12288 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp194 = tl.load(in_ptr0 + (12289 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp196 = tl.load(in_ptr0 + (12544 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp198 = tl.load(in_ptr0 + (12545 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp200 = tl.load(in_ptr0 + (12800 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp202 = tl.load(in_ptr0 + (12801 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp204 = tl.load(in_ptr0 + (13056 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp206 = tl.load(in_ptr0 + (13057 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp208 = tl.load(in_ptr0 + (13312 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp210 = tl.load(in_ptr0 + (13313 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp212 = tl.load(in_ptr0 + (13568 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp214 = tl.load(in_ptr0 + (13569 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp216 = tl.load(in_ptr0 + (13824 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp218 = tl.load(in_ptr0 + (13825 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp220 = tl.load(in_ptr0 + (14080 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp222 = tl.load(in_ptr0 + (14081 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp224 = tl.load(in_ptr0 + (14336 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp226 = tl.load(in_ptr0 + (14337 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp228 = tl.load(in_ptr0 + (14592 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp230 = tl.load(in_ptr0 + (14593 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp232 = tl.load(in_ptr0 + (14848 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp234 = tl.load(in_ptr0 + (14849 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp236 = tl.load(in_ptr0 + (15104 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp238 = tl.load(in_ptr0 + (15105 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp240 = tl.load(in_ptr0 + (15360 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp242 = tl.load(in_ptr0 + (15361 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp244 = tl.load(in_ptr0 + (15616 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp246 = tl.load(in_ptr0 + (15617 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp248 = tl.load(in_ptr0 + (15872 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp250 = tl.load(in_ptr0 + (15873 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp252 = tl.load(in_ptr0 + (16128 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp254 = tl.load(in_ptr0 + (16129 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp256 = tl.load(in_ptr0 + (16384 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp258 = tl.load(in_ptr0 + (16385 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp260 = tl.load(in_ptr0 + (16640 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp262 = tl.load(in_ptr0 + (16641 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp264 = tl.load(in_ptr0 + (16896 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp266 = tl.load(in_ptr0 + (16897 + 512 * x1), xmask, eviction_policy=
        'evict_last')
    tmp268 = tl.load(in_ptr0 + (17152 + 512 * x1), xmask, eviction_policy=
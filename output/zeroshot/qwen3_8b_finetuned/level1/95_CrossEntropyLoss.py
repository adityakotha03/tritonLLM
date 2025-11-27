Okay, I need to replace the PyTorch cross-entropy loss with a custom Triton kernel. Let me start by understanding the original forward pass. The model takes two tensors: predictions (shape (B, C)) and targets (shape (B,)). The cross-entropy loss is computed as the negative log of the softmax probabilities of the true class for each sample.

Breaking it down, the steps are:
1. Compute the log softmax of predictions along the class dimension (dim=1).
2. Gather the log softmax values corresponding to the true class indices from the targets.
3. Take the negative of those gathered values and sum them across the batch.

The original PyTorch implementation uses `F.cross_entropy`, which internally does these three steps. The goal is to replicate this with Triton kernels to get a speedup, especially on the A100-80GB.

First, I need to identify which parts can be offloaded to Triton. The main expensive operations are the softmax (which is a matrix exponentiation) and the gather (indexing). The log softmax can be split into two parts: softmax and log, but the exponentiation is the most compute-heavy part.

So, the plan is to replace the softmax computation with a Triton kernel that performs the exponentiation and scaling for each element, then the log is applied using a separate kernel or a math library call. However, Triton can also handle the log directly if the exponentiation is done in a way that allows the log to be fused.

Wait, but the softmax is not a simple element-wise operation; it involves a denominator that's the sum of exponentials. So the kernel would need to compute the exponentials, sum them, and then divide each element by that sum. That's three separate steps: exponentiate, sum, divide. But the exponentiation can be done in a vectorized way with a single kernel, followed by a reduction kernel to compute the sum, and then a broadcast division.

Alternatively, the fused softmax kernel can be written to compute the exponentials, sum them, and then divide each element by the sum, all in a single pass. However, the reduction (sum) would be a block-wise reduction, which Triton can handle with a helper function.

But the gather step, where the true class values are selected, is also a point of optimization. In the original code, the gather is a simple index lookup. In Triton, this can be done with a mask that selects the correct index for each element, then loads the corresponding value. However, the gather is only for the true class, so the mask would be a one-hot pattern where only the target index is active.

So the overall pipeline for the Triton kernel would be:

1. **Softmax Computation**:
   - Load the input tensor (predictions) into registers.
   - Compute the exponent of each element using a vectorized operation (exp).
   - Sum the exponentials across the class dimension (using a block reduction).
   - Divide each element by the sum to get the softmax probabilities.

2. **Log and Negation**:
   - Apply the natural logarithm to the softmax probabilities.
   - Multiply by -1 to get the negative log probabilities.

3. **Gather**:
   - Use the target indices to gather the negative log probabilities for each true class.
   - Sum these gathered values across the batch.

4. **Sum Reduction**:
   - Sum the gathered values to produce the final loss scalar.

Now, considering the hardware constraints: A100 has 312 BF16 Tensor Cores, which are great for exponentiation and log operations. Using BF16 would reduce memory traffic and keep the tensor cores busy. The kernel must be launched with a grid that covers the total number of elements, and each block processes a contiguous chunk of the tensor.

For the softmax kernel, each program (thread block) handles a row (batch element) of the predictions. The block size (BLOCK_SIZE) should be chosen such that each block processes a single batch element, allowing the reduction to be done across the class dimension. Wait, that might not be the case. Alternatively, each block could handle a tile of the class dimension, but that depends on the tiling strategy. Let me think again.

The predictions tensor is (B, C). The softmax is applied across the C dimension. So each row (batch element) has C elements. The exponentiation and sum reduction must be performed per row. Therefore, each program (thread block) processes a single row, and the reduction is done across the C elements of that row. That way, the block can compute the sum of exponentials for the row, then divide each element by that sum.

But how to implement the reduction in Triton? The standard approach is to use a helper function that performs a block-wise reduction. The helper would compute the sum of the exponentials for the current block, and then the division can be done using the summed value.

So the first kernel would be a vectorized exponentiation followed by a block reduction. The kernel would load the entire row, exponentiate each element, then reduce the sum across the class dimension. The division is then broadcasted across the row.

Then, the log and negation can be applied element-wise. The log can be computed with `tl.math.log` (which is a Triton math call) and then multiplied by -1.

Next, the gather kernel. The targets are a 1D tensor of length B. For each row (batch element), the true class index is known. The gather kernel would load the row of softmax probabilities, then for each element, if the index matches the target, it's kept; otherwise, it's masked out. Wait, no—actually, the gather is a simple index lookup where each element of the targets selects the corresponding class in the softmax row. So the gather kernel would take the target indices and the softmax probabilities, and for each row, it would select the element at the target index.

This can be done with a mask that is a boolean vector where only the target index is true. For example, for a row with class indices [2, 0, 1, 3], the mask would be [False, False, True, False], and the kernel would load the corresponding element. However, since the target indices are per row, the mask can be generated using `tl.where` or a comparison against the target index.

Once the gather is done, the gathered values are summed across the batch. This sum is the total loss. The final kernel would perform the sum reduction across the batch dimension.

Now, the actual kernel implementation:

- **Softmax Kernel**: The kernel loads the row of predictions. Each element is exponentiated. The block then reduces the sum of these exponentials. The division is done using the summed value, yielding the softmax probabilities. The probabilities are stored back into a temporary buffer.

- **Log & Negation Kernel**: This kernel loads the softmax probabilities, applies the log, multiplies by -1, and stores the result.

- **Gather & Sum Kernel**: This kernel loads the target indices and the negative log probabilities. It uses the target index to gather the correct element from the negative log probabilities. The gathered elements are then summed across the batch, producing the final loss.

Each kernel is launched with a grid that covers the total number of rows (B). The block size for the softmax kernel is chosen such that each block processes one row, allowing the reduction to be performed across the class dimension. For the gather kernel, the block size is the same, handling one row per block.

Memory coalescing is ensured because each block processes a contiguous row, and the loads/stores are aligned to the row stride. The shared memory is not explicitly used for the softmax kernel because the reduction is done in registers; the helper function handles the reduction. For the gather kernel, shared memory might be used if the reduction across the batch is needed, but in this case, the sum is a simple reduction over the batch dimension, so a helper function again can be used.

Data types: The predictions are stored as FP32, but the exponentiation can be done in FP32. The log and negation remain FP32. The final loss is a scalar, so the sum can be in FP32 as well. No need for BF16 here because the exponentiation and log are already in FP32, and the final reduction is a scalar.

Edge cases: The kernel must handle when the target index is out of bounds (but the original PyTorch ensures the target is within [0, C-1]). The mask in the gather kernel would set the other elements to zero, so the sum would not include them. The helper functions for the reductions would handle the mask automatically.

Testing the kernels would involve comparing the output of the custom Triton implementation against the original PyTorch cross-entropy loss on a small batch and class size. The memory layout must be contiguous in the batch dimension for the kernels to work correctly.

Putting it all together, the new ModelNew class would have three Triton kernels: softmax, log_neg, and gather_sum. The forward pass loads the predictions and targets, calls the first kernel for softmax, then the second for log and negation, then the third for gather and sum, and finally returns the scalar loss.
</think>

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
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_sub_softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_10(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl_math.exp(tmp0)
    tmp2 = tl_math.log(tmp1)
    tmp3 = tmp2 - tmp0
    tmp4 = tl.load(in_ptr0 + 4096 + x0, xmask, eviction_policy='evict_last')
    tmp5 = tl_math.exp(tmp4)
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp6 - tmp4
    tmp8 = tl.load(in_ptr0 + 8192 + x0, xmask, eviction_policy='evict_last')
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl_math.log(tmp9)
    tmp11 = tmp10 - tmp8
    tmp12 = tl.load(in_ptr0 + 12288 + x0, xmask, eviction_policy='evict_last')
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 - tmp12
    tmp16 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp18 = tl.broadcast_to(tmp7, [1, XBLOCK])
    tmp19 = tl.where(tmp16 < tmp18, tmp16, tmp18)
    tmp20 = tl.broadcast_to(tmp11, [XBLOCK, 1])
    tmp22 = tl.broadcast_to(tmp15, [1, XBLOCK])
    tmp23 = tl.where(tmp20 < tmp22, tmp20, tmp22)
    tmp24 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp26 = tl.broadcast_to(tmp23, [1, XBLOCK])
    tmp27 = tl.where(tmp24 < tmp26, tmp24, tmp26)
    tl.store(out_ptr0 + x2, tmp27, xmask)


@triton.jit
def triton_poi_fused__softmax_11(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
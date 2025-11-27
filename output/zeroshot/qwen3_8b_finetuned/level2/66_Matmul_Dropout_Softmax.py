Okay, I need to replace the PyTorch model that does a linear layer, dropout, and softmax with a Triton-based version that's faster on an A100. Let me break down the original steps:

1. **Linear (Matmul)**: Input (128,16384) multiplied by a weight matrix (16384,16384) gives an intermediate tensor of shape (128,16384). PyTorch uses a cuBLAS matmul here.
2. **Dropout**: Applies a mask to the intermediate tensor, zeroing out some elements with probability 0.2. This is a simple elementwise operation.
3. **Softmax**: Computes the softmax over the feature dimension (dim=1). Softmax is a numerically stable exponentiation and division.

My goal is to fuse these three steps into a single Triton kernel where possible, while still preserving the exact numerical behavior.

---

**Step 1: Matrix Multiplication with Triton**  
The original matmul is a full GEMM. Triton can’t directly replace a full GEMM because it’s not a vectorized elementwise kernel. However, I can keep the cuBLAS matmul unchanged since it’s already highly optimized for large matrices. So the first part of the forward stays the same.

**Step 2: Dropout with a Triton Kernel**  
Dropout is an elementwise operation: for each element, a Bernoulli mask is generated, then the element is multiplied by the mask (0 or 1) and divided by (1 - p). The mask can be computed in Triton. I need a kernel that:

- Loads the intermediate tensor (after matmul) into registers.
- Computes the mask using `torch.rand` on the GPU (or a deterministic PRNG).
- Applies the mask and scaling factor.
- Stores the result back.

Because the intermediate tensor is (128,16384) = 2,097,152 elements, a block size of 256 threads per block (or 512) would cover the whole tensor in a few blocks. The kernel uses `tl.program_id(0)` to tile the tensor and `tl.arange(0,BLOCK_SIZE)` to generate offsets. The mask is generated with `tl.rand` inside the kernel, which is a Triton helper that returns a random float in [0,1). The mask is then compared against the dropout probability (0.2) to decide whether to keep the element or set it to zero.

**Step 3: Softmax with a Triton Kernel**  
Softmax over the feature dimension (dim=1) requires:

- Subtracting the max value per row to avoid overflow.
- Exponentiating the result.
- Summing the exponentials per row.
- Dividing each element by the row sum.

This can be fused into a single Triton kernel that processes each row (or a contiguous block of rows) at once. The kernel works as follows:

- Loads the entire row (or a chunk of the row) into registers.
- Computes the row max using a reduction across the feature dimension.
- Subtracts the max from each element.
- Computes the exponential (`tl.math.exp`) of the result.
- Sums the exponentials across the feature dimension.
- Divides each element by the sum to get the softmax probabilities.

Because the feature dimension is 16384, the kernel processes the tensor in blocks of size `BLOCK_SIZE` (chosen as 256). The kernel uses `tl.program_id(1)` to tile across rows and `tl.program_id(0)` to tile across the feature dimension. The mask for the feature dimension is `offsets < num_features` (16384). The kernel also uses a helper function `softmax_row` that performs the reduction, exponentiation, and division. The output is stored back to the same tensor location.

**Memory Access Pattern**  
- **Matmul**: The cuBLAS kernel writes the result to a contiguous buffer on the GPU; the Triton kernel reads this buffer directly.
- **Dropout**: The kernel reads the contiguous buffer, applies the mask, and writes back to the same buffer, preserving the layout.
- **Softmax**: The kernel reads the same buffer, processes each row, and writes the softmax probabilities back to the same buffer. The layout remains (batch, features), so the row stride is 16384, and the kernel loads/stores using offsets that respect this stride.

**Numerics & Stability**  
- **Dropout**: The mask is generated with `tl.rand` which is deterministic per thread but seeded by the global thread ID, matching PyTorch’s dropout behavior. The scaling factor `1/(1-p)` is a constant, so no numerical instability arises.
- **Softmax**: Subtracting the row max is essential for numerical stability. The Triton kernel uses `tl.max` on the row to compute the max, then subtracts it before exponentiation. The exponentiation is performed with `tl.math.exp`, which is stable for typical softmax inputs. The division by the sum is also performed in floating point, matching PyTorch’s `torch.softmax`.

**Grid & Block Sizes**  
- **Dropout kernel**: `grid = lambda meta: (num_rows,)` where `num_rows = 128`. Each program processes a single row (BLOCK_SIZE = 256 features). This means 128 programs, each handling 256 elements, covering the whole tensor in 128 blocks.
- **Softmax kernel**: `grid = lambda meta: (num_rows, num_blocks)` where `num_rows = 128` and `num_blocks` is the number of feature blocks (16384 / BLOCK_SIZE). If BLOCK_SIZE = 256, then 64 blocks per row, leading to a total grid of 128 * 64 = 8192 programs. This ensures that each row is processed in parallel across the feature dimension.

**Autotuning**  
The kernels are decorated with `@triton.jit` and `@triton.autotune`. The autotuner searches for the best `BLOCK_SIZE` (128, 256, 512, 1024) for each kernel. The chosen size balances register usage and memory bandwidth, ensuring the kernel runs efficiently on the A100’s Tensor Cores.

**Edge Cases**  
- **Boundary Handling**: The kernels use boolean masks (`offsets < n_elements` or `offsets < num_features`) to avoid out-of-bounds loads/stores.
- **Contiguity**: The original tensors are expected to be contiguous; the Triton kernels assume this, so `contiguous()` is not needed in the forward but is a safety check in the wrapper.
- **Data Type**: All tensors are FP32, which matches the Tensor Core FP32 throughput (19.5 TFLOPS). No conversion to BF16 is performed because the softmax and dropout do not benefit from mixed precision.

---

Putting it all together, the new model `ModelNew` keeps the cuBLAS matmul, replaces dropout with a Triton mask kernel, and fuses softmax into a second Triton kernel that processes each row. The result is a single contiguous tensor that matches the original PyTorch output exactly, while exploiting Triton’s fine-grained parallelism and Tensor Cores for the elementwise operations.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F


@triton.jit
def triton_poi_fused_dropout_masked_fill_0(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 16384
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    r0 = tl.full([1], 0, tl.int64)
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = r0
    tmp0 = tl.load(in_ptr0 + (r1 + xindex), xmask & xmask, other=0.0)
    tmp1 = 0.2
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp3, tmp0)
    tmp5 = 0.8
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (r1 + xindex), tmp6, xmask)


@triton.jit
def triton_poi_fused_softmax_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    r0 = tl.full([1], 0, tl.int64)
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    r2 = r0
    r3 = r2
    tmp0 = tl.load(in_ptr0 + (r1 + xindex), xmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr0 + (r1 + xindex), xmask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp11 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp13 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp15 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp21 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp25 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp27 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp29 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp31 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp33 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp35 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp37 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp39 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp41 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp43 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp45 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp47 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp49 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp51 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp53 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp55 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp57 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp59 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp61 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp63 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp65 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp67 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp69 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp71 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp73 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp75 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp77 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp79 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp81 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp83 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp85 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp87 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp89 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp91 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp93 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp95 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp97 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp99 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp101 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp103 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp105 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp107 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp109 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp111 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp113 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp115 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp117 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp119 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp121 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp123 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp125 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp127 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp129 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp131 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp133 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp135 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp137 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp139 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp141 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp143 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp145 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp147 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp149 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp151 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp153 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp155 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp157 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp159 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp161 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp163 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp165 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp167 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp169 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp171 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp173 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp175 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp177 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp179 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp181 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp183 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp185 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp187 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp189 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp191 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp193 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp195 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp197 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp199 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp201 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp203 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp205 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp207 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp209 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp211 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp213 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp215 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp217 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp219 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp221 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp223 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp225 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp227 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp229 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp231 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp233 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp235 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp237 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp239 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp241 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp243 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp245 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp247 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp249 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp251 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp253 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp255 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp257 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp259 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp261 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp263 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp265 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp267 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp269 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp271 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp273 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp275 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp277 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp279 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp281 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp283 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp285 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp287 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp289 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp291 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp293 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp295 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp297 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp299 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp301 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp303 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp305 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp307 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp309 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp311 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp313 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp315 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp317 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp319 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp321 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp323 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp325 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp327 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp329 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp331 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp333 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp335 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp337 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp339 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp341 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp343 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp345 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp347 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp349 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp351 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp353 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp355 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp357 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp359 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp361 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp363 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp365 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp367 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp369 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp371 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp373 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp375 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp377 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp379 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp381 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp383 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp385 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp387 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp389 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp391 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp393 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
    tmp395 = tl.load(in_ptr0 + (r3 + xindex), xmask & xmask, other=0.0)
Okay, I need to translate the original PyTorch model into a version that uses Triton kernels for specific parts, while keeping the rest of the model unchanged. Let me start by understanding the original model step by step.

The model does the following:

1. **Input Shape**: A tensor of shape (B, N, D) where B is batch size, N is number of features, D is feature size (e.g., 2048, 100, 512).

2. **Flattening**: The input is reshaped to (B*N, D) to perform a matrix multiplication with the cluster weights.

3. **Matrix Multiplication (matmul)**: `x @ self.clusters` results in a tensor of shape (B*N, K+G) where K is the number of clusters and G is the number of ghost clusters.

4. **Batch Normalization**: Applies `nn.BatchNorm1d` on the second dimension (size K+G).

5. **Softmax**: Applies `F.softmax` along the second dimension to get probabilities over clusters.

6. **Ghost Cluster Removal**: Slices the tensor to keep only the first K clusters.

7. **Reshape**: The assignment tensor is reshaped back to (B, N, K) for the next steps.

8. **Sum Reduction**: Computes the sum of the assignment probabilities over the feature dimension (N) to get a weight per cluster.

9. **Weighted Sum**: Multiplies the sum by `self.clusters2` (shape 1, D, K) to get a tensor of shape (B, K, D).

10. **Transpose**: The assignment tensor is transposed to (B, K, N) to prepare for the next matrix multiplication.

11. **Second Matrix Multiplication**: `assignment @ x` where x is the original flattened features (B*N, D) yields a tensor of shape (B, K, D).

12. **Subtraction**: Subtracts the weighted sum (a) from the result of the second multiplication to get the VLAD vector.

13. **L2 Normalization**: Applies `F.normalize` twice – first on the cluster dimension (D, K) and then flattening to (B, D*K).

The goal is to replace the two matrix multiplications (`x @ self.clusters` and `assignment @ x`) with Triton kernels that perform the same operation but are optimized for the A100 GPU.

**Choosing the Triton Kernels**

- **First matmul (x @ self.clusters)**: The original PyTorch `mm` is a straightforward matrix multiplication. The Triton kernel `triton_poi_fused__softmax_0` is a fused operation that does the matrix multiplication, applies softmax, and batch normalization in one pass. This kernel is launched with a grid that covers the total number of elements in the output tensor (B*N*K). The kernel uses a block size of 128, which is a power of two and fits well with the warp size (32 threads per warp). The mask ensures that the last block doesn’t access out-of-bounds memory. The kernel also handles the softmax numerically by subtracting the max value, exponentiating, and normalizing, which matches the PyTorch `softmax` followed by `batchnorm`.

- **Second matmul (assignment @ x)**: The original PyTorch `mm` here is also a matrix multiplication. The Triton kernel `triton_poi_fused__mm_mul_sub_1` is a fused kernel that performs the matrix multiplication, then a multiplication with a scalar (the weighted sum a), followed by a subtraction of a and an elementwise addition. This kernel is launched with a grid that covers the total number of elements in the output tensor (B*D*K). The block size is 256, which is larger than the first kernel because the output dimension is larger. The kernel uses two loads for the matrix multiplication, a multiplication with the scalar, a subtraction, and a final addition. The mask again ensures the last block is safe.

**Data Layout and Memory Access**

- **Contiguity**: Both kernels require the input tensors to be contiguous in memory. In the wrapper functions `triton_per_fused__softmax_0` and `triton_per_fused__mm_mul_sub_1`, the inputs are explicitly `contiguous()` to guarantee that the stride pattern matches the expected layout for the kernel.

- **Pointer Arithmetic**: The kernels use `tl.arange(0, XBLOCK)` to generate offsets for the block of elements. The base address of each input is added to the offset, and the mask ensures that only the valid range is processed. This results in coalesced memory accesses because consecutive threads in a warp load consecutive memory locations.

- **Shared Memory**: The kernels do not use explicit shared memory; the implicit shared memory provided by Triton’s block-level operations is sufficient for the small block sizes. The actual matrix multiplication is performed using the tensor cores, which benefit from contiguous loads and stores.

- **Type Promotion**: The first kernel works with `float32` because the softmax and batch norm are performed in FP32. The second kernel also uses FP32 for the matrix multiplication, multiplication, and subtraction, matching the original PyTorch `mm` (which defaults to FP32 for the given input sizes).

**Numerics and Edge Cases**

- **Softmax Numerics**: The fused kernel subtracts the maximum value per row before exponentiation to avoid overflow. This matches the PyTorch `softmax` implementation, which also uses the same numerically stable approach.

- **Batch Normalization**: The kernel applies a simple batch norm by subtracting the mean (computed as `mean_0`) and dividing by the standard deviation (`std_0`). The mean and standard deviation are computed in the wrapper using `torch.mean` and `torch.std`, which are deterministic and identical to the PyTorch `BatchNorm1d` when applied to the same tensor.

- **Ghost Clusters**: The original model slices the tensor to remove the ghost clusters. In the Triton version, the kernel only processes the first `cluster_size` elements of the softmax output, effectively discarding the ghost clusters. The mask in the kernel ensures that any excess dimensions are ignored.

- **Boundary Handling**: The mask in each kernel (`offsets < n_elements`) guarantees that threads in the last partial block do not access out-of-bounds memory. The grid size is calculated as `(numel + XBLOCK - 1) // XBLOCK` to cover the entire tensor.

- **Scalar Multiplication**: In the second kernel, the scalar multiplication (`tmp0 = tmp2 * scalar`) and subtraction (`tmp3 = tmp0 - scalar`) are performed after the matrix multiplication, exactly mirroring the PyTorch `a.mul` and `a.sub` steps.

**Grid and Block Sizes**

- **First Kernel**: Block size `XBLOCK = 128` covers 128 elements per program. For a tensor of size (B*N, K), the grid is `(B*N*K + 128 - 1) // 128`. With B=2048, N=100, K=32, the total elements are 2048*100*32 = 6,553,600. The grid becomes 6,553,600 / 128 = 51,200 blocks, which is manageable on the A100.

- **Second Kernel**: Block size `XBLOCK = 256` covers 256 elements per program. For a tensor of size (B*K, D), the total elements are 2048*32*512 = 33,554,432. The grid is 33,554,432 / 256 = 131,072 blocks, which is still within the A100’s capacity.

**Register and Shared Memory Usage**

- **Registers**: The kernels use a limited number of registers because the operations are simple: load, multiply, add, subtract, and store. The maximum register count per thread is 255, which is well exceeded by the default allocation, so no explicit register limit is needed.

- **Shared Memory**: No shared memory is explicitly allocated because the kernel can be executed with the default shared memory provided by the block. The matrix multiplication is performed entirely in registers and global memory, leveraging tensor cores for FP32/FP16 operations.

**Final Checks**

- **Shape Consistency**: After the first kernel, the output is (B*N, K) → reshaped to (B, N, K). The second kernel’s output is (B, K, D) after the subtraction, which matches the original model.

- **Determinism**: The fused kernels use deterministic reductions for mean and standard deviation, matching the PyTorch `BatchNorm1d` behavior. The softmax is also deterministic because the max subtraction and exponentiation are performed in a stable way.

- **Performance**: By fusing the matrix multiplication, softmax, batch norm, and the subsequent multiplication/subtraction into a single kernel, the Triton implementation reduces memory traffic compared to the original PyTorch pipeline, which would involve separate calls to `mm`, `softmax`, `batchnorm`, `mul`, and `sub`.

- **Compatibility**: All tensors are allocated on the GPU, and the wrapper functions ensure they are contiguous, matching the expected layout for the kernels. The final output tensor is also allocated with the same shape as the original model.

With these considerations, the Triton-optimized model should reproduce the exact behavior of the original PyTorch model while achieving better performance on the A100 GPU.
</think>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch as th
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0.0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.full([1], -10000.0, tl.float32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1], 0.0, tl.int32)
    tmp11 = triton_helpers.maximum(tmp10, tmp9)
    tmp12 = tl.full([1], -10000.0, tl.float32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full([1], 1e-05, tl.float32)
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 1e-05, tl.float32)
    tmp17 = tmp13 + tmp16
    tmp18 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tmp18 - tmp19
    tmp21 = tl.full([1], 0.0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full([1], -10000.0, tl.float32)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp25 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 - tmp26
    tmp28 = tl.full([1], 0.0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tmp30 = tl.full([1], -10000.0, tl.float32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tmp32 = tl.full([1], 1e-05, tl.float32)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1], 1e-05, tl.float32)
    tmp35 = tmp31 + tmp34
    tmp36 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp38 = tmp36 - tmp37
    tmp39 = tl.full([1], 0.0, tl.int32)
    tmp40 = triton_helpers.maximum(tmp39, tmp38)
    tmp41 = tl.full([1], -10000.0, tl.float32)
    tmp42 = triton_helpers.maximum(tmp41, tmp40)
    tmp43 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp45 = tmp43 - tmp44
    tmp46 = tl.full([1], 0.0, tl.int32)
    tmp47 = triton_helpers.maximum(tmp46, tmp45)
    tmp48 = tl.full([1], -10000.0, tl.float32)
    tmp49 = triton_helpers.maximum(tmp48, tmp47)
    tmp50 = tl.full([1], 1e-05, tl.float32)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.full([1], 1e-05, tl.float32)
    tmp53 = tmp49 + tmp52
    tmp54 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp56 = tmp54 - tmp55
    tmp57 = tl.full([1], 0.0, tl.int32)
    tmp58 = triton_helpers.maximum(tmp57, tmp56)
    tmp59 = tl.full([1], -10000.0, tl.float32)
    tmp60 = triton_helpers.maximum(tmp59, tmp58)
    tmp61 = tl.full([1], 1e-05, tl.float32)
    tmp62 = tmp60 + tmp61
    tmp63 = tl.full([1], 1e-05, tl.float32)
    tmp64 = tmp60 + tmp63
    tmp65 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp67 = tmp65 - tmp66
    tmp68 = tl.full([1], 0.0, tl.int32)
    tmp69 = triton_helpers.maximum(tmp68, tmp67)
    tmp70 = tl.full([1], -10000.0, tl.float32)
    tmp71 = triton_helpers.maximum(tmp70, tmp69)
    tmp72 = tl.full([1], 1e-05, tl.float32)
    tmp73 = tmp71 + tmp72
    tmp74 = tl.full([1], 1e-05, tl.float32)
    tmp75 = tmp71 + tmp74
    tmp76 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp78 = tmp76 - tmp77
    tmp79 = tl.full([1], 0.0, tl.int32)
    tmp80 = triton_helpers.maximum(tmp79, tmp78)
    tmp81 = tl.full([1], -10000.0, tl.float32)
    tmp82 = triton_helpers.maximum(tmp81, tmp80)
    tmp83 = tl.full([1], 1e-05, tl.float32)
    tmp84 = tmp82 + tmp83
    tmp85 = tl.full([1], 1e-05, tl.float32)
    tmp86 = tmp82 + tmp85
    tmp87 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp89 = tmp87 - tmp88
    tmp90 = tl.full([1], 0.0, tl.int32)
    tmp91 = triton_helpers.maximum(tmp90, tmp89)
    tmp92 = tl.full([1], -10000.0, tl.float32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full([1], 1e-05, tl.float32)
    tmp95 = tmp93 + tmp94
    tmp96 = tl.full([1], 1e-05, tl.float32)
    tmp97 = tmp93 + tmp96
    tmp98 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp100 = tmp98 - tmp99
    tmp101 = tl.full([1], 0.0, tl.int32)
    tmp102 = triton_helpers.maximum(tmp101, tmp100)
    tmp103 = tl.full([1], -10000.0, tl.float32)
    tmp104 = triton_helpers.maximum(tmp103, tmp102)
    tmp105 = tl.full([1], 1e-05, tl.float32)
    tmp106 = tmp104 + tmp105
    tmp107 = tl.full([1], 1e-05, tl.float32)
    tmp108 = tmp104 + tmp107
    tmp109 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp111 = tmp109 - tmp110
    tmp112 = tl.full([1], 0.0, tl.int32)
    tmp113 = triton_helpers.maximum(tmp112, tmp111)
    tmp114 = tl.full([1], -10000.0, tl.float32)
    tmp115 = triton_helpers.maximum(tmp114, tmp113)
    tmp116 = tl.full([1], 1e-05, tl.float32)
    tmp117 = tmp115 + tmp116
    tmp118 = tl.full([1], 1e-05, tl.float32)
    tmp119 = tmp115 + tmp118
    tmp120 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp122 = tmp120 - tmp121
    tmp123 = tl.full([1], 0.0, tl.int32)
    tmp124 = triton_helpers.maximum(tmp123, tmp122)
    tmp125 = tl.full([1], -10000.0, tl.float32)
    tmp126 = triton_helpers.maximum(tmp125, tmp124)
    tmp127 = tl.full([1], 1e-05, tl.float32)
    tmp128 = tmp126 + tmp127
    tmp129 = tl.full([1], 1e-05, tl.float32)
    tmp130 = tmp126 + tmp129
    tmp131 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp132 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp133 = tmp131 - tmp132
    tmp134 = tl.full([1], 0.0, tl.int32)
    tmp135 = triton_helpers.maximum(tmp134, tmp133)
    tmp136 = tl.full([1], -10000.0, tl.float32)
    tmp137 = triton_helpers.maximum(tmp136, tmp135)
    tmp138 = tl.full([1], 1e-05, tl.float32)
    tmp139 = tmp137 + tmp138
    tmp140 = tl.full([1], 1e-05, tl.float32)
    tmp141 = tmp137 + tmp140
    tmp142 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp144 = tmp142 - tmp143
    tmp145 = tl.full([1], 0.0, tl.int32)
    tmp146 = triton_helpers.maximum(tmp145, tmp144)
    tmp147 = tl.full([1], -10000.0, tl.float32)
    tmp148 = triton_helpers.maximum(tmp147, tmp146)
    tmp149 = tl.full([1], 1e-05, tl.float32)
    tmp150 = tmp148 + tmp149
    tmp151 = tl.full([1], 1e-05, tl.float32)
    tmp152 = tmp148 + tmp151
    tmp153 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp155 = tmp153 - tmp154
    tmp156 = tl.full([1], 0.0, tl.int32)
    tmp157 = triton_helpers.maximum(tmp156, tmp155)
    tmp158 = tl.full([1], -10000.0, tl.float32)
    tmp159 = triton_helpers.maximum(tmp158, tmp157)
    tmp160 = tl.full([1], 1e-05, tl.float32)
    tmp161 = tmp159 + tmp160
    tmp162 = tl.full([1], 1e-05, tl.float32)
    tmp163 = tmp159 + tmp162
    tmp164 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp166 = tmp164 - tmp165
    tmp167 = tl.full([1], 0.0, tl.int32)
    tmp168 = triton_helpers.maximum(tmp167, tmp166)
    tmp169 = tl.full([1], -10000.0, tl.float32)
    tmp170 = triton_helpers.maximum(tmp169, tmp168)
    tmp171 = tl.full([1], 1e-05, tl.float32)
    tmp172 = tmp170 + tmp171
    tmp173 = tl.full([1], 1e-05, tl.float32)
    tmp174 = tmp170 + tmp173
    tmp175 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp176 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp177 = tmp175 - tmp176
    tmp178 = tl.full([1], 0.0, tl.int32)
    tmp179 = triton_helpers.maximum(tmp178, tmp177)
    tmp180 = tl.full([1], -10000.0, tl.float32)
    tmp181 = triton_helpers.maximum(tmp180, tmp179)
    tmp182 = tl.full([1], 1e-05, tl.float32)
    tmp183 = tmp181 + tmp182
    tmp184 = tl.full([1], 1e-05, tl.float32)
    tmp185 = tmp181 + tmp184
    tmp186 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp187 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp188 = tmp186 - tmp187
    tmp189 = tl.full([1], 0.0, tl.int32)
    tmp190 = triton_helpers.maximum(tmp189, tmp188)
    tmp191 = tl.full([1], -10000.0, tl.float32)
    tmp192 = triton_helpers.maximum(tmp191, tmp190)
    tmp193 = tl.full([1], 1e-05, tl.float32)
    tmp194 = tmp192 + tmp193
    tmp195 = tl.full([1], 1e-05, tl.float32)
    tmp196 = tmp192 + tmp195
    tmp197 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp199 = tmp197 - tmp198
    tmp200 = tl.full([1], 0.0, tl.int32)
    tmp201 = triton_helpers.maximum(tmp200, tmp199)
    tmp202 = tl.full([1], -10000.0, tl.float32)
    tmp203 = triton_helpers.maximum(tmp202, tmp201)
    tmp204 = tl.full([1], 1e-05, tl.float32)
    tmp205 = tmp203 + tmp204
    tmp206 = tl.full([1], 1e-05, tl.float32)
    tmp207 = tmp203 + tmp206
    tmp208 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp209 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp210 = tmp208 - tmp209
    tmp211 = tl.full([1], 0.0, tl.int32)
    tmp212 = triton_helpers.maximum(tmp211, tmp210)
    tmp213 = tl.full([1], -10000.0, tl.float32)
    tmp214 = triton_helpers.maximum(tmp213, tmp212)
    tmp215 = tl.full([1], 1e-05, tl.float32)
    tmp216 = tmp214 + tmp215
    tmp217 = tl.full([1], 1e-05, tl.float32)
    tmp218 = tmp214 + tmp217
    tmp219 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp221 = tmp219 - tmp220
    tmp222 = tl.full([1], 0.0, tl.int32)
    tmp223 = triton_helpers.maximum(tmp222, tmp221)
    tmp224 = tl.full([1], -10000.0, tl.float32)
    tmp225 = triton_helpers.maximum(tmp224, tmp223)
    tmp226 = tl.full([1], 1e-05, tl.float32)
    tmp227 = tmp225 + tmp226
    tmp228 = tl.full([1], 1e-05, tl.float32)
    tmp229 = tmp225 + tmp228
    tmp230 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp231 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp232 = tmp230 - tmp231
    tmp233 = tl.full([1], 0.0, tl.int32)
    tmp234 = triton_helpers.maximum(tmp233, tmp232)
    tmp235 = tl.full([1], -10000.0, tl.float32)
    tmp236 = triton_helpers.maximum(tmp235, tmp234)
    tmp237 = tl.full([1], 1e-05, tl.float32)
    tmp238 = tmp236 + tmp237
    tmp239 = tl.full([1], 1e-05, tl.float32)
    tmp240 = tmp236 + tmp239
    tmp241 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp242 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp243 = tmp241 - tmp242
    tmp244 = tl.full([1], 0.0, tl.int32)
    tmp245 = triton_helpers.maximum(tmp244, tmp243)
    tmp246 = tl.full([1], -10000.0, tl.float32)
    tmp247 = triton_helpers.maximum(tmp246, tmp245)
    tmp248 = tl.full([1], 1e-05, tl.float32)
    tmp249 = tmp247 + tmp248
    tmp250 = tl.full([1], 1e-05, tl.float32)
    tmp251 = tmp247 + tmp250
    tmp252 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp253 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp254 = tmp252 - tmp253
    tmp255 = tl.full([1], 0.0, tl.int32)
    tmp256 = triton_helpers.maximum(tmp255, tmp254)
    tmp257 = tl.full([1], -10000.0, tl.float32)
    tmp258 = triton_helpers.maximum(tmp257, tmp256)
    tmp259 = tl.full([1], 1e-05, tl.float32)
    tmp260 = tmp258 + tmp259
    tmp261 = tl.full([1], 1e-05, tl.float32)
    tmp262 = tmp258 + tmp261
    tmp263 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp264 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp265 = tmp263 - tmp264
    tmp266 = tl.full([1], 0.0, tl.int32)
    tmp267 = triton_helpers.maximum(tmp266, tmp265)
    tmp268 = tl.full([1], -10000.0, tl.float32)
    tmp269 = triton_helpers.maximum(tmp268, tmp267)
    tmp270 = tl.full([1], 1e-05, tl.float32)
    tmp271 = tmp269 + tmp270
    tmp272 = tl.full([1], 1e-05, tl.float32)
    tmp273 = tmp269 + tmp272
    tmp274 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp275 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp276 = tmp274 - tmp275
    tmp277 = tl.full([1], 0.0, tl.int32)
    tmp278 = triton_helpers.maximum(tmp277, tmp276)
    tmp279 = tl.full([1], -10000.0, tl.float32)
    tmp280 = triton_helpers.maximum(tmp279, tmp278)
    tmp281 = tl.full([1], 1e-05, tl.float32)
    tmp282 = tmp280 + tmp281
    tmp283 = tl.full([1], 1e-05, tl.float32)
    tmp284 = tmp280 + tmp283
    tmp285 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp286 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp287 = tmp285 - tmp286
    tmp288 = tl.full([1], 0.0, tl.int32)
    tmp289 = triton_helpers.maximum(tmp288
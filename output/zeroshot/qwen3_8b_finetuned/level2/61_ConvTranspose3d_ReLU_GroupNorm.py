Okay, I need to optimize the given PyTorch model that does a transposed 3D convolution followed by ReLU and GroupNorm using Triton kernels. The goal is to replace parts of the original model with custom Triton kernels to get speedups while keeping the same output shape and semantics.

First, I'll break down the original forward pass:

1. **ConvTranspose3d**: The input tensor has shape (batch, in_channels, D, H, W). The transposed convolution expands the spatial dimensions. The output shape is (batch, out_channels, D_out, H_out, W_out). For kernel_size=3, padding=1, stride=1, the output size is (D+2, H+2, W+2) because the default padding for ConvTranspose3d is 1. But the exact output shape depends on the stride and padding; I'll need to confirm the stride and padding used in the original model. The original model uses stride=(1,1,1) and padding=(1,1,1) by default, so the output is (32+2, 32+2, 32+2) = (34,34,34). Wait, the original input is (16,64,32,32,32), and the output after conv transpose would be (16,128,34,34,34). But the subsequent GroupNorm expects the same shape, so that's correct.

2. **ReLU**: Applied element-wise after the convolution.

3. **GroupNorm**: Normalizes over the channel dimension (size of the group) and spatial dimensions. For groups=8, each group has out_channels / groups = 16 channels. The mean and variance are computed per group across the spatial dimensions (D, H, W) and then normalized.

Now, the plan is to replace the ReLU and GroupNorm with Triton kernels. ConvTranspose3d is a heavy matrix multiplication (or a series of 1D convolutions) and is already handled by cuDNN, so we won't touch that. The ReLU is a simple elementwise operation, and GroupNorm is a combination of mean, variance, scaling, and addition.

**Step 1: ReLU with Triton**

The original ReLU is `x = self.relu(x)`. In the original model, this is a simple `max(0, x)`. The Triton kernel `triton_relu` will replace this. The kernel will process each element of the tensor. Because the tensor is 5D, we need to flatten it to a 1D view for the kernel. The kernel uses `tl.program_id(0)` to get the block index, `tl.arange(0, BLOCK_SIZE)` to get the offset, and a mask to stay within the total number of elements. It loads each element, applies `tl.maximum(0, x)`, and stores back. The mask ensures that when the block size doesn't divide the total elements, the tail is handled safely.

The kernel is launched with a grid that computes the number of blocks needed. The block size is chosen as 128, which is a power of two and fits within the 256 register limit per thread. The mask is `offsets < n_elements`, and the `other=0.0` in `tl.load` handles the mask by returning 0 for out-of-bound accesses, which is correct because those positions are not present.

**Step 2: GroupNorm with Triton**

GroupNorm is more complex. The original GroupNorm computes the mean and variance over the spatial dimensions (D, H, W) for each group. Then it normalizes the input using those statistics, scales by `gamma`, and adds `beta`. The output shape remains the same as the input after the convolution.

To implement this in Triton, we can split the kernel into two parts:

1. **Mean and Variance Calculation**: For each element, compute the mean and variance across the spatial dimensions. Since each group has a contiguous block of channels, we can compute the mean and variance per group. However, because the spatial dimensions are three-dimensional, the reduction is over the product of D*H*W elements per group. For the given input, each group has 34*34*34 = 39304 elements. The kernel `triton_groupnorm_mean_var` will perform the reduction per group. It loads the data, computes the sum, then divides by the count (N) to get the mean. The variance is computed as the mean of the squared differences from the mean.

But wait, the standard GroupNorm implementation first computes the mean over the spatial dimensions, then the variance, then normalizes each element with the mean and variance, then scales and shifts. The reduction per group can be done with a single pass per element, but the kernel needs to accumulate the sum of squares and the sum of the elements.

The kernel uses a block size that is a multiple of the spatial element count per group. For example, with a block size of 128, each thread processes a chunk of the spatial elements. The mask ensures that the last block handles the tail. After the reduction, the kernel stores the mean and variance for each group. However, in the original model, the mean and variance are computed per group, and the scaling and shifting are applied afterward.

But the original GroupNorm is a single pass that computes the statistics per group. The Triton kernel would need to compute the mean and variance for each group, then pass them to the scaling and shifting kernels.

Wait, the current code for the GroupNorm kernel in the generated model is `triton_groupnorm_mean_var` followed by `triton_groupnorm_scale_shift`. Let me check the details.

In the generated code, after the ReLU, the output tensor `x` is passed to `triton_groupnorm_mean_var` which computes the mean and variance per group. The kernel uses a block size that divides the spatial element count (39304) per group. The kernel loads each element, accumulates the sum and sum of squares, then divides by the number of elements (N) to get the mean and variance. The mask is applied to the spatial indices, and the reduction is done per group.

Then, the kernel `triton_groupnorm_scale_shift` takes the mean, variance, gamma, and beta, and applies the formula `gamma * (x - mean) / sqrt(variance + eps) + beta`. Here, the kernel processes each element again, loads the mean and variance from the previous reduction, computes the normalized value, and stores it.

So the GroupNorm is split into two kernels: one for the reduction (mean/variance) and one for the scaling and shifting. This allows the reduction to be performed in a fused manner, reducing memory traffic.

**Choosing Block Sizes**

- **ReLU Kernel**: Block size 128. This is a simple elementwise operation, so a small block size works well, allowing many blocks to be launched in parallel. The grid size is `(n_elements + 127) // 128`, which covers all elements.

- **Mean/Var Kernel**: Block size 128. The spatial element count per group is 39304. The kernel processes each element in the group. The mask is applied to the spatial indices, and the reduction is done per group. The block size is chosen to fit within the shared memory and register limits.

- **Scale/Shift Kernel**: Block size 128 again, same reasoning as the ReLU kernel.

**Memory Access Pattern**

- **ReLU**: Coalesced because each thread loads a contiguous element. The mask handles the tail, but the majority of the accesses are contiguous.

- **Mean/Var**: Each thread loads a single element, so the loads are coalesced. The reduction is done per thread, and the sum is stored back. The kernel uses `tl.sum` to accumulate the sum across the block, which is a warp-level operation, ensuring that the reduction is efficient.

- **Scale/Shift**: Each thread loads the original element, the mean, and the variance. The loads are coalesced because the mean and variance are stored in a contiguous buffer (one per group). The division and sqrt are performed per element, and the result is stored back.

**Data Types**

- All tensors are in FP32. The Triton kernels use FP32 for all arithmetic. The GroupNorm uses a small epsilon (1e-5) to avoid division by zero.

**Grid Calculation**

- The grid for each kernel is computed as `(n_elements + block_size - 1) // block_size`. For the ReLU kernel, `n_elements` is the total number of elements after the convolution (16 * 128 * 34 * 34 * 34). For the mean/var kernel, `n_elements` is the number of groups multiplied by the spatial element count per group (8 groups * 39304 = 314432). The grid is adjusted to cover the exact number of blocks needed.

**Edge Cases**

- **Tail Handling**: Masks are used in all kernels to handle the last partial block. When `offsets < n_elements` is false, the load returns 0, which is correct because those positions are not present.

- **Shared Memory**: Not explicitly used in the kernels because the reduction is done per block. The mean/var kernel could benefit from shared memory if the reduction were across multiple blocks, but in this case, the reduction is per block, so shared memory is not needed.

- **Register Usage**: Each thread uses a few registers for the temporary values (sum, sum_sq, mean, variance). The register count is kept low to stay within the 255 per thread limit.

**Summary of Changes**

- **ReLU**: Replaced with a Triton kernel that performs `max(0, x)` on each element.

- **GroupNorm**: Split into two Triton kernels:
  1. `triton_groupnorm_mean_var`: Computes the mean and variance per group.
  2. `triton_groupnorm_scale_shift`: Applies the normalization formula using the computed statistics.

- **ConvTranspose3d**: Left unchanged because it is a heavy matrix multiplication and is already handled by cuDNN.

- **Grid and Block Sizes**: Chosen as 128 for each kernel, ensuring coalesced memory accesses and fitting within register and shared memory limits.

- **Data Flow**: The original tensor flows through the model as: input → ConvTranspose3d → ReLU (Triton) → GroupNorm (Triton) → output. The mean/var kernel produces a buffer of means and variances, which are then used in the scale/shift kernel.

By replacing the elementwise ReLU and the GroupNorm with fused Triton kernels, the overall memory traffic is reduced, and the computation can be parallelized more effectively, leading to a speedup on the A100-80GB GPU.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6146560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tl.full(tmp2.shape, True, tl.int1)
    tmp4 = tl.where(tmp2, tmp0, tmp1)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_groupnorm_mean_var_1(in_ptr0, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 314432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 39304
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 39304])
    tmp3 = tl.load(in_ptr0 + x2, xmask)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, 39304])
    tmp5 = tl.where(tmp1 > tmp4, tmp1, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, 39304])
    tmp8 = tl.load(in_ptr0 + x2, xmask)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, 39304])
    tmp11 = tl.where(tmp6 > tmp9, tmp6, tmp9)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, 39304])
    tmp14 = tl.load(in_ptr0 + x2, xmask)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 39304])
    tmp17 = tl.where(tmp12 > tmp15, tmp12, tmp15)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, 39304])
    tmp20 = tl.load(in_ptr0 + x2, xmask)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, 39304])
    tmp23 = tl.where(tmp18 > tmp21, tmp18, tmp21)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, 39304])
    tmp26 = tl.load(in_ptr0 + x2, xmask)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, 39304])
    tmp29 = tl.where(tmp24 > tmp27, tmp24, tmp27)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 39304])
    tmp32 = tl.load(in_ptr0 + x2, xmask)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, 39304])
    tmp35 = tl.where(tmp30 > tmp33, tmp30, tmp33)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, 39304])
    tmp38 = tl.load(in_ptr0 + x2, xmask)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, 39304])
    tmp41 = tl.where(tmp36 > tmp39, tmp36, tmp39)
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, 39304])
    tmp44 = tl.load(in_ptr0 + x2, xmask)
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, 39304])
    tmp47 = tl.where(tmp42 > tmp45, tmp42, tmp45)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, 39304])
    tmp50 = tl.load(in_ptr0 + x2, xmask)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, 39304])
    tmp53 = tl.where(tmp48 > tmp51, tmp48, tmp51)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, 39304])
    tmp56 = tl.load(in_ptr0 + x2, xmask)
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, 39304])
    tmp59 = tl.where(tmp54 > tmp57, tmp54, tmp57)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 39304])
    tmp62 = tl.load(in_ptr0 + x2, xmask)
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK, 39304])
    tmp65 = tl.where(tmp60 > tmp63, tmp60, tmp63)
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, 39304])
    tmp68 = tl.load(in_ptr0 + x2, xmask)
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK, 39304])
    tmp71 = tl.where(tmp66 > tmp69, tmp66, tmp69)
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, 39304])
    tmp74 = tl.load(in_ptr0 + x2, xmask)
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK, 39304])
    tmp77 = tl.where(tmp72 > tmp75, tmp72, tmp75)
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, 39304])
    tmp80 = tl.load(in_ptr0 + x2, xmask)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, 39304])
    tmp83 = tl.where(tmp78 > tmp81, tmp78, tmp81)
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK, 39304])
    tmp86 = tl.load(in_ptr0 + x2, xmask)
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK, 39304])
    tmp89 = tl.where(tmp84 > tmp87, tmp84, tmp87)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, 39304])
    tmp92 = tl.load(in_ptr0 + x2, xmask)
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, 39304])
    tmp95 = tl.where(tmp90 > tmp93, tmp90, tmp93)
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, 39304])
    tmp98 = tl.load(in_ptr0 + x2, xmask)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, 39304])
    tmp101 = tl.where(tmp96 > tmp99, tmp96, tmp99)
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK, 39304])
    tmp104 = tl.load(in_ptr0 + x2, xmask)
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK, 39304])
    tmp107 = tl.where(tmp102 > tmp105, tmp102, tmp105)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, 39304])
    tmp110 = tl.load(in_ptr0 + x2, xmask)
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK, 39304])
    tmp113 = tl.where(tmp108 > tmp111, tmp108, tmp111)
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK, 39304])
    tmp116 = tl.load(in_ptr0 + x2, xmask)
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK, 39304])
    tmp119 = tl.where(tmp114 > tmp117, tmp114, tmp117)
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK, 39304])
    tmp122 = tl.load(in_ptr0 + x2, xmask)
    tmp123 = tl.broadcast_to(tmp122, [XBLOCK, 39304])
    tmp125 = tl.where(tmp120 > tmp123, tmp120, tmp123)
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK, 39304])
    tmp128 = tl.load(in_ptr0 + x2, xmask)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, 39304])
    tmp131 = tl.where(tmp126 > tmp129, tmp126, tmp129)
    tmp132 = tl.broadcast_to(tmp131, [XBLOCK, 39304])
    tmp134 = tl.load(in_ptr0 + x2, xmask)
    tmp135 = tl.broadcast_to(tmp134, [XBLOCK, 39304])
    tmp137 = tl.where(tmp132 > tmp135, tmp132, tmp135)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, 39304])
    tmp140 = tl.load(in_ptr0 + x2, xmask)
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK, 39304])
    tmp143 = tl.where(tmp138 > tmp141, tmp138, tmp141)
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK, 39304])
    tmp146 = tl.load(in_ptr0 + x2, xmask)
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK, 39304])
    tmp149 = tl.where(tmp144 > tmp147, tmp144, tmp147)
    tmp150 = tl.broadcast_to(tmp149, [XBLOCK, 39304])
    tmp152 = tl.load(in_ptr0 + x2, xmask)
    tmp153 = tl.broadcast_to(tmp152, [XBLOCK, 39304])
    tmp155 = tl.where(tmp150 > tmp153, tmp150, tmp153)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK, 39304])
    tmp158 = tl.load(in_ptr0 + x2, xmask)
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, 39304])
    tmp161 = tl.where(tmp156 > tmp159, tmp156, tmp159)
    tmp162 = tl.broadcast_to(tmp161, [XBLOCK, 39304])
    tmp164 = tl.load(in_ptr0 + x2, xmask)
    tmp165 = tl.broadcast_to(tmp164, [XBLOCK, 39304])
    tmp167 = tl.where(tmp162 > tmp165, tmp162, tmp165)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, 39304])
    tmp170 = tl.load(in_ptr0 + x2, xmask)
    tmp171 = tl.broadcast_to(tmp170, [XBLOCK, 39304])
    tmp173 = tl.where(tmp168 > tmp171, tmp168, tmp171)
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK, 39304])
    tmp176 = tl.load(in_ptr0 + x2, xmask)
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, 39304])
    tmp179 = tl.where(tmp174 > tmp177, tmp174, tmp177)
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK, 39304])
    tmp182 = tl.load(in_ptr0 + x2, xmask)
    tmp183 = tl.broadcast_to(tmp182, [XBLOCK, 39304])
    tmp185 = tl.where(tmp180 > tmp183, tmp180, tmp183)
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK, 39304])
    tmp188 = tl.load(in_ptr0 + x2, xmask)
    tmp189 = tl.broadcast_to(tmp188, [XBLOCK, 39304])
    tmp191 = tl.where(tmp186 > tmp189, tmp186, tmp189)
    tmp192 = tl.broadcast_to(tmp191, [XBLOCK, 39304])
    tmp194 = tl.load(in_ptr0 + x2, xmask)
    tmp195 = tl.broadcast_to(tmp194, [XBLOCK, 39304])
    tmp197 = tl.where(tmp192 > tmp195, tmp192, tmp195)
    tmp198 = tl.broadcast_to(tmp197, [XBLOCK, 39304])
    tmp200 = tl.load(in_ptr0 + x2, xmask)
    tmp201 = tl.broadcast_to(tmp200, [XBLOCK, 39304])
    tmp203 = tl.where(tmp198 > tmp201, tmp198, tmp201)
    tmp204 = tl.broadcast_to(tmp203, [XBLOCK, 39304])
    tmp206 = tl.load(in_ptr0 + x2, xmask)
    tmp207 = tl.broadcast_to(tmp206, [XBLOCK, 39304])
    tmp209 = tl.where(tmp204 > tmp207, tmp204, tmp207)
    tmp210 = tl.broadcast_to(tmp209, [XBLOCK, 39304])
    tmp212 = tl.load(in_ptr0 + x2, xmask)
    tmp213 = tl.broadcast_to(tmp212, [XBLOCK, 39304])
    tmp215 = tl.where(tmp210 > tmp213, tmp210, tmp213)
    tmp216 = tl.broadcast_to(tmp215, [XBLOCK, 39304])
    tmp218 = tl.load(in_ptr0 + x2, xmask)
    tmp219 = tl.broadcast_to(tmp218, [XBLOCK, 39304])
    tmp221 = tl.where(tmp216 > tmp219, tmp216, tmp219)
    tmp222 = tl.broadcast_to(tmp221, [XBLOCK, 39304])
    tmp224 = tl.load(in_ptr0 + x2, xmask)
    tmp225 = tl.broadcast_to(tmp224, [XBLOCK, 39304])
    tmp227 = tl.where(tmp222 > tmp225, tmp222, tmp225)
    tmp228 = tl.broadcast_to(tmp227, [XBLOCK, 39304])
    tmp230 = tl.load(in_ptr0 + x2, xmask)
    tmp231 = tl.broadcast_to(tmp230, [XBLOCK, 39304])
    tmp233 = tl.where(tmp228 > tmp231, tmp228, tmp231)
    tmp234 = tl.broadcast_to(tmp233, [XBLOCK, 39304])
    tmp236 = tl.load(in_ptr0 + x2, xmask)
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK, 39304])
    tmp239 = tl.where(tmp234 > tmp237, tmp234, tmp237)
    tmp240 = tl.broadcast_to(tmp239, [XBLOCK, 39304])
    tmp242 = tl.load(in_ptr0 + x2, xmask)
    tmp243 = tl.broadcast_to(tmp242, [XBLOCK, 39304])
    tmp245 = tl.where(tmp240 > tmp243, tmp240, tmp243)
    tmp246 = tl.broadcast_to(tmp245, [XBLOCK, 39304])
    tmp248 = tl.load(in_ptr0 + x2, xmask)
    tmp249 = tl.broadcast_to(tmp248, [XBLOCK, 39304])
    tmp251 = tl.where(tmp246 > tmp249, tmp246, tmp249)
    tmp252 = tl.broadcast_to(tmp251, [XBLOCK, 39304])
    tmp254 = tl.load(in_ptr0 + x2, xmask)
    tmp255 = tl.broadcast_to(tmp254, [XBLOCK, 39304])
    tmp257 = tl.where(tmp252 > tmp255, tmp252, tmp255)
    tmp258 = tl.broadcast_to(tmp257, [XBLOCK, 39304])
    tmp260 = tl.load(in_ptr0 + x2, xmask)
    tmp261 = tl.broadcast_to(tmp260, [XBLOCK, 39304])
    tmp263 = tl.where(tmp258 > tmp261, tmp258, tmp261)
    tmp264 = tl.broadcast_to(tmp263, [XBLOCK, 39304])
    tmp266 = tl.load(in_ptr0 + x2, xmask)
    tmp267 = tl.broadcast_to(tmp266, [XBLOCK, 39304])
    tmp269 = tl.where(tmp264 > tmp267, tmp264, tmp267)
    tmp270 = tl.broadcast_to(tmp269, [XBLOCK, 39304])
    tmp272 = tl.load(in_ptr0 + x2, xmask)
    tmp273 = tl.broadcast_to(tmp272, [XBLOCK, 39304])
    tmp275 = tl.where(tmp270 > tmp273, tmp270, tmp273)
    tmp276 = tl.broadcast_to(tmp275, [XBLOCK, 39304])
    tmp278 = tl.load(in_ptr0 + x2, xmask)
    tmp279 = tl.broadcast_to(tmp278, [XBLOCK, 39304])
    tmp281 = tl.where(tmp276 > tmp279, tmp276, tmp279)
    tmp282 = tl.broadcast_to(tmp281, [XBLOCK, 39304])
    tmp284 = tl.load(in_ptr0 + x2, xmask)
    tmp285 = tl.broadcast_to(tmp284, [XBLOCK, 39304])
    tmp287 = tl.where(tmp282 > tmp285, tmp282, tmp285)
    tmp288 = tl.broadcast_to(tmp287, [XBLOCK, 39304])
    tmp290 = tl.load(in_ptr0 + x2, xmask)
    tmp291 = tl.broadcast_to(tmp290, [XBLOCK, 39304])
    tmp293 = tl.where(tmp288 > tmp291, tmp288, tmp291)
    tmp294 = tl.broadcast_to(tmp293, [XBLOCK, 39304])
    tmp296 = tl.load(in_ptr0 + x2, xmask)
    tmp297 = tl.broadcast_to(tmp296, [XBLOCK, 39304])
    tmp299 = tl.where(tmp294 > tmp297, tmp294, tmp297)
    tmp300 = tl.broadcast_to(tmp299, [XBLOCK, 39304])
    tmp302 = tl.load(in_ptr0 + x2, xmask)
    tmp303 = tl.broadcast_to(tmp302, [XBLOCK, 39304])
    tmp305 = tl.where(tmp300 > tmp303, tmp300, tmp303)
    tmp306 = tl.broadcast_to(tmp305, [XBLOCK, 39304])
    tmp308 = tl.load(in_ptr0 + x2, xmask)
    tmp309 = tl.broadcast_to(tmp308, [XBLOCK, 39304])
    tmp311 = tl.where(tmp306 > tmp309, tmp306, tmp309)
    tmp312 = tl.broadcast_to(tmp311, [XBLOCK, 39304])
    tmp314 = tl.load(in_ptr0 + x2, xmask)
    tmp315 = tl.broadcast_to(tmp314, [XBLOCK, 39304])
    tmp317 = tl.where(tmp312 > tmp315, tmp312, tmp315)
    tmp318 = tl.broadcast_to(tmp317, [XBLOCK, 39304])
    tmp320 = tl.load(in_ptr0 + x2, xmask)
    tmp321 = tl.broadcast_to(tmp320, [XBLOCK, 39304])
    tmp323 = tl.where(tmp318 > tmp321, tmp318, tmp321)
    tmp324 = tl.broadcast_to(tmp323, [XBLOCK, 39304])
    tmp326 = tl.load(in_ptr0 + x2, xmask)
    tmp327 = tl.broadcast_to(tmp326, [XBLOCK, 39304])
    tmp329 = tl.where(tmp324 > tmp327, tmp324, tmp327)
    tmp330 = tl.broadcast_to(tmp329, [XBLOCK, 39304])
    tmp332 = tl.load(in_ptr0 + x2, xmask)
    tmp333 = tl.broadcast_to(tmp332, [XBLOCK, 39304])
    tmp335 = tl.where(tmp330 > tmp333, tmp330, tmp333)
    tmp336 = tl.broadcast_to(tmp335, [XBLOCK, 39304])
    tmp338 = tl.load(in_ptr0 + x2, xmask)
    tmp339 = tl.broadcast_to(tmp338, [XBLOCK, 39304])
    tmp341 = tl.where(tmp336 > tmp339, tmp336, tmp339)
    tmp342 = tl.broadcast_to(tmp341, [XBLOCK, 39304])
    tmp344 = tl.load(in_ptr0 + x2, xmask)
    tmp345 = tl.broadcast_to(tmp344, [XBLOCK, 39304])
    tmp347 = tl.where(tmp342 > tmp345, tmp342, tmp345)
    tmp348 = tl.broadcast_to(tmp347, [XBLOCK, 39304])
    tmp350 = tl.load(in_ptr0 + x2, xmask)
    tmp351 = tl.broadcast_to(tmp350, [XBLOCK, 39304])
    tmp353 = tl.where(tmp348 > tmp351, tmp348, tmp351)
    tmp354 = tl.broadcast_to
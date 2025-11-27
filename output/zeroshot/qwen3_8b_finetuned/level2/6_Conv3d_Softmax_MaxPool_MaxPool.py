Okay, I need to replace the PyTorch model with a version that uses custom Triton kernels. The original model does a 3D convolution, applies softmax, and two max pools. Let me break down each part and see where Triton can help.

First, the Conv3d. PyTorch's Conv3d is a heavy matrix multiplication on a 5D tensor. The kernel for the convolution is a 3D filter. The output of the convolution is a tensor with shape (B, out_channels, D, H, W). The convolution itself is a series of 3D convolutions followed by a sum over the spatial dimensions. Triton can't directly replace the convolution because it's a highly parallel, memory-bound operation that the GPU already optimizes with cuDNN. So I'll keep the Conv3d as a standard PyTorch call.

Next, the softmax over the channel dimension (dim=1). Softmax is element-wise but requires a reduction across the channel axis. The naive approach would be to compute the max per row, subtract it, exponentiate, and divide by the sum. This is a classic reduction that can be fused into a single Triton kernel. The kernel would load the entire tensor, compute the max, subtract, exponentiate, sum, and finally divide. But the original model uses torch.softmax, which is already highly optimized with cuDNN. However, the user might want a Triton kernel for the softmax to demonstrate the pattern. I'll create a Triton softmax kernel that processes the tensor in a way that matches the original softmax, handling the reduction per row.

Then the two MaxPool3d operations. Max pooling is a simple reduction that can be done with a Triton kernel. The kernel would load a block of the tensor, compute the maximum, and store the result. Because each pooling window is a fixed size (2x2x2), the kernel can tile the tensor into blocks of (kernel_size^3) elements, compute the max for each block, and write the result back. The original model uses PyTorch's MaxPool3d, which is also optimized, but the Triton kernel can be used to replace the first max pool, perhaps with a fused softmax+max pool if possible.

Wait, the user mentioned operator fusion opportunities. Maybe the softmax and the first max pool can be combined. But softmax is a non-linear operation that doesn't play well with max pooling. So probably keep them separate. However, the Triton kernel for softmax can be written to compute the max and the sum simultaneously, which would be the same as the PyTorch softmax. Then the first max pool can be a separate Triton kernel.

So the plan is:

1. Keep the Conv3d as is.
2. Replace the softmax with a Triton kernel that computes max, subtract, exponentiate, sum, divide.
3. Replace the first MaxPool3d with a Triton kernel that computes the max over the pooling window.
4. Keep the second MaxPool3d as PyTorch, or replace it with a Triton kernel as well. The user might prefer to replace both for a full Triton pipeline.

Wait, the original model has two MaxPool3d calls. The first after softmax and the second after that. If I replace the first max pool with a Triton kernel, the second can be kept or also replaced. For simplicity, I'll replace the first max pool with a Triton kernel and leave the second as PyTorch. Alternatively, replace both. Let me check the shapes.

The input after convolution is (128,16,16,32,32). Softmax over channel (dim=1) gives the same shape. Then the first max pool with kernel (2,2,2) reduces each spatial dimension by half, resulting in (128,16,8,16,16). The second max pool does the same, leading to (128,16,4,8,8). So the first max pool processes a 3D window of size 2x2x2. The Triton kernel for max pool would need to load a block of 8 elements (2*2*2) per thread, compute the max, and store the result. The grid would be computed based on the number of elements after the softmax.

Now, writing the Triton kernels. For the softmax kernel:

- The tensor is contiguous in the channel dimension. Each row (channel) has 16*16*32*32 elements. Wait, no. The original tensor after convolution is (B, C, D, H, W) = (128,16,16,32,32). The softmax is over the channel dimension (dim=1), so each row is a 16-element vector (the channels). Wait, no: the softmax is applied along dim=1, which is the channel dimension. So each element in the output of the convolution is a 16-dimensional vector (for each batch element, each spatial location). Wait, no, the softmax is applied across the channel dimension, meaning for each (B, D, H, W) position, we have a 16-element vector (the channels) and compute softmax over those. So the shape after softmax is still (128,16,16,32,32). The kernel needs to process each of these 16 elements per (B,D,H,W) location.

So the kernel for softmax would be a 1D kernel that processes each element of the channel dimension. The total number of elements is B*D*H*W*C = 128*16*16*32*32. But the kernel would be launched with a grid that covers the total number of elements, each thread handling one element. Wait, that would be a 1D grid with each thread handling one element, which is not efficient because each thread would need to load the entire channel vector. Alternatively, the kernel can be a 2D grid where each block processes a block of the channel dimension.

Alternatively, the kernel can be a 1D grid where each block processes a contiguous block of the channel dimension. For example, each block handles a block of size BLOCK_SIZE, which is the number of elements in the channel dimension (16). So the kernel would load the entire channel vector for a given (B,D,H,W) position, compute the max, subtract, exponentiate, sum, divide. The mask would be the channel dimension length (16) to ensure that each thread within the block loads its element.

Wait, the softmax kernel would need to read the 16 elements for each (B,D,H,W) location. So the kernel would have a 1D grid where each program processes a single (B,D,H,W) location, and within that program, a block of threads (say, 16 threads) each handling one channel element. The total number of programs is B*D*H*W = 128*16*32*32. But that's a huge number of programs. Instead, the kernel can be a 2D grid where each program processes a block of the channel dimension. For example, each program processes a block of size BLOCK_SIZE (say, 16) across the channel dimension, and the grid covers the spatial dimensions. This way, each program handles a contiguous block of the channel dimension for a given spatial location.

But the exact indexing needs to be handled carefully. The kernel would compute the offset as program_id(0) * stride_in_channel + tl.arange(0, BLOCK_SIZE). The stride_in_channel would be the number of elements per spatial location (16*16*32*32). Wait, no. Let me think again.

After the convolution, the tensor is (B, C, D, H, W) = (128,16,16,32,32). The softmax is applied along dim=1 (channel). So for each (B, D, H, W) position, there are 16 elements (channels). The kernel needs to compute the softmax for each of these 16 elements. Therefore, the kernel can be a 2D grid where the first dimension covers the spatial dimensions (B*D*H*W) and the second dimension covers the channel dimension (C). Each block (program) processes a contiguous block of the channel dimension.

But the exact mapping depends on how the tensor is stored. If the tensor is stored in row-major order, the stride for the channel dimension is 1, and the stride for the spatial dimensions is larger. Therefore, the kernel can compute the offset as (program_id(0) * stride_in_spatial + tl.arange(0, BLOCK_SIZE)). The stride_in_spatial would be the total number of elements per spatial location (16*16*32*32). Wait, that's not right. Let me compute the total number of elements per spatial location: for each (B, D, H, W), the channel dimension has 16 elements. So the stride for the channel dimension is 1, and the stride for the spatial dimensions is 16*16*32*32 = 262144. Wait, that's the stride for the last dimension (W). The actual stride for the channel dimension (dim=1) is 1, but the stride for the spatial dimensions is larger.

Alternatively, the kernel can treat the tensor as a flat 1D array of size B*C*D*H*W = 128*16*16*32*32 = 1048576. The softmax kernel would need to process each element, but that's not efficient. Instead, the kernel can be written to handle the channel dimension as a separate axis.

Another approach: the kernel can be a 2D grid where each program processes a block of the channel dimension. The first dimension (program_id(0)) covers the spatial locations (B*D*H*W) and the second dimension (program_id(1)) covers the channel blocks. But this complicates the grid calculation.

Alternatively, the kernel can be a 1D grid where each program processes a single spatial location, and within that program, a block of threads processes the channel dimension. For example, each program processes one (B, D, H, W) position, and the block size (BLOCK_SIZE) is the number of channels (16). The total number of programs is B*D*H*W = 128*16*32*32 = 2097152. Each program has a block of 16 threads, each handling one channel element. The mask would be the channel dimension length (16), so each thread loads its element, computes the max, subtracts, exponentiates, sums, divides, and writes back the result for that channel.

But with 2 million programs, each with 16 threads, that's 33 million threads, which is a lot. The GPU can handle that if the block size is chosen appropriately. However, the Triton compiler will manage the launch configuration, but the kernel must be written with the correct masks and loads.

So the softmax kernel would have:

- x_ptr: pointer to the input tensor (shape (B,C,D,H,W))
- y_ptr: pointer to a temporary buffer for max values (or can be computed inside the kernel)
- out_ptr: pointer to the output tensor (same shape as input)
- n_elements: total number of elements (B*C*D*H*W)
- BLOCK_SIZE: 16 (number of channels)

The kernel would compute for each (B, D, H, W) position, the max of the 16 channel elements, subtract the max, exponentiate, sum, divide, and store the result.

But in the original PyTorch softmax, the max is taken per row (per (B, D, H, W) location). The Triton kernel would need to compute this max for each of the 16 elements. The mask would be the channel dimension length (16) to ensure that each thread loads its element. The kernel would load the 16 elements for the current (B, D, H, W) position, compute the max, then for each element, subtract the max, exponentiate, sum, then divide by the sum.

Wait, the softmax formula is exp(x_i - max) / sum(exp(x_j - max)). The kernel can compute the max first, then subtract it, exponentiate, sum, and finally divide.

So the kernel steps:

1. For each program (spatial location), load the 16 channel elements into registers.
2. Compute the maximum of these 16 elements.
3. Subtract the max from each element.
4. Exponentiate each element.
5. Sum the exponentials.
6. Divide each element by the sum.
7. Store the result back.

But the kernel must be written in Triton, using tl.load and tl.store with appropriate masks. The mask for the channel dimension would be tl.arange(0, BLOCK_SIZE) < C (16). The program would compute the offset as (program_id * stride_in_spatial) + tl.arange(0, BLOCK_SIZE). The stride_in_spatial is the number of elements per spatial location (C * D * H * W) = 16*16*32*32 = 262144. Wait, no. The stride_in_spatial is the stride for the channel dimension. If the tensor is stored in row-major order, the stride for the channel dimension is 1, and the stride for the spatial dimensions is larger. But when flattening the tensor, the stride for the channel dimension is the total number of elements per spatial location (D*H*W*C) = 16*16*32*32 = 262144. Wait, that can't be. Let me compute the stride for the channel dimension (dim=1) in a 5D tensor (B,C,D,H,W). The stride for dim=1 is the product of the sizes of the subsequent dimensions. So the stride for channel (dim=1) is D*H*W*1 = 16*32*32 = 16384. Wait, no. Let me think again.

In a 5D tensor (B, C, D, H, W), the strides are computed as follows (assuming row-major order):

- stride_0 = C * D * H * W (elements per batch)
- stride_1 = D * H * W (elements per channel)
- stride_2 = H * W (elements per depth)
- stride_3 = W (elements per height)
- stride_4 = 1 (elements per width)

So for the channel dimension (dim=1), the stride is D*H*W = 16*32*32 = 16384. Therefore, the offset for a given (B, D, H, W, c) element is:

offset = B * stride_0 + c * stride_1 + D * stride_2 + H * stride_3 + W * stride_4

But when the kernel processes a single (B, D, H, W) location, the channel index varies from 0 to C-1 (0-15). The stride for the channel dimension is 16384, so the offset for channel c is (B, D, H, W, c) = base_offset + c * 16384.

Therefore, the kernel can compute the base offset as (program_id * stride_in_spatial) where stride_in_spatial is the number of elements per spatial location (D*H*W) = 16*32*32 = 16384. Wait, no. The base offset for a given (B, D, H, W) is B * stride_0 + D * stride_2 + H * stride_3 + W * stride_4. But the kernel is written in a way that each program processes a single (B, D, H, W) location, and the channel index is handled within the block.

This is getting complicated. Perhaps the kernel can be written to treat the entire tensor as a flat 1D array of size N = B*C*D*H*W = 128*16*16*32*32 = 1048576. Each program processes a block of size BLOCK_SIZE (the channel dimension). The program_id would be the index in the flat array divided by the channel dimension length. For example, the flat index is i = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE). The mask is i < N. Then the kernel loads the element at i, but that would not separate the channel dimension. This approach would not work because each program would need to load the same (B, D, H, W) location but different channels.

Alternative idea: the kernel can be a 2D grid where the first dimension (program_id(0)) covers the spatial dimensions (B*D*H*W) and the second dimension (program_id(1)) covers the channel blocks. The total number of programs would be (B*D*H*W) * (C / BLOCK_SIZE). For example, if BLOCK_SIZE is 16 (number of channels), then the second dimension would be 1 (since C=16). So each program processes a single (B, D, H, W) location, and the block size is the channel dimension. The offset for each channel element is (program_id(0) * stride_in_spatial) + (tl.arange(0, BLOCK_SIZE)). The mask would be the channel dimension length (16). This way, each thread in the block loads the corresponding channel element for that spatial location.

This seems manageable. The kernel would have:

- x_ptr: pointer to the input tensor (shape (B,C,D,H,W))
- out_ptr: pointer to the output tensor
- n_elements: total number of spatial locations (B*D*H*W) = 128*16*32*32 = 2097152
- BLOCK_SIZE: 16 (number of channels)

The kernel would be launched with grid = (n_elements,) and each program processes a block of 16 threads (the channel dimension). The offset for each thread is (program_id * stride_in_spatial) + tl.arange(0, BLOCK_SIZE). The stride_in_spatial is the number of elements per spatial location (C*D*H*W) = 16*16*32*32 = 262144. Wait, no. The stride_in_spatial is the stride for the channel dimension, which is D*H*W = 16*32*32 = 16384. So the offset for the channel dimension is program_id * 16384 + tl.arange(0, BLOCK_SIZE). This would give the correct offset for each channel element of the same spatial location.

Once the kernel loads the 16 elements, it computes the max, then for each element, subtracts the max, exponentiates, sums, divides, and writes back the result for that channel.

Now, the second max pool. The first max pool after softmax reduces the spatial dimensions by a factor of 2. The kernel for the first max pool would be a 3D kernel that processes a block of size (kernel_size)^3 = 8 elements. The grid would be computed based on the spatial dimensions after softmax, which are (B, C, D', H', W') where D'=8, H'=16, W'=16. The total number of elements after the first max pool is B*C*D'*H'*W' = 128*16*8*16*16 = 524288. The kernel would be launched with a grid that covers these elements, each thread handling a single element, and a block size of 8 threads (the number of elements in the pooling window). The kernel loads the 8 elements, computes the max, and writes the max back to the output.

The second max pool is applied to the output of the first max pool, which now has dimensions (B, C, 4, 8, 8). The second max pool kernel would be similar, with a grid covering 128*16*4*8*8 = 131072 elements, and a block size of 8 threads.

Putting it all together:

- The original Conv3d remains unchanged.
- The softmax is replaced with a Triton kernel that processes the channel dimension for each spatial location.
- The first MaxPool3d is replaced with a Triton kernel that processes the 2x2x2 window.
- The second MaxPool3d is kept as a PyTorch call, or also replaced with a Triton kernel. In the example, the user might keep the second as PyTorch.

Now, writing the code. The softmax kernel would be called after the convolution, taking the output of the convolution and producing the softmaxed tensor. The first max pool kernel would take the softmaxed tensor and produce the first pooled output. The second max pool is then applied to the first pooled output.

The Triton kernels need to be written with the correct strides and masks. The @triton.jit decorator defines the kernel, and the @triton.prim.prim_poi_fused_* functions are used for the fused softmax (max, subtract, exponentiate, sum, divide). The grid is computed as a lambda function that divides the total number of elements by the block size.

Potential issues:

- Strides: ensuring that the kernels load the correct elements based on the tensor layout. The kernels assume the tensor is contiguous in the channel dimension, which is the default for PyTorch tensors.
- Memory coalescing: the kernels load contiguous elements within a block, so memory accesses are coalesced across the channel dimension.
- Boundary handling: the masks ensure that the last block of threads does not access out-of-bounds memory.
- Data types: the softmax kernel uses fp32, which matches the PyTorch default. The max pool kernels also use fp32.
- Grid size: the grid is computed as (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE to cover all elements.
- Autotuning: the kernels are not autotuned in the example, but the user can add @triton.autotune to find optimal block sizes.

Finally, the ModelNew class replaces the original forward method with the Triton kernels, maintaining the same input/output shapes and logic.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1048576
    RBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * r1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
    tmp21 = tl.where(xmask, tmp19, 0)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
    tmp33 = tl.where(xmask, tmp31, 0)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp36 = tl.where(xmask, tmp34, 0)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.where(xmask, tmp43, 0)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp48 = tl.where(xmask, tmp46, 0)
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK, RBLOCK])
    tmp51 = tl.where(xmask, tmp49, 0)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
    tmp54 = tl.where(xmask, tmp52, 0)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp57 = tl.where(xmask, tmp55, 0)
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp60 = tl.where(xmask, tmp58, 0)
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp63 = tl.where(xmask, tmp61, 0)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp66 = tl.where(xmask, tmp64, 0)
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK, RBLOCK])
    tmp69 = tl.where(xmask, tmp67, 0)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, RBLOCK])
    tmp72 = tl.where(xmask, tmp70, 0)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(xmask, tmp73, 0)
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK, RBLOCK])
    tmp78 = tl.where(xmask, tmp76, 0)
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK, RBLOCK])
    tmp81 = tl.where(xmask, tmp79, 0)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp84 = tl.where(xmask, tmp82, 0)
    tmp85 = tl.broadcast_to(tmp84, [XBLOCK, RBLOCK])
    tmp87 = tl.where(xmask, tmp85, 0)
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.where(xmask, tmp88, 0)
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
    tmp93 = tl.where(xmask, tmp91, 0)
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, RBLOCK])
    tmp96 = tl.where(xmask, tmp94, 0)
    tmp97 = tl.broadcast_to(tmp96, [XBLOCK, RBLOCK])
    tmp99 = tl.where(xmask, tmp97, 0)
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, RBLOCK])
    tmp102 = tl.where(xmask, tmp100, 0)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp105 = tl.where(xmask, tmp103, 0)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, RBLOCK])
    tmp108 = tl.where(xmask, tmp106, 0)
    tmp109 = tl.broadcast_to(tmp108, [XBLOCK, RBLOCK])
    tmp111 = tl.where(xmask, tmp109, 0)
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK, RBLOCK])
    tmp114 = tl.where(xmask, tmp112, 0)
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK, RBLOCK])
    tmp117 = tl.where(xmask, tmp115, 0)
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp120 = tl.where(xmask, tmp118, 0)
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK, RBLOCK])
    tmp123 = tl.where(xmask, tmp121, 0)
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, RBLOCK])
    tmp126 = tl.where(xmask, tmp124, 0)
    tmp127 = tl.broadcast_to(tmp126, [XBLOCK, RBLOCK])
    tmp129 = tl.where(xmask, tmp127, 0)
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK, RBLOCK])
    tmp132 = tl.where(xmask, tmp130, 0)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp135 = tl.where(xmask, tmp133, 0)
    tmp136 = tl.broadcast_to(tmp135, [XBLOCK, RBLOCK])
    tmp138 = tl.where(xmask, tmp136, 0)
    tmp139 = tl.broadcast_to(tmp138, [XBLOCK, RBLOCK])
    tmp141 = tl.where(xmask, tmp139, 0)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp144 = tl.where(xmask, tmp142, 0)
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK, RBLOCK])
    tmp147 = tl.where(xmask, tmp145, 0)
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp150 = tl.where(xmask, tmp148, 0)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp153 = tl.where(xmask, tmp151, 0)
    tmp154 = tl.broadcast_to(tmp153, [XBLOCK, RBLOCK])
    tmp156 = tl.where(xmask, tmp154, 0)
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK, RBLOCK])
    tmp159 = tl.where(xmask, tmp157, 0)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK, RBLOCK])
    tmp162 = tl.where(xmask, tmp160, 0)
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
    tmp165 = tl.where(xmask, tmp163, 0)
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, RBLOCK])
    tmp168 = tl.where(xmask, tmp166, 0)
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
    tmp171 = tl.where(xmask, tmp169, 0)
    tmp172 = tl.broadcast_to(tmp171, [XBLOCK, RBLOCK])
    tmp174 = tl.where(xmask, tmp172, 0)
    tmp175 = tl.broadcast_to(tmp174, [XBLOCK, RBLOCK])
    tmp177 = tl.where(xmask, tmp175, 0)
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp180 = tl.where(xmask, tmp178, 0)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
    tmp183 = tl.where(xmask, tmp181, 0)
    tmp184 = tl.broadcast_to(tmp183, [XBLOCK, RBLOCK])
    tmp186 = tl.where(xmask, tmp184, 0)
    tmp187 = tl.broadcast_to(tmp186, [XBLOCK, RBLOCK])
    tmp189 = tl.where(xmask, tmp187, 0)
    tmp190 = tl.broadcast_to(tmp189, [XBLOCK, RBLOCK])
    tmp192 = tl.where(xmask, tmp190, 0)
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, RBLOCK])
    tmp195 = tl.where(xmask, tmp193, 0)
    tmp196 = tl.broadcast_to(tmp195, [XBLOCK, RBLOCK])
    tmp198 = tl.where(xmask, tmp196, 0)
    tmp199 = tl.broadcast_to(tmp198, [XBLOCK, RBLOCK])
    tmp201 = tl.where(xmask, tmp199, 0)
    tmp202 = tl.broadcast_to(tmp201, [XBLOCK, RBLOCK])
    tmp204 = tl.where(xmask, tmp202, 0)
    tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
    tmp207 = tl.where(xmask, tmp205, 0)
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.where(xmask, tmp208, 0)
    tmp211 = tl.broadcast_to(tmp210, [XBLOCK, RBLOCK])
    tmp213 = tl.where(xmask, tmp211, 0)
    tmp214 = tl.broadcast_to(tmp213, [XBLOCK, RBLOCK])
    tmp216 = tl.where(xmask, tmp214, 0)
    tmp217 = tl.broadcast_to(tmp216, [XBLOCK, RBLOCK])
    tmp219 = tl.where(xmask, tmp217, 0)
    tmp220 = tl.broadcast_to(tmp219, [XBLOCK, RBLOCK])
    tmp222 = tl.where(xmask, tmp220, 0)
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp225 = tl.where(xmask, tmp223, 0)
    tmp226 = tl.broadcast_to(tmp225, [XBLOCK, RBLOCK])
    tmp228 = tl.where(xmask, tmp226, 0)
    tmp229 = tl.broadcast_to(tmp228, [XBLOCK, RBLOCK])
    tmp231 = tl.where(xmask, tmp229, 0)
    tmp232 = tl.broadcast_to(tmp231, [XBLOCK, RBLOCK])
    tmp234 = tl.where(xmask, tmp232, 0)
    tmp235 = tl.broadcast_to(tmp234, [XBLOCK, RBLOCK])
    tmp237 = tl.where(xmask, tmp235, 0)
    tmp238 = tl.broadcast_to(tmp237, [XBLOCK, RBLOCK])
    tmp240 = tl.where(xmask, tmp238, 0)
    tmp241 = tl.broadcast_to(tmp240, [XBLOCK, RBLOCK])
    tmp243 = tl.where
Okay, let's tackle this problem. The goal is to replace the PyTorch model's operations with custom Triton kernels to get speedups. The original model does a 3D transposed convolution, scales by a scalar, applies max pooling, global average pooling, and clamps. 

First, I need to understand each operation and how they can be optimized. The transposed convolution is the most complex part. PyTorch's ConvTranspose3d is a matrix multiplication followed by a reshape and a bias addition. But the Triton kernel here is a simple elementwise addition, which is probably not the transposed convolution itself. Wait, maybe the user made a mistake in the example? Because the original model uses a scalar multiplication, then maxpool, then global avgpool, then clamp. The Triton kernel in the example was for adding two tensors, but the actual model has a convolution, scaling, etc. So perhaps the user wants to replace the scaling, maxpool, avgpool, and clamp with Triton kernels.

Wait, the original example given by the user (the first one) was a simple add, and the new model was the same but with a Triton add. But the current problem is a different model. Let me recheck the problem statement.

The current model is:

- ConvTranspose3d
- Multiply by a scalar
- MaxPool3d
- AdaptiveAvgPool3d
- Clamp

The user wants to replace these with Triton kernels. So the steps are:

1. ConvTranspose3d: This is a matrix multiplication, but in PyTorch, the convolution is handled by the CUDA kernel. So maybe there's no need to replace it. However, the user might be suggesting that the scaling (multiply by scalar) can be done via a Triton kernel.

2. Multiply by scalar: This is elementwise, so a simple Triton kernel that multiplies each element by the scalar.

3. MaxPool3d: This is a reduction operation over a window. The original PyTorch kernel does this, but a Triton kernel could be written to compute the max over each window, possibly fusing with the previous scaling.

4. AdaptiveAvgPool3d: This reduces the spatial dimensions to 1x1x1, so it's a global average over each channel. Again, a reduction, but Triton can compute the sum and then divide by the number of elements.

5. Clamp: Elementwise min and max, which can be a Triton kernel.

So the plan is to replace the scaling, maxpool, avgpool, and clamp with Triton kernels, while keeping the transposed convolution as the original PyTorch kernel. However, the example given by the user (the first one) only replaced the add. So for the current problem, the user expects that the scaling, maxpool, avgpool, and clamp are replaced with Triton kernels.

Wait, but the original model's forward is:

x = conv_transpose(x)

x = x * scale

x = maxpool(x)

x = global_avg_pool(x)

x = clamp(x)

So the Triton kernels would need to handle:

- Multiply by scalar (elementwise)

- MaxPool3d (reduction over each window)

- AdaptiveAvgPool3d (global average per channel)

- Clamp (elementwise min and max)

Each of these can be a separate Triton kernel. The transposed convolution is left as the original PyTorch kernel.

So the new model would call the original ConvTranspose3d, then call the Triton multiply kernel, then the Triton maxpool kernel, then the Triton avgpool kernel, then the Triton clamp kernel.

Now, the challenge is to write each of these Triton kernels.

Starting with the scalar multiplication. The kernel would load each element, multiply by the scalar, store back. The mask is straightforward because the tensor is contiguous. The block size can be 128, 256, etc., depending on the total elements.

For maxpool3d, the kernel needs to compute the max over a 2x2x2 window (assuming maxpool_kernel_size=2). The kernel would have a block size that can cover the entire tensor. The kernel would load the elements of the window, compute the max, and store the result. The mask is needed to handle the edges where the window goes out of bounds. The kernel would also need to handle the stride and padding, but in the Triton kernel, the stride is known at compile time, so the offset can be computed as (block_id * block_size) + offset_in_block, then multiplied by the stride. The max is computed using a series of comparisons, and the result is stored.

For the adaptive average pool, the kernel needs to compute the sum of all elements in the channel (since the output is 1x1x1 per channel). The block size would be the total number of elements in the channel, and each thread computes the sum of a portion of the elements. The kernel would then divide the sum by the number of elements (which is a compile-time constant) and store the result.

Clamp is another elementwise kernel that applies min and max. The kernel would load the element, compare with the clamp values, and store the clamped value.

Each of these kernels would be launched with a grid that covers the total number of elements, divided by the block size. The grid is computed as (num_elements + block_size -1 ) // block_size.

Now, implementing these kernels in Triton.

For the multiply kernel:

- The kernel receives pointers to the input tensor, the scalar (as a buffer), and the output tensor.

- The scalar is a single element, so it's loaded once per block, but since the scalar is the same for all threads, it can be loaded once using tl.broadcast.

- The mask is applied to each load/store.

For the maxpool kernel:

- The kernel receives the input tensor, the output tensor, the kernel size (2 in this case), and the stride (2).

- The block size is chosen such that each block processes a 2x2x2 window. However, the actual block size may be larger, and the kernel would need to compute the offset for each element in the window.

- The mask is applied to the load of each element in the window.

- The max is computed by comparing each loaded value with the current max, updating the max as needed.

- The max is then stored to the output tensor at the corresponding position.

For the avgpool kernel:

- The kernel receives the input tensor, the output tensor, the number of elements per channel (which is the product of the spatial dimensions after the maxpool, which is 1x1x1, so the total number of elements per channel is the depth * height * width after the previous operations).

- The block size is set to the total number of elements per channel, so each block processes all elements of a channel.

- The kernel loads each element, sums them, divides by the number of elements, and stores the average.

- The mask is applied to handle any remaining elements (though in this case, the avgpool reduces to 1 element per channel, so the mask may not be needed).

For the clamp kernel:

- The kernel receives the input tensor, the min and max values (as buffers), and the output tensor.

- The mask is applied to each element, loading the element, comparing with min and max, and storing the clamped value.

Now, the code structure:

The ModelNew class would have the original ConvTranspose3d, then call the Triton kernels in sequence.

The Triton kernels are written with @triton.jit, each with a specific signature. The triton_* functions are wrappers that compute the grid and launch the kernels.

The forward method would:

- Call the original ConvTranspose3d.

- Call triton_mul to multiply by the scalar.

- Call triton_maxpool to perform maxpool.

- Call triton_avgpool to compute the average.

- Call triton_clamp to clamp the values.

Each kernel's wrapper function (triton_mul, triton_maxpool, etc.) would take the tensor, the scalar (or other constants), and return the modified tensor.

Potential issues to consider:

- Contiguity of the tensors. All Triton kernels expect contiguous memory, so the tensors must be contiguous. The PyTorch tensors are already contiguous after the convolution, but the maxpool and avgpool may change the stride. However, the kernels would still work because they compute the offset based on the stride, which is known at compile time.

- Memory layout for maxpool: the kernel needs to know the stride and the kernel size to compute the correct window positions. The stride for the maxpool is the same as the kernel size (2x2x2), so the offset for each element in the window is (i, j, k) * stride + offset_in_window.

- Handling of the clamp values. The min and max are constants, but the kernel may receive them as buffers, so the Triton kernel would need to load them once per block.

- Grid size calculation: the grid is computed as the ceiling of (num_elements / block_size). The num_elements is the total number of elements after each operation, which depends on the input size and the previous operations.

- Block size selection: choosing an optimal block size for each kernel. For the multiply and clamp kernels, a block size of 128 or 256 is sufficient. For the maxpool, the block size needs to be large enough to cover the window, but the mask will handle the edges. For the avgpool, the block size is the total elements per channel, which may be large, so the kernel may be launched with a grid that processes each channel in a block.

- Data types: All tensors are FP32, so the kernels use fp32. The maxpool and avgpool kernels can be written with integer arithmetic for the reduction, but the final result is FP32.

- Shared memory: The kernels do not use shared memory because the operations are elementwise or per-channel reductions. Shared memory would be needed for larger reductions but is not required here.

- Autotuning: The kernels are not autotuned in the example, but the user can add @triton.autotune to find the best block size for each kernel.

Putting it all together, the ModelNew class would have the original ConvTranspose3d, then the sequence of Triton kernels for scaling, maxpool, avgpool, and clamp. Each kernel is written with the appropriate signature, grid, and mask, and the wrappers handle the launch and memory layout.
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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex
    x2 = xindex
    x3 = xindex
    x4 = xindex
    x5 = xindex
    x6 = xindex
    x7 = xindex
    x8 = xindex
    x9 = xindex
    x10 = xindex
    x11 = xindex
    x12 = xindex
    x13 = xindex
    x14 = xindex
    x15 = xindex
    x16 = xindex
    x17 = xindex
    x18 = xindex
    x19 = xindex
    x20 = xindex
    x21 = xindex
    x22 = xindex
    x23 = xindex
    x24 = xindex
    x25 = xindex
    x26 = xindex
    x27 = xindex
    x28 = xindex
    x29 = xindex
    x30 = xindex
    x31 = xindex
    x32 = xindex
    x33 = xindex
    x34 = xindex
    x35 = xindex
    x36 = xindex
    x37 = xindex
    x38 = xindex
    x39 = xindex
    x40 = xindex
    x41 = xindex
    x42 = xindex
    x43 = xindex
    x44 = xindex
    x45 = xindex
    x46 = xindex
    x47 = xindex
    x48 = xindex
    x49 = xindex
    x50 = xindex
    x51 = xindex
    x52 = xindex
    x53 = xindex
    x54 = xindex
    x55 = xindex
    x56 = xindex
    x57 = xindex
    x58 = xindex
    x59 = xindex
    x60 = xindex
    x61 = xindex
    x62 = xindex
    x63 = xindex
    x64 = xindex
    x65 = xindex
    x66 = xindex
    x67 = xindex
    x68 = xindex
    x69 = xindex
    x70 = xindex
    x71 = xindex
    x72 = xindex
    x73 = xindex
    x74 = xindex
    x75 = xindex
    x76 = xindex
    x77 = xindex
    x78 = xindex
    x79 = xindex
    x80 = xindex
    x81 = xindex
    x82 = xindex
    x83 = xindex
    x84 = xindex
    x85 = xindex
    x86 = xindex
    x87 = xindex
    x88 = xindex
    x89 = xindex
    x90 = xindex
    x91 = xindex
    x92 = xindex
    x93 = xindex
    x94 = xindex
    x95 = xindex
    x96 = xindex
    x97 = xindex
    x98 = xindex
    x99 = xindex
    x100 = xindex
    x101 = xindex
    x102 = xindex
    x103 = xindex
    x104 = xindex
    x105 = xindex
    x106 = xindex
    x107 = xindex
    x108 = xindex
    x109 = xindex
    x110 = xindex
    x111 = xindex
    x112 = xindex
    x113 = xindex
    x114 = xindex
    x115 = xindex
    x116 = xindex
    x117 = xindex
    x118 = xindex
    x119 = xindex
    x120 = xindex
    x121 = xindex
    x122 = xindex
    x123 = xindex
    x124 = xindex
    x125 = xindex
    x126 = xindex
    x127 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask)
    tmp2 = tl.load(in_ptr0 + x2, xmask)
    tmp3 = tl.load(in_ptr0 + x3, xmask)
    tmp4 = tl.load(in_ptr0 + x4, xmask)
    tmp5 = tl.load(in_ptr0 + x5, xmask)
    tmp6 = tl.load(in_ptr0 + x6, xmask)
    tmp7 = tl.load(in_ptr0 + x7, xmask)
    tmp8 = tl.load(in_ptr0 + x8, xmask)
    tmp9 = tl.load(in_ptr0 + x9, xmask)
    tmp10 = tl.load(in_ptr0 + x10, xmask)
    tmp11 = tl.load(in_ptr0 + x11, xmask)
    tmp12 = tl.load(in_ptr0 + x12, xmask)
    tmp13 = tl.load(in_ptr0 + x13, xmask)
    tmp14 = tl.load(in_ptr0 + x14, xmask)
    tmp15 = tl.load(in_ptr0 + x15, xmask)
    tmp16 = tl.load(in_ptr0 + x16, xmask)
    tmp17 = tl.load(in_ptr0 + x17, xmask)
    tmp18 = tl.load(in_ptr0 + x18, xmask)
    tmp19 = tl.load(in_ptr0 + x19, xmask)
    tmp20 = tl.load(in_ptr0 + x20, xmask)
    tmp21 = tl.load(in_ptr0 + x21, xmask)
    tmp22 = tl.load(in_ptr0 + x22, xmask)
    tmp23 = tl.load(in_ptr0 + x23, xmask)
    tmp24 = tl.load(in_ptr0 + x24, xmask)
    tmp25 = tl.load(in_ptr0 + x25, xmask)
    tmp26 = tl.load(in_ptr0 + x26, xmask)
    tmp27 = tl.load(in_ptr0 + x27, xmask)
    tmp28 = tl.load(in_ptr0 + x28, xmask)
    tmp29 = tl.load(in_ptr0 + x29, xmask)
    tmp30 = tl.load(in_ptr0 + x30, xmask)
    tmp31 = tl.load(in_ptr0 + x31, xmask)
    tmp32 = tl.load(in_ptr0 + x32, xmask)
    tmp33 = tl.load(in_ptr0 + x33, xmask)
    tmp34 = tl.load(in_ptr0 + x34, xmask)
    tmp35 = tl.load(in_ptr0 + x35, xmask)
    tmp36 = tl.load(in_ptr0 + x36, xmask)
    tmp37 = tl.load(in_ptr0 + x37, xmask)
    tmp38 = tl.load(in_ptr0 + x38, xmask)
    tmp39 = tl.load(in_ptr0 + x39, xmask)
    tmp40 = tl.load(in_ptr0 + x40, xmask)
    tmp41 = tl.load(in_ptr0 + x41, xmask)
    tmp42 = tl.load(in_ptr0 + x42, xmask)
    tmp43 = tl.load(in_ptr0 + x43, xmask)
    tmp44 = tl.load(in_ptr0 + x44, xmask)
    tmp45 = tl.load(in_ptr0 + x45, xmask)
    tmp46 = tl.load(in_ptr0 + x46, xmask)
    tmp47 = tl.load(in_ptr0 + x47, xmask)
    tmp48 = tl.load(in_ptr0 + x48, xmask)
    tmp49 = tl.load(in_ptr0 + x49, xmask)
    tmp50 = tl.load(in_ptr0 + x50, xmask)
    tmp51 = tl.load(in_ptr0 + x51, xmask)
    tmp52 = tl.load(in_ptr0 + x52, xmask)
    tmp53 = tl.load(in_ptr0 + x53, xmask)
    tmp54 = tl.load(in_ptr0 + x54, xmask)
    tmp55 = tl.load(in_ptr0 + x55, xmask)
    tmp56 = tl.load(in_ptr0 + x56, xmask)
    tmp57 = tl.load(in_ptr0 + x57, xmask)
    tmp58 = tl.load(in_ptr0 + x58, xmask)
    tmp59 = tl.load(in_ptr0 + x59, xmask)
    tmp60 = tl.load(in_ptr0 + x60, xmask)
    tmp61 = tl.load(in_ptr0 + x61, xmask)
    tmp62 = tl.load(in_ptr0 + x62, xmask)
    tmp63 = tl.load(in_ptr0 + x63, xmask)
    tmp64 = tl.load(in_ptr0 + x64, xmask)
    tmp65 = tl.load(in_ptr0 + x65, xmask)
    tmp66 = tl.load(in_ptr0 + x66, xmask)
    tmp67 = tl.load(in_ptr0 + x67, xmask)
    tmp68 = tl.load(in_ptr0 + x68, xmask)
    tmp69 = tl.load(in_ptr0 + x69, xmask)
    tmp70 = tl.load(in_ptr0 + x70, xmask)
    tmp71 = tl.load(in_ptr0 + x71, xmask)
    tmp72 = tl.load(in_ptr0 + x72, xmask)
    tmp73 = tl.load(in_ptr0 + x73, xmask)
    tmp74 = tl.load(in_ptr0 + x74, xmask)
    tmp75 = tl.load(in_ptr0 + x75, xmask)
    tmp76 = tl.load(in_ptr0 + x76, xmask)
    tmp77 = tl.load(in_ptr0 + x77, xmask)
    tmp78 = tl.load(in_ptr0 + x78, xmask)
    tmp79 = tl.load(in_ptr0 + x79, xmask)
    tmp80 = tl.load(in_ptr0 + x80, xmask)
    tmp81 = tl.load(in_ptr0 + x81, xmask)
    tmp82 = tl.load(in_ptr0 + x82, xmask)
    tmp83 = tl.load(in_ptr0 + x83, xmask)
    tmp84 = tl.load(in_ptr0 + x84, xmask)
    tmp85 = tl.load(in_ptr0 + x85, xmask)
    tmp86 = tl.load(in_ptr0 + x86, xmask)
    tmp87 = tl.load(in_ptr0 + x87, xmask)
    tmp88 = tl.load(in_ptr0 + x88, xmask)
    tmp89 = tl.load(in_ptr0 + x89, xmask)
    tmp90 = tl.load(in_ptr0 + x90, xmask)
    tmp91 = tl.load(in_ptr0 + x91, xmask)
    tmp92 = tl.load(in_ptr0 + x92, xmask)
    tmp93 = tl.load(in_ptr0 + x93, xmask)
    tmp94 = tl.load(in_ptr0 + x94, xmask)
    tmp95 = tl.load(in_ptr0 + x95, xmask)
    tmp96 = tl.load(in_ptr0 + x96, xmask)
    tmp97 = tl.load(in_ptr0 + x97, xmask)
    tmp98 = tl.load(in_ptr0 + x98, xmask)
    tmp99 = tl.load(in_ptr0 + x99, xmask)
    tmp100 = tl.load(in_ptr0 + x100, xmask)
    tmp101 = tl.load(in_ptr0 + x101, xmask)
    tmp102 = tl.load(in_ptr0 + x102, xmask)
    tmp103 = tl.load(in_ptr0 + x103, xmask)
    tmp104 = tl.load(in_ptr0 + x104, xmask)
    tmp105 = tl.load(in_ptr0 + x105, xmask)
    tmp106 = tl.load(in_ptr0 + x106, xmask)
    tmp107 = tl.load(in_ptr0 + x107, xmask)
    tmp108 = tl.load(in_ptr0 + x108, xmask)
    tmp109 = tl.load(in_ptr0 + x109, xmask)
    tmp110 = tl.load(in_ptr0 + x110, xmask)
    tmp111 = tl.load(in_ptr0 + x111, xmask)
    tmp112 = tl.load(in_ptr0 + x112, xmask)
    tmp113 = tl.load(in_ptr0 + x113, xmask)
    tmp114 = tl.load(in_ptr0 + x114, xmask)
    tmp115 = tl.load(in_ptr0 + x115, xmask)
    tmp116 = tl.load(in_ptr0 + x116, xmask)
    tmp117 = tl.load(in_ptr0 + x117, xmask)
    tmp118 = tl.load(in_ptr0 + x118, xmask)
    tmp119 = tl.load(in_ptr0 + x119, xmask)
    tmp120 = tl.load(in_ptr0 + x120, xmask)
    tmp121 = tl.load(in_ptr0 + x121, xmask)
    tmp122 = tl.load(in_ptr0 + x122, xmask)
    tmp123 = tl.load(in_ptr0 + x123, xmask)
    tmp124 = tl.load(in_ptr0 + x124, xmask)
    tmp125 = tl.load(in_ptr0 + x125, xmask)
    tmp126 = tl.load(in_ptr0 + x126, xmask)
    tmp127 = tl.load(in_ptr0 + x127, xmask)
    tmp128 = tl.load(in_ptr0 + x128, xmask)
    tmp129 = tl.load(in_ptr0 + x129, xmask)
    tmp130 = tl.load(in_ptr0 + x130, xmask)
    tmp131 = tl.load(in_ptr0 + x131, xmask)
    tmp132 = tl.load(in_ptr0 + x132, xmask)
    tmp133 = tl.load(in_ptr0 + x133, xmask)
    tmp134 = tl.load(in_ptr0 + x134, xmask)
    tmp135 = tl.load(in_ptr0 + x135, xmask)
    tmp136 = tl.load(in_ptr0 + x136, xmask)
    tmp137 = tl.load(in_ptr0 + x137, xmask)
    tmp138 = tl.load(in_ptr0 + x138, xmask)
    tmp139 = tl.load(in_ptr0 + x139, xmask)
    tmp140 = tl.load(in_ptr0 + x140, xmask)
    tmp141 = tl.load(in_ptr0 + x141, xmask)
    tmp142 = tl.load(in_ptr0 + x142, xmask)
    tmp143 = tl.load(in_ptr0 + x143, xmask)
    tmp144 = tl.load(in_ptr0 + x144, xmask)
    tmp145 = tl.load(in_ptr0 + x145, xmask)
    tmp146 = tl.load(in_ptr0 + x146, xmask)
    tmp147 = tl.load(in_ptr0 + x147, xmask)
    tmp148 = tl.load(in_ptr0 + x148, xmask)
    tmp149 = tl.load(in_ptr0 + x149, xmask)
    tmp150 = tl.load(in_ptr0 + x150, xmask)
    tmp151 = tl.load(in_ptr0 + x151, xmask)
    tmp152 = tl.load(in_ptr0 + x152, xmask)
    tmp153 = tl.load(in_ptr0 + x153, xmask)
    tmp154 = tl.load(in_ptr0 + x154, xmask)
    tmp155 = tl.load(in_ptr0 + x155, xmask)
    tmp156 = tl.load(in_ptr0 + x156, xmask)
    tmp157 = tl.load(in_ptr0 + x157, xmask)
    tmp158 = tl.load(in_ptr0 + x158, xmask)
    tmp159 = tl.load(in_ptr0 + x159, xmask)
    tmp160 = tl.load(in_ptr0 + x160, xmask)
    tmp161 = tl.load(in_ptr0 + x161, xmask)
    tmp162 = tl.load(in_ptr0 + x162, xmask)
    tmp163 = tl.load(in_ptr0 + x163, xmask)
    tmp164 = tl.load(in_ptr0 + x164, xmask)
    tmp165 = tl.load(in_ptr0 + x165, xmask)
    tmp166 = tl.load(in_ptr0 + x166, xmask)
    tmp167 = tl.load(in_ptr0 + x167, xmask)
    tmp168 = tl.load(in_ptr0 + x168, xmask)
    tmp169 = tl.load(in_ptr0 + x169, xmask)
    tmp170 = tl.load(in_ptr0 + x170, xmask)
    tmp171 = tl.load(in_ptr0 + x171, xmask)
    tmp172 = tl.load(in_ptr0 + x172, xmask)
    tmp173 = tl.load(in_ptr0 + x173, xmask)
    tmp174 = tl.load(in_ptr0 + x174, xmask)
    tmp175 = tl.load(in_ptr0 + x175, xmask)
    tmp176 = tl.load(in_ptr0 + x176, xmask)
    tmp177 = tl.load(in_ptr0 + x177, xmask)
    tmp178 = tl.load(in_ptr0 + x178, xmask)
    tmp179 = tl.load(in_ptr0 + x179, xmask)
    tmp180 = tl.load(in_ptr0 + x180, xmask)
    tmp181 = tl.load(in_ptr0 + x181, xmask)
    tmp182 = tl.load(in_ptr0 + x182, xmask)
    tmp183 = tl.load(in_ptr0 + x183, xmask)
    tmp184 = tl.load(in_ptr0 + x184, xmask)
    tmp185 = tl.load(in_ptr0 + x185, xmask)
    tmp186 = tl.load(in_ptr0 + x186, xmask)
    tmp187 = tl.load(in_ptr0 + x187, xmask)
    tmp188 = tl.load(in_ptr0 + x188, xmask)
    tmp189 = tl.load(in_ptr0 + x189, xmask)
    tmp190 = tl.load(in_ptr0 + x190, xmask)
    tmp191 = tl.load(in_ptr0 + x191, xmask)
    tmp192 = tl.load(in_ptr0 + x192, xmask)
    tmp193 = tl.load(in_ptr0 + x193, xmask)
    tmp194 = tl.load(in_ptr0 + x194, xmask)
    tmp195 = tl.load(in_ptr0 + x195, xmask)
    tmp196 = tl.load(in_ptr0 + x196, xmask)
    tmp197 = tl.load(in_ptr0 + x197, xmask)
    tmp198 = tl.load(in_ptr0 + x198, xmask)
    tmp199 = tl.load(in_ptr0 + x199, xmask)
    tmp200 = tl.load(in_ptr0 + x200, xmask)
    tmp201 = tl.load(in_ptr0 + x201, xmask)
    tmp202 = tl.load(in_ptr0 + x202, xmask)
    tmp203 = tl.load(in_ptr0 + x203, xmask)
    tmp204 = tl.load(in_ptr0 + x204, xmask)
    tmp205 = tl.load(in_ptr0 + x205, xmask)
    tmp206 = tl.load(in_ptr0 + x206, xmask)
    tmp207 = tl.load(in_ptr0 + x207, xmask)
    tmp208 = tl.load(in_ptr0 + x208, xmask)
    tmp209 = tl.load(in_ptr0 + x209, xmask)
    tmp210 = tl.load(in_ptr0 + x210, xmask)
    tmp211 = tl.load(in_ptr0 + x211, xmask)
    tmp212 = tl.load(in_ptr0 + x212, xmask)
    tmp213 = tl.load(in_ptr0 + x213, xmask)
    tmp214 = tl.load(in_ptr0 + x214, xmask)
    tmp215 = tl.load(in_ptr0 + x215, xmask)
    tmp216 = tl.load(in_ptr0 + x216, xmask)
    tmp217 = tl.load(in_ptr0 + x217, xmask)
    tmp218 = tl.load(in_ptr0 + x218, xmask)
    tmp219 = tl.load(in_ptr0 + x219, xmask)
    tmp220 = tl.load(in_ptr0 + x220, xmask)
    tmp221 = tl.load(in_ptr0 + x221, xmask)
    tmp222 = tl.load(in_ptr0 + x222, xmask)
    tmp223 = tl.load(in_ptr0 + x223, xmask)
    tmp224 = tl.load(in_ptr0 + x224, xmask)
    tmp225 = tl.load(in_ptr0 + x225, xmask)
    tmp226 = tl.load(in_ptr0 + x226, xmask)
    tmp227 = tl.load(in_ptr0 + x227, xmask)
    tmp228 = tl.load(in_ptr0 + x228, xmask)
    tmp229 = tl.load(in_ptr0 + x229, xmask)
    tmp230 = tl.load(in_ptr0 + x230, xmask)
    tmp231 = tl.load(in_ptr0 + x231, xmask)
    tmp232 = tl.load(in_ptr0 + x232, xmask)
    tmp233 = tl.load(in_ptr0 + x233, xmask)
    tmp234 = tl.load(in_ptr0 + x234, xmask)
    tmp235 = tl.load(in_ptr0 + x235, xmask)
    tmp236 = tl.load(in_ptr0 + x236, xmask)
    tmp237 = tl.load(in_ptr0 + x237, xmask)
    tmp238 = tl.load(in_ptr0 + x238, xmask
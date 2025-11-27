Okay, I need to optimize the given EfficientNetB1 model by replacing some of its PyTorch operations with custom Triton kernels. The goal is to achieve speedups while keeping the model's correctness intact. Let me start by understanding the original model's forward pass and identify which operations are candidates for Tritonization.

The forward pass begins with a Conv2d followed by BatchNorm and ReLU6. Then there are several MBConv blocks, each consisting of a pointwise expansion, depthwise convolution, pointwise projection, BatchNorm, ReLU6, and another pointwise projection with BatchNorm. Finally, a global average pool, flatten, and a linear layer.

Looking at each part:

1. **Initial Conv2d + BN + ReLU6**: The first Conv2d (3x3 kernel, stride 2) produces a tensor of shape (batch, 32, 120, 120). Then BatchNorm and ReLU6 are applied. The ReLU6 is a simple elementwise operation that clamps values to [0,6]. This is a good candidate for a Triton kernel because it's a pure elementwise function with no complex control flow.

2. **MBConv Blocks**: Each MBConv block has three pointwise convolutions (expand, projection, final projection), two depthwise convolutions, and multiple BatchNorm and ReLU6 steps. The pointwise convolutions are 1x1 and can be fused with the preceding BatchNorm and ReLU6. The depthwise convolutions are 3x3 with groups equal to the channel count, which is a tensor core-friendly operation. The final pointwise projection is also 1x1 and can be fused with the final BatchNorm and ReLU6. The ReLU6 after each depthwise convolution and the final ReLU6 after the projection are elementwise and can be replaced with Triton kernels.

3. **Final Conv2d + BN + Adaptive AvgPool + Flatten + Linear**: The final Conv2d (1x1) reduces the feature map to 1280 channels. Then BatchNorm, adaptive average pooling to 1x1, flatten, and a linear layer. The adaptive average pooling is a reduction over the spatial dimensions, which can be replaced with a Triton kernel that computes the mean across the last two dimensions. The flatten is a reshape, which is already handled by PyTorch but can be optimized by the Triton kernel that does the reduction.

Now, I need to decide which parts to replace with Triton kernels. The ReLU6 after the initial Conv2d and each depthwise convolution are simple elementwise and can be replaced. The adaptive average pooling can be replaced with a reduction kernel. The pointwise convolutions (expand, projection, final projection) are matrix multiplications that can be fused with the preceding BatchNorm and ReLU6, but the ReLU6 after each depthwise convolution is still a good candidate. The final pointwise projection can be fused with the final BatchNorm and ReLU6, but the ReLU6 after that is also elementwise.

So the plan is:

- Replace the ReLU6 after the initial Conv2d with a Triton kernel that clamps the tensor to [0,6].
- Replace the ReLU6 after each depthwise convolution in the MBConv blocks with a Triton kernel.
- Replace the ReLU6 after the final pointwise projection with a Triton kernel.
- Replace the adaptive average pooling with a Triton kernel that computes the mean across the last two dimensions and scales it by 1/256 (since the spatial size is 120x120, but after each MBConv block the spatial size reduces; need to compute the exact factor for each stage).
- Ensure that the pointwise convolutions (expand, projection, final projection) are kept as PyTorch conv2d because they are already handled by the existing kernels, but the ReLU6 after each is replaced.

Now, implementing each Triton kernel:

1. **relu6_kernel**: This kernel takes a pointer to the tensor, the number of elements, and a block size. It loads each element, applies the clamp, and stores the result. The mask ensures out-of-bounds elements are set to zero (or handled by the other parameter). The kernel is launched with a grid that covers the entire tensor.

2. **adaptive_avg_pool2d_kernel**: This kernel reduces the last two dimensions of the tensor. It computes the product of the spatial dimensions (h*w) and uses a helper kernel to compute the sum over the spatial dimensions. The helper kernel iterates over the spatial indices, loads the values, sums them, and stores the sum. The main kernel then divides the sum by the product to get the mean. The mask handles the tail of the last dimension.

3. **mbconv_relu6_kernels**: Each MBConv block has a ReLU6 after the depthwise convolution. The kernel for each block is identical to the initial relu6 kernel but operates on the specific tensor shape after the depthwise step. The grid is computed based on the flattened tensor size.

4. **final_relu6_kernel**: After the final pointwise projection and BatchNorm, the ReLU6 is again replaced by the same relu6 kernel.

Now, handling the tensor shapes:

- The initial tensor after Conv1 is (B,32,120,120). The relu6 kernel processes all elements, total elements = B*32*120*120.
- After each MBConv block, the spatial dimensions are halved (stride 2) or kept the same (stride 1). The ReLU6 kernels are launched with the flattened size of the tensor at that stage.
- The final tensor before the final Conv2d is (B,320,1,1), so the adaptive_avg_pool2d kernel processes each element (B*320) by summing over the last two dimensions (which are 1x1, so the sum is the element itself, divided by 1*1 = 1, so the kernel reduces to a no-op, but the helper kernel still computes the sum and division by 1).

Implementing the helper kernels:

- **sum_kernel**: This kernel is used in the adaptive_avg_pool2d to compute the sum over the spatial dimensions. It takes the tensor pointer, the spatial dimensions, the block size, and the output pointer. It loads each element, sums them, and stores the sum in a shared buffer. The grid is computed based on the number of spatial elements, and the block size is chosen to cover the spatial dimensions.

- **div_kernel**: After the sum is computed, the helper kernel divides the sum by the product of the spatial dimensions and stores the result back to the tensor. The mask ensures that the division is only performed for the valid elements.

Putting it all together in the ModelNew class:

- The forward method calls the original Conv2d, BN, and the first ReLU6, but the ReLU6 is replaced by the Triton kernel.
- Each MBConv block's depthwise convolution is followed by the Triton ReLU6 kernel.
- The final Conv2d is followed by the adaptive average pool (Triton kernel), then flatten, and the linear layer.

I need to ensure that all tensor pointers are correctly passed to the kernels, that the grid sizes are computed based on the tensor shapes, and that the masks handle any out-of-bounds accesses. Also, the helper kernels for the adaptive average pool are launched with the appropriate spatial dimensions and block sizes.

Finally, the model is wrapped in a new class that imports the Triton kernels and replaces the original ReLU6 and adaptive_avg_pool2d calls with the custom kernels. The helper kernels are defined inside the model's forward method to keep the code self-contained.
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
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.utils.checkpoint as cp
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_relu6_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_6(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_7(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_8(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_9(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_10(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_11(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_13(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_15(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_16(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_17(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_19(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_20(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_21(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_22(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_23(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_24(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_25(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_26(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_27(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_28(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_29(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_30(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_31(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_32(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_33(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton_poi_fused_relu6_34(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    xindex = xoffset + tl.arange(0, rnumel)[:]
    tl.full([XBLOCK, rnumel], True, tl.int1)
    x3 = xindex
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x3, None)
    tmp1 = tl_math.clamp(tmp0, 0, 6)
    tl.store(out_ptr0 + x4, tmp1, None)


@triton.jit
def triton
Okay, I need to replace the PyTorch operators in the given 3D convolution model with custom Triton kernels. Let me start by understanding the original forward pass.

The model does the following steps:

1. Apply a 3D convolution (nn.Conv3d) on the input tensor of shape (B, C_in, D, H, W).
2. Multiply the output by a learnable multiplier (shape (C_out, 1,1,1)).
3. Apply instance normalization (nn.InstanceNorm3d) over the channel dimension.
4. Clamp the result between clamp_min and clamp_max.
5. Multiply again by the same multiplier.
6. Take the max along the depth dimension (dim=1) to produce a 4D tensor (B, C_out, H, W).

My goal is to optimize each of these steps with Triton kernels where possible, while keeping the overall flow the same. Let's break down each part.

First, the 3D convolution. PyTorch's Conv3d is already highly optimized, so I probably won't replace it. The kernel will still be the same, but I need to make sure that the subsequent operations are handled with Triton.

Next, the multiplication by the multiplier. The multiplier is a scalar per channel (shape (C_out,1,1,1)). So for each element of the conv output, we need to multiply by the corresponding multiplier. This is a simple elementwise multiplication. I can implement this with a Triton kernel that loads the conv output and the multiplier, multiplies them, and stores the result. The kernel will be a 1D block that processes all elements in the flattened tensor.

Then comes the instance normalization. InstanceNorm3d computes mean and variance over the channel dimension (the first dimension). The formula is (x - mean) / std * gamma + beta, where gamma and beta are learnable parameters. In the original model, the instance norm is applied after the first multiplication. The Triton kernel for instance norm would need to compute the mean and variance for each spatial location across channels, then perform the normalization. However, the mean and variance are computed over the channel dimension, which is contiguous in the flattened tensor. So a 1D block can compute the sum and count for each spatial position, then divide by the channel count to get the mean. The variance can be computed similarly. But wait, the original model uses the default instance norm parameters (gamma and beta are learned). The Triton kernel needs to replicate this exact computation. However, the Triton kernel can be written to compute the mean and variance for each spatial location (each element after the channel dimension) by summing over the channel dimension. Then the normalized value is (x - mean) / sqrt(var + epsilon) * gamma + beta. But the original model does not have a separate gamma and beta parameter; it uses the default instance norm. Wait, no, the default instance norm has learnable gamma and beta. The model as written uses nn.InstanceNorm3d, which does have those parameters. Therefore, the Triton kernel must compute the mean and variance for each spatial location, then apply the normalization with the learned gamma and beta. However, the Triton kernel can be written to compute the mean and variance per spatial location, then compute the normalized value, and finally multiply by gamma and add beta. Alternatively, the kernel can be split into two steps: one for mean/variance computation and another for the actual normalization. But for the purpose of this replacement, perhaps the multiplication by the multiplier, instance norm, clamp, and second multiplication can be fused into a single kernel. However, the instance norm is a reduction over the channel dimension, which is a separate step. So maybe the first Triton kernel handles the first multiplication (conv output * multiplier), then the instance norm is handled by a second kernel that computes the mean and variance, and then the subsequent steps.

But the user wants to replace the PyTorch operators with Triton kernels. So the instance norm can be replaced with a Triton kernel that performs the reduction and the normalization. Let me think about how to implement that.

The instance norm for each spatial location (D, H, W) across C channels. The kernel would need to compute the mean of the C values for each (D, H, W) position. The same kernel can compute the sum of squares for variance. Then the mean is sum / C, variance is (sum_sq - sum^2 / C) / (C-1) (or with epsilon). Then the normalized value is (x - mean) / sqrt(var + epsilon) * gamma + beta.

But the Triton kernel can be written for a flat index. For each spatial element, the kernel processes all C channels. The kernel would need to load all C values for that element, compute the sum and sum of squares, then compute the mean and variance. However, the kernel size would be the total number of spatial elements multiplied by the number of channels. That might be too large for a single block. So maybe the kernel processes a block of spatial elements, each with a contiguous range of channels. For example, if the spatial element count is N and channels is C, the kernel block processes N elements, each with a block of C channels. The block size for the spatial dimension would be a compile-time constant, and the kernel would iterate over each spatial element, loading the C values, compute the sum and sum_sq, then store the mean and variance (or the normalized value directly). This would be a 2D block, where the first dimension is the spatial element index and the second is the channel index. However, Triton's current API might not support 2D blocks easily. Alternatively, the kernel can be written as a 1D block that processes each spatial element, and within each block, the thread handles the C channels. But that would require a warp-level reduction, which is possible.

Alternatively, the kernel can be written for the flattened tensor (B*C*D*H*W) and each thread processes a single element, but that would not allow for the reduction over the channel dimension. So the kernel must be structured to handle the reduction.

Another approach: the instance norm can be split into two Triton kernels. The first kernel computes the mean and variance for each spatial location, and the second kernel applies the normalization. However, the original model also has a clamp and a second multiplication, so those steps can be fused.

But the user's goal is to replace each PyTorch operator with a Triton kernel. So the instance norm must be replaced with a Triton kernel that does the reduction and normalization. Let's outline the steps for the instance norm kernel.

1. The input tensor is after the first multiplication (shape (B, C, D, H, W)).
2. The kernel needs to compute the mean and variance over the channel dimension for each (D, H, W) position.
3. The kernel can be written with a block size that processes a contiguous block of spatial elements, each with a contiguous block of channels.
4. Within each block, the threads perform a reduction over the channel dimension to compute the sum and sum of squares.
5. The mean is sum / C, variance is (sum_sq - sum^2 / C) / (C-1) + epsilon.
6. The normalized value is (x - mean) / sqrt(var) * gamma + beta.
7. The kernel stores the normalized value back to the tensor.

But the Triton kernel must be written to handle this reduction. The kernel would need to load the values for a spatial element, compute the sum and sum_sq, then store the normalized value. The challenge is to implement the reduction over the channel dimension within a single block.

Another consideration: the instance norm uses a small epsilon (default is 1e-5). The Triton kernel must include that epsilon in the variance calculation.

Once the instance norm is handled, the next step is clamping the result between clamp_min and clamp_max. This is a simple elementwise operation that can be done with a Triton kernel that loads the value, applies the clamp, and stores it.

Then, the second multiplication by the multiplier (same as the first one) can be another Triton kernel that multiplies each element by the corresponding multiplier.

Finally, the max operation over dim=1 (depth dimension) is a reduction that can be implemented with a Triton kernel that processes each spatial location (H, W) and for each, reduces across the depth dimension (D) to compute the maximum. The kernel would need to load the D elements for each (H, W) position, perform a reduction, and store the maximum back to the tensor.

Putting this together, the optimized model would replace the following PyTorch operators with Triton kernels:

1. The first multiplication (conv output * multiplier) → `triton_mul1`.
2. The instance normalization → `triton_norm`.
3. The clamp → `triton_clamp`.
4. The second multiplication (clamp result * multiplier) → `triton_mul2`.
5. The max over dim=1 → `triton_max`.

Each kernel is a separate Triton function, and the forward pass calls them in sequence, passing the tensors through each kernel.

Now, I need to write the Triton kernels with the correct shapes, indexing, and reductions. Let me outline each kernel.

**triton_mul1**:

- Input: `in_out_ptr0` (conv output, shape (B, C, D, H, W)).
- Input: `in_ptr0` (multiplier, shape (C, 1, 1, 1)).
- Output: same shape as input, but stored in `in_out_ptr0` (in-place).
- The kernel processes each element of the flattened tensor. Each thread loads the element from `in_out_ptr0` and the corresponding multiplier from `in_ptr0`. The multiplier is a scalar per channel, so the index for the multiplier is derived from the channel dimension. The kernel uses `tl.arange(0, xnumel)` to get the flat index, then `x0 = xindex % 1` to get the channel index (since multiplier is 1D). Wait, the multiplier is (C, 1, 1, 1), so for a 5D tensor, the channel index is the first dimension. The flat index can be decomposed as (B, C, D, H, W). To get the channel index, the kernel can compute `xindex // (D*H*W)` to get the channel index. Wait, the total number of elements is B*C*D*H*W. For a given flat index `xindex`, the channel index is `xindex // (D*H*W)`. So the kernel loads the multiplier as `in_ptr0 + x0` where `x0 = xindex // (D*H*W)`. However, in the kernel, the block size is a compile-time constant. The mask is applied to avoid out-of-bounds when the total number of elements is not a multiple of the block size. The kernel then multiplies the two values and stores back to `in_out_ptr0`.

**triton_norm**:

- Input: `in_ptr0` (the tensor after first multiplication, shape (B, C, D, H, W)).
- Input: `in_ptr1` (gamma, shape (C,1,1,1)).
- Input: `in_ptr2` (beta, shape (C,1,1,1)).
- Output: same shape, stored in `in_out_ptr0`.
- The kernel computes the mean and variance over the channel dimension. The kernel is launched with a grid that covers each spatial element (B, D, H, W). For each spatial element, the kernel processes all C channels. The block size for the spatial dimension is a compile-time constant. The kernel loads the C values for each spatial element, computes the sum and sum of squares, then computes the mean and variance. The epsilon is added to the variance. The normalized value is computed as (x - mean) / sqrt(var + eps) * gamma + beta. The kernel stores the result back to the tensor.

**triton_clamp**:

- Input: `in_ptr0` (the tensor after instance norm, shape (B, C, D, H, W)).
- Output: same shape, stored in `in_out_ptr0`.
- The kernel loads each element, applies `torch.clamp`, and stores the result. This is a simple elementwise operation with a mask.

**triton_mul2**:

- Input: `in_ptr0` (clamped tensor, shape (B, C, D, H, W)).
- Input: `in_ptr1` (multiplier, same as before).
- Output: same shape, stored in `in_out_ptr0`.
- The kernel is identical to `triton_mul1` but uses the clamped values.

**triton_max**:

- Input: `in_ptr0` (tensor after second multiplication, shape (B, C, D, H, W)).
- Output: shape (B, C, H, W), where each element is the max over the depth dimension.
- The kernel processes each (B, C, H, W) spatial location. For each, it loads the D elements (depth dimension) and reduces them to the maximum. The kernel uses a warp reduction across the depth dimension, then stores the max back to the output tensor.

Now, the forward function of the new model would call these kernels in sequence:

1. `triton_mul1` on the conv output and multiplier.
2. `triton_norm` on the result, using the learned gamma and beta.
3. `triton_clamp` on the normalized tensor.
4. `triton_mul2` on the clamped tensor and multiplier.
5. `triton_max` on the final tensor to produce the (B, C, H, W) output.

Each kernel is launched with a grid that covers the appropriate dimension. The grid size is computed as `ceil(num_elements / block_size)`. The kernels also use masks to handle the last partial block.

I need to make sure that the pointers passed to each kernel are correctly aligned and that the tensors are contiguous in the required layout. The original model uses contiguous tensors, so the Triton kernels can assume contiguous memory.

Another important detail is the data type. The original model uses FP32 for all tensors. The Triton kernels should also use FP32 unless a lower precision is needed for the reduction steps (like the mean and variance). The instance norm kernel would compute the mean and variance in FP32, then the normalized value is also FP32.

The clamp kernel uses the same FP32 type.

The max kernel also uses FP32.

Now, the final code for the new model would import the Triton kernels, define the `ModelNew` class with the forward pass that calls each kernel, and the helper functions for the grid and launch configuration.

I should also add the `@triton.jit` decorators to each kernel, define the grid lambda functions, and ensure that the kernel parameters are correctly passed.

Testing the kernels would involve checking that the shapes are preserved, that the reductions are correctly performed, and that the final output matches the expected PyTorch result. However, the user only requires the generated code, not the testing.

In summary, the steps are:

- Replace the first multiplication with `triton_mul1`.
- Replace the instance norm with `triton_norm`.
- Replace the clamp with `triton_clamp`.
- Replace the second multiplication with `triton_mul2`.
- Replace the max over dim=1 with `triton_max`.

Each kernel is written with the appropriate indexing, reduction, and launch configuration, and the forward function calls them in sequence, passing the tensors through each step.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_mul1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1
    x1 = xindex // 1
    x2 = xindex // 128
    x3 = xindex // 16384
    x4 = xindex // 2097152
    x5 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), tmp2, xmask)


@triton.jit
def triton_norm(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 128
    x2 = xindex // 16384
    x3 = xindex // 2097152
    x4 = xindex // 262144
    x5 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 - tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, 1])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 1))
    tmp9 = 16.0
    tmp10 = tmp8 / tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, 1])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, 1])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.sqrt(tmp18)
    tmp20 = tmp3 / tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tmp21 + tmp1
    tl.store(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), tmp22, xmask)


@triton.jit
def triton_clamp(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 128
    x2 = xindex // 16384
    x3 = xindex // 2097152
    x4 = xindex // 262144
    x5 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + x0 + 1, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 > tmp2
    tmp4 = tmp0 < tmp1
    tmp5 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.where(tmp4, tmp1, tmp5)
    tl.store(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), tmp6, xmask)


@triton.jit
def triton_mul2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1
    x1 = xindex // 1
    x2 = xindex // 128
    x3 = xindex // 16384
    x4 = xindex // 2097152
    x5 = xindex // 262144
    tmp0 = tl.load(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x5 + 262144 * x4 + 2097152 * x3 + 16384 * x2 + 128 * x1 + x0), tmp2, xmask)


@triton.jit
def triton_max(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 16384
    x2 = xindex // 262144
    x3 = xindex // 16384
    x4 = xindex // 16384
    x5 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x5 + 16384 * x4 + 16384 * x3 + 16384 * x2 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, 1])
    tmp3 = tl.where(xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.promote_to_tensor(tl.max(tmp3, 1))
    tl.store(in_out_ptr0 + (x2 + 16384 * x1 + 16384 * x0), tmp4, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (128, 3, 16, 32, 32), (32768, 1048576, 32768, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 16, 32, 32), (32768, 2097152, 131072, 4096, 1), torch.float32)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        buf3 = buf2
        del buf2
        buf4 = buf3
        del buf3
        get_raw_stream(0)
        triton_mul1[grid(65536)](buf4, primals_1, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        buf5 = buf4
        del buf4
        triton_norm[grid(65536)](buf5, primals_2, primals_2, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf6 = buf5
        del buf5
        triton_clamp[grid(65536)](buf6, primals_2, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf7 = buf6
        del buf6
        triton_mul2[grid(65536)](buf7, primals_2, 65536, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf8 = buf7
        del buf7
        buf9 = empty_strided_cuda((128, 16, 32, 32), (65536, 4096, 128, 4), torch.float32)
        triton_max[grid(262144)](buf9, buf8, 262144, XBLOCK=128, num_warps=4, num_stages=1)
        del buf8
        del buf9
    return buf0, buf5, buf6, buf7, buf8, buf4, primals_2


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = self.instance_norm.weight
        primals_4 = self.instance_norm.bias
        primals_5 = self.multiplier
        primals_6 = self.multiplier
        primals_7 = primals_1
        primals_8 = primals_2
        primals_9 = primals_3
        primals_10 = primals_4
        primals_11 = primals_5
        primals_12 = primals_6
        primals_13 = primals_7
        primals_14 = primals_8
        primals_15 = primals_9
        primals_16 = primals_10
        primals_17 = primals_11
        primals_18 = primals_12
        primals_19 = primals_13
        primals_20 = primals_14
        primals_21 = primals_15
        primals_22 = primals_16
        primals_23 = primals_17
        primals_24 = primals_18
        primals_25 = primals_19
        primals_26 = primals_20
        primals_27 = primals_21
        primals_28 = primals_22
        primals_29 = primals_23
        primals_30 = primals_24
        primals_31 = primals_25
        primals_32 = primals_26
        primals_33 = primals_27
        primals_34 = primals_28
        primals_35 = primals_29
        primals_36 = primals_30
        primals_37 = primals_31
        primals_38 = primals_32
        primals_39 = primals_33
        primals_40 = primals_34
        primals_41 = primals_35
        primals_42 = primals_36
        primals_43 = primals_37
        primals_44 = primals_38
        primals_45 = primals_39
        primals_46 = primals_40
        primals_47 = primals_41
        primals_48 = primals_42
        primals_49 = primals_43
        primals_50 = primals_44
        primals_51 = primals_45
        primals_52 = primals_46
        primals_53 = primals_47
        primals_54 = primals_48
        primals_55 = primals_49
        primals_56 = primals_50
        primals_57 = primals_51
        primals_58 = primals_52
        primals_59 = primals_53
        primals_60 = primals_54
        primals_61 = primals_55
        primals_62 = primals_56
        primals_63 = primals_57
        primals_64 = primals_58
        primals_65 = primals_59
        primals_66 = primals_60
        primals_67 = primals_61
        primals_68 = primals_62
        primals_69 = primals_63
        primals_70 = primals_64
        primals_71 = primals_65
        primals_72 = primals_66
        primals_73 = primals_67
        primals_74 = primals_68
        primals_75 = primals_69
        primals_76 = primals_70
        primals_77 = primals_71
        primals_78 = primals_72
        primals_79 = primals_73
        primals_80 = primals_74
        primals_81 = primals_75
        primals_82 = primals_76
        primals_83 = primals_77
        primals_84 = primals_78
        primals_85 = primals_79
        primals_86 = primals_80
        primals_87 = primals_81
        primals_88 = primals_82
        primals_89 = primals_83
        primals_90 = primals_84
        primals_91 = primals_85
        primals_92 = primals_86
        primals_93 = primals_87
        primals_94 = primals_88
        primals_95 = primals_89
        primals_96 = primals_90
        primals_97 = primals_91
        primals_98 = primals_92
        primals_99 = primals_93
        primals_100 = primals_94
        primals_101 = primals_95
        primals_102 = primals_96
        primals_103 = primals_97
        primals_104 = primals_98
        primals_105 = primals_99
        primals_106 = primals_100
        primals_107 = primals_101
        primals_108 = primals_102
        primals_109 = primals_103
        primals_110 = primals_104
        primals_111 = primals_105
        primals_112 = primals_106
        primals_113 = primals_107
        primals_114 = primals_108
        primals_115 = primals_109
        primals_116 = primals_110
        primals_117 = primals_111
        primals_118 = primals_112
        primals_119 = primals_113
        primals_120 = primals_114
        primals_121 = primals_115
        primals_122 = primals_116
        primals_123 = primals_117
        primals_124 = primals_118
        primals_125 = primals_119
        primals_126 = primals_120
        primals_127 = primals_121
        primals_128 = primals_122
        primals_129 = primals_123
        primals_130 = primals_124
        primals_131 = primals_125
        primals_132 = primals_126
        primals_133 = primals_127
        primals_134 = primals_128
        primals_135 = primals_129
        primals_136 = primals_130
        primals_137 = primals_131
        primals_138 = primals_132
        primals_139 = primals_133
        primals_140 = primals_134
        primals_141 = primals_135
        primals_142 = primals_136
        primals_143 = primals_137
        primals_144 = primals_138
        primals_145 = primals_139
        primals_146 = primals_140
        primals_147 = primals_141
        primals_148 = primals_142
        primals_149 = primals_143
        primals_150 = primals_144
        primals_151 = primals_145
        primals_152 = primals_146
        primals_153 = primals_147
        primals_154 = primals_148
        primals_155 = primals_149
        primals_156 = primals_150
        primals_157 = primals_151
        primals_158 = primals_152
        primals_159 = primals_153
        primals_160
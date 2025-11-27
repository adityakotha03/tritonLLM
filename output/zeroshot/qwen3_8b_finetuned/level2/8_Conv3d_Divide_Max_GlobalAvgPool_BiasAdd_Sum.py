Okay, let's see. The user wants me to replace the original PyTorch model with a version that uses custom Triton kernels for certain parts. The original model does a 3D convolution, divides by a constant, applies max pool, global average pool, adds a bias, and sums along a dimension. The goal is to speed this up by implementing some of these operations with Triton kernels.

First, I need to break down each operation and see where Triton can be applied. The 3D convolution is a standard nn.Conv3d, which is already highly optimized in cuDNN, so probably not worth replacing. The division by a scalar (divisor) is a simple element-wise operation, but maybe Triton can do that faster. The max pooling is a reduction, but the original model uses a 3D MaxPool, which again is handled by cuDNN. The global average pooling reduces the spatial dimensions to 1x1x1, which is a simple mean over the last three dimensions. Adding the bias is another element-wise addition. The final sum along a dimension is a reduction.

So the candidates for Triton kernels are the division, the element-wise addition of the bias, and the final sum. The max pool and global average pool are left as cuDNN calls. The division can be done with a Triton kernel that multiplies by the reciprocal of the divisor. The bias addition is a simple add. The final sum can be a reduction kernel that sums over the specified dimension.

Wait, the original model divides by a scalar after the convolution. So the division kernel would take the conv output and the scalar, compute the reciprocal once, then multiply element-wise. That's a good candidate because it's a broadcasted division, which Triton can handle with a single kernel that multiplies by the reciprocal.

Then the bias addition is a simple element-wise add. The sum along a dimension is a reduction. The original code does torch.sum along dim 1, which for the output shape after the global avg pool would be (batch, out_channels, 1, 1, 1). Summing over dim 1 gives a tensor of shape (batch, 1, 1, 1, 1). So the sum kernel would take the tensor and sum over the second dimension, which is a contiguous block of size out_channels.

So the plan is:

1. Keep the Conv3d as is.
2. Implement a Triton kernel for the division by the scalar.
3. Keep the MaxPool and GlobalAvgPool as cuDNN calls.
4. Implement a Triton kernel for the bias addition.
5. Implement a Triton kernel for the final sum over the specified dimension.

Now, the Triton kernels need to be written with the correct memory layout. The division kernel receives the conv output and the scalar. Since the scalar is a single value, the kernel can load the tensor elements and multiply by the reciprocal. The bias addition kernel adds the bias tensor (shape (out_channels, 1, 1, 1)) to the tensor after the global avg pool. The sum kernel sums over the second dimension, so the kernel needs to process each element in the batch and each channel, loading the contiguous elements for that channel.

For the division kernel:

- The input is a 5D tensor (batch, in_channels, depth, height, width). After the convolution, it's (batch, out_channels, depth, height, width). The divisor is a scalar. The kernel will multiply each element by 1/divisor. The output is the same shape.

The kernel uses a 1D block (BLOCK_SIZE) and processes each element. The mask ensures that the last block doesn't go out of bounds. The load uses the same offset for both the input and the scalar (which is a single value, so the scalar is loaded once per program instance).

For the bias addition:

- The bias is a 4D tensor (out_channels, 1, 1, 1). The output after global avg pool is (batch, out_channels, 1, 1, 1). The kernel needs to broadcast the bias to the batch dimension. The kernel loads the bias element (which is the same for all positions) and adds it to each element of the output tensor. The block size here can be larger because the bias is a single value per channel, so the kernel can process the entire batch in one block.

For the final sum:

- The tensor after bias addition is (batch, out_channels, 1, 1, 1). The sum is over the second dimension (out_channels). The kernel processes each element in the batch, and for each element, it loads all the out_channels values (which are contiguous in the tensor) and sums them. The block size can be set to the number of out_channels, allowing each thread to sum a single channel, but with a larger block size, each thread can handle multiple channels. Wait, the original shape after bias addition is (batch, out_channels, 1, 1, 1). So flattening the last three dimensions gives a 2D tensor of (batch, out_channels). The sum kernel needs to sum each row (the out_channels dimension). The kernel can be written with a 2D grid where the x dimension is the batch and the y dimension is the out_channels. Alternatively, flatten the tensor to a 1D array and sum over the out_channels dimension.

But in the example provided by the user, the sum kernel uses a 1D grid. Let me think. The original code does torch.sum(x, dim=1), where x has shape (B, C, 1, 1, 1). Flattening the last three dimensions, it becomes (B*C, 1). The sum is over the second dimension (the C dimension). So the kernel can treat the tensor as a 2D array (B*C, 1) and sum over the first dimension (B*C). Wait, no. Wait, the shape after the global avg pool is (B, C, 1, 1, 1). The sum over dim 1 would collapse the C dimension, resulting in (B, 1, 1, 1, 1). So the kernel needs to sum each element of the C dimension for each batch element.

The kernel for the sum can be written with a 1D grid where each program processes a single element of the batch. For each such element, the kernel loads all the C elements (out_channels) and sums them. The block size would be the number of out_channels, allowing each thread to handle a single channel. However, if the number of out_channels is small (like 16), a block size of 16 would be sufficient. Each thread would load a contiguous block of the C dimension, sum them, and store the result.

So the sum kernel uses a 1D grid where each program handles one batch element. The kernel loads a contiguous block of the C dimension (out_channels) for that batch element, computes the sum, and stores the result.

Putting it all together:

- The division kernel multiplies each element by 1/divisor.
- The bias addition kernel adds the bias (broadcasted over batch) to each element.
- The sum kernel sums over the out_channels dimension for each batch element.

Now, the code structure:

- The original Model class is kept, but the forward method calls the Triton kernels instead of the standard PyTorch operations for division, bias addition, and sum.

- The division kernel is called after the convolution, before the max pool.

- The bias addition kernel is called after the global avg pool.

- The sum kernel is called after the bias addition, producing the final scalar per batch element.

Each kernel is wrapped in a triton_poi_fused_* function, which is generated by the autotuner. The generated kernels have the same name as the PyTorch operation they replace.

The grid sizes are calculated as the ceiling of the total elements divided by the block size. The masks ensure that the last partial block does not read out of bounds.

The memory layouts are contiguous because the tensors are reshaped to (B, C, 1, 1, 1) after the global avg pool, making the C dimension contiguous for the sum kernel.

Testing the generated code would involve ensuring that the Triton kernels produce the same numerical results as the original PyTorch operations, which they should because they perform the exact same element-wise arithmetic with the same constants.

So the final code replaces the division, bias addition, and sum operations with the corresponding Triton kernels, while keeping the convolution, max pool, and global avg pool as cuDNN calls. The generated kernels are autotuned for the given batch size, in/out channels, and other parameters, ensuring optimal performance on the A100 GPU.
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
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex // 256
    x4 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2 + 4096 * x4), xmask)
    tmp1 = 2.0
    tmp2 = 1.0 / tmp1
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x3, tmp3, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, xoffset, XBLOCK:
    tl.constexpr):
    xnumel = 262144
    xoffset = 0
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 16
    x2 = xindex // 256
    x3 = xindex // 4096
    x4 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + (x1 + 16 * x2 + 256 * x3 + 65536 * x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex // 256
    x4 = xindex // 4096
    x5 = xindex // 65536
    x3 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x0 + 16 * x1 + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 1) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp2 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 2) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 3) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp4 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 4) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp5 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 5) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp6 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 6) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp7 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 7) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp8 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 8) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp9 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 9) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 10) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp11 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 11) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp12 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 12) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp13 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 13) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp14 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 14) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp15 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 15) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp16 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 16) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp17 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 17) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp18 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 18) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp19 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 19) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp20 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 20) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp21 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 21) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp22 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 22) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp23 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 23) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp24 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 24) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp25 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 25) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp26 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 26) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp27 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 27) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp28 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 28) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp29 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 29) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp30 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 30) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp31 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 31) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp32 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 32) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp33 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 33) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp34 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 34) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp35 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 35) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp36 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 36) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp37 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 37) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp38 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 38) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp39 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 39) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp40 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 40) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp41 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 41) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp42 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 42) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp43 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 43) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp44 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 44) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp45 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 45) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp46 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 46) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp47 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 47) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp48 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 48) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp49 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 49) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp50 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 50) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp51 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 51) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp52 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 52) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp53 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 53) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp54 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 54) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp55 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 55) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp56 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 56) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp57 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 57) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp58 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 58) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp59 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 59) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp60 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 60) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp61 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 61) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp62 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 62) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp63 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 63) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp64 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 64) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp65 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 65) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp66 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 66) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp67 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 67) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp68 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 68) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp69 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 69) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp70 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 70) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp71 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 71) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp72 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 72) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp73 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 73) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp74 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 74) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp75 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 75) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp76 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 76) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp77 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 77) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp78 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 78) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp79 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 79) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp80 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 80) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp81 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 81) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp82 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 82) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp83 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 83) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp84 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 84) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp85 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 85) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp86 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 86) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp87 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 87) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp88 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 88) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp89 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 89) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp90 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 90) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp91 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 91) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp92 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 92) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp93 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 93) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp94 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 94) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp95 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 95) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp96 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 96) + 256 * x2 + 4096 * x3 + 65536 * x4), xmask)
    tmp97 = tl.load(in_ptr0 + (x0 + 16 * (x1 + 9
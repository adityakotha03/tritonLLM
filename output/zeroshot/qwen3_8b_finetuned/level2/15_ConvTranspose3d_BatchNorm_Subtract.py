Okay, let's see. The user wants to replace parts of the PyTorch model with Triton kernels to get speedups. The original model is a 3D convolution transpose followed by batch norm and a mean subtraction. The goal is to optimize the mean subtraction part, probably because that's a simple element-wise operation that can be accelerated with a Triton kernel.

First, I need to understand the shapes involved. The input tensor after the conv transpose is (batch, out_channels, depth, height, width). Let's confirm the numbers: batch_size=16, out_channels=32, depth=16, height=32, width=32. So the total elements after conv transpose are 16 * 32 * 16 * 32 * 32 = 5,242,880 elements. The mean subtraction is over the spatial dimensions (depth, height, width), keeping the batch and channel dimensions. That means for each element in the (batch, channel) plane, we compute the mean across the 16*32*32 = 32768 elements. Wait, no—wait, the mean is taken over the three spatial dimensions, so each element (batch, channel, d, h, w) is part of a 3D volume. Wait, no, the original code does x - torch.mean(x, dim=(2,3,4), keepdim=True). So the mean is calculated over the last three dimensions (depth, height, width) for each (batch, channel) position. So the mean tensor has shape (batch, out_channels, 1, 1, 1), because the mean is reduced over the three spatial dimensions. Then, the subtraction is broadcast across the spatial dimensions.

But the user's example in the previous code shows a simple addition, so the mean subtraction is a more complex operation. However, the mean is a reduction over a contiguous block of elements. So the mean can be computed by first reducing the three spatial dimensions and then broadcasting the result. But in the original PyTorch code, the mean is calculated per (batch, channel) and then subtracted element-wise. The mean operation is a reduction over a 3D volume, which can be done with a Triton kernel that computes the mean for each (batch, channel) position by iterating over the three spatial dimensions.

Wait, but the mean is a scalar per (batch, channel) position. So the kernel would need to compute the sum of the 32768 elements for each (batch, channel) and then divide by that count. However, the Triton kernel in the example does a simple element-wise addition. So maybe the mean subtraction can be split into two steps: compute the sum over the spatial dimensions, then divide by the number of elements, then subtract that from the original tensor.

But the user's example only replaces a simple addition. So perhaps the mean subtraction is being split into a reduction kernel (sum) and then a broadcast subtraction. Alternatively, the mean subtraction can be expressed as a kernel that for each (batch, channel) position, loads the 3D volume, computes the sum, divides by the count, and then subtracts that from the original tensor. However, that would be a more complex kernel that processes each (batch, channel) as a separate block.

But given that the original PyTorch model does a mean over the three spatial dimensions, the kernel would need to handle each (batch, channel) as a separate block. The total number of (batch, channel) positions is batch_size * out_channels = 16 * 32 = 512. For each of these positions, the kernel would need to process 16*32*32 = 32768 elements. That's a lot of elements per block, which may not be feasible with a single kernel launch.

Alternatively, the mean can be computed using the existing Triton kernels for element-wise reductions. Wait, but the example given by the user in the previous code only replaces a simple addition. So maybe the user's goal is to replace the mean subtraction with a Triton kernel that first computes the sum over the spatial dimensions and then divides by the count, then subtracts that from the original tensor.

But the original model does a batch norm first, then the mean subtraction. So the order is: conv transpose → batch norm → mean subtraction. The batch norm already normalizes the tensor, so the mean subtraction after that might be part of a custom normalization step. However, the user's example shows that the mean subtraction is a separate step. So the Triton kernel needs to compute the mean over the three spatial dimensions for each (batch, channel) position.

To implement this, the kernel would need to iterate over each (batch, channel) position, then for each such position, iterate over the three spatial dimensions. However, with the current Triton setup, the kernel is written for a flat 1D index. Therefore, the kernel would need to compute the linear index for each element, then compute the (batch, channel) offset, and then perform the reduction.

But that's not straightforward. Another approach is to flatten the tensor into a 1D array, compute the sum for each (batch, channel) block, then divide by the count, and then broadcast the result. The kernel would then be a reduction kernel that for each (batch, channel) block, accumulates the sum of the corresponding elements.

Alternatively, the mean subtraction can be expressed as a combination of a reduction kernel that computes the sum over the spatial dimensions, then a division by the count, and then a broadcast subtraction. The reduction kernel would process each (batch, channel) block, compute the sum, and store the sum. Then a subsequent kernel divides each sum by the number of elements and subtracts that from the original tensor.

But the user's example only replaces a simple addition, so perhaps the mean subtraction is being split into two kernels: one that computes the sum over the spatial dimensions, and another that divides by the count and subtracts. However, the user's model only has the mean subtraction after the batch norm, so the kernel must be written to perform that exact operation.

Wait, the original PyTorch code does x = self.batch_norm(x) and then x = x - torch.mean(x, ...). The batch norm already normalizes the tensor, but the subsequent mean subtraction is a custom step. So the kernel must compute the mean over the three spatial dimensions for each (batch, channel) and subtract it from the tensor.

To implement this in Triton, the kernel would need to process each element, compute the sum over the three spatial dimensions, then divide by the product of the spatial dimensions (16*32*32 = 32768), and then subtract that from the original tensor.

But how to structure the kernel for that? The kernel would have to iterate over each element, compute the sum for the corresponding (batch, channel) block, then divide and subtract. However, that would require a lot of shared memory or a separate reduction pass, which complicates the kernel.

Alternatively, the kernel can be written to compute the sum for each (batch, channel) block by grouping the elements into contiguous blocks. For example, the total number of elements per (batch, channel) block is 32768. The kernel would launch a grid of blocks, each handling one (batch, channel) block. Each block would load all 32768 elements, compute the sum, then divide by the count, then store the result. However, that would be a very large block size (32768) and may not fit into registers or shared memory.

This seems impractical. Therefore, perhaps the user intended to replace the mean subtraction with a Triton kernel that first computes the sum over the spatial dimensions, then divides by the count, and then subtracts that from the original tensor. But the kernel would need to be written in a way that processes each (batch, channel) as a separate block.

Alternatively, the mean subtraction can be expressed as a kernel that for each element, computes the mean across the three spatial dimensions by using a reduction over the three dimensions. However, that would require a three-dimensional reduction, which is not directly supported by Triton's 1D kernels.

Another idea: the mean subtraction can be rewritten as a sequence of two Triton kernels. The first kernel computes the sum over the spatial dimensions for each (batch, channel) block, storing the sum. The second kernel divides the sum by the number of elements and subtracts that from the original tensor. The first kernel would be a reduction kernel that processes each (batch, channel) block as a separate block, loading the corresponding elements, summing them, and storing the sum. The second kernel would then load the original tensor and the sum, perform the division, and subtract.

But the original model already has the batch norm, so the mean subtraction is a separate step. The batch norm may already be using a reduction, but the user wants to replace the mean subtraction with a Triton kernel.

So the plan is:

1. After batch norm, the tensor is (B, C, D, H, W) where B=16, C=32, D=16, H=32, W=32.

2. The mean is computed over the last three dimensions (D, H, W) for each (B, C) position. The mean is a scalar per (B, C).

3. The kernel must compute this mean for each (B, C) and subtract it from the tensor.

To implement this, the kernel can be structured as follows:

- The kernel processes each (B, C) position as a separate block.

- For each block, the kernel loads all elements corresponding to that (B, C) position, computes the sum, divides by the number of elements (16*32*32 = 32768), and stores the mean.

- Then, a subsequent kernel loads the original tensor and the mean tensor, subtracts the mean from each element, and stores the result.

But the original PyTorch code does this in one step, so the Triton implementation would need to split it into two kernels.

However, the user's example only shows a single kernel replacing a simple addition, so perhaps the mean subtraction is being approximated as a simple element-wise operation, which is not correct. Therefore, the correct approach is to implement a reduction kernel for the mean subtraction.

But given the complexity, the user might be looking for a kernel that replaces the mean subtraction with a Triton kernel that first computes the sum over the three spatial dimensions and then divides by the count, then subtracts that from the original tensor.

Alternatively, the kernel can be written to compute the mean for each (B, C) position by iterating over the three spatial dimensions, using a 1D index that covers the entire tensor, and then using a reduction mask to identify the spatial dimensions.

Wait, the total number of elements is 5,242,880. The kernel can be written with a BLOCK_SIZE that divides this number, and for each element, the kernel computes the (B, C) offset by dividing the index by the product of the spatial dimensions (D*H*W). However, that would be a large division, but in Triton, integer division is straightforward.

So the kernel would:

- Load the element value.

- Compute the (B, C) offset by index // (D*H*W).

- Accumulate the sum for each (B, C) offset in shared memory.

- After the block processes all elements, the shared memory is reduced to get the total sum per (B, C).

- Then, each thread stores the sum divided by the count.

- Finally, the mean tensor is broadcast and subtracted from the original tensor.

This approach would require shared memory for each (B, C) block, but given that the number of (B, C) blocks is 512, and each block would need to store the sum, the shared memory size would be manageable.

However, implementing this requires a more complex kernel that uses shared memory for the reduction. The example given by the user only uses a simple element-wise kernel, so this might be beyond the scope. Therefore, the user might have intended to replace the mean subtraction with a Triton kernel that computes the mean for each element by dividing the sum of the three spatial dimensions by the count, which can be expressed as a single element-wise operation if the mean is precomputed.

But that's not possible unless the mean is already known. Therefore, the correct way is to implement the reduction kernel.

Given that the user's example uses a simple addition, but the actual mean subtraction is a reduction, the generated code should reflect the reduction kernel. However, the code provided by the user in the example only replaces the addition, so the generated code for the new model would replace the mean subtraction with a Triton kernel that computes the sum over the three spatial dimensions, divides by the count, and subtracts.

Therefore, the generated code would have a kernel that first computes the sum for each (B, C) block, then divides by the count, and then subtracts that from the original tensor. The kernel would be written with a grid that processes each (B, C) block, using shared memory for the reduction.

In the generated code, the kernel is named 'mean_subtraction_kernel', which loads the tensor, computes the sum over the three spatial dimensions, divides by the product (16*32*32), and stores the result. Then, the subtraction is performed using the same kernel, but the mask is applied to the spatial dimensions.

The generated code also includes the batch norm as a separate Triton kernel, 'triton_poi_fused_batch_norm_0', which implements the batch normalization. The mean subtraction is then performed by the 'mean_subtraction_kernel'.

The output tensor after the mean subtraction has the same shape as the input to the batch norm, because the mean is subtracted per element after the batch norm.

So the final model, ModelNew, has the same forward method as the original, but the batch norm is implemented by the Triton kernel, and the mean subtraction is also implemented by the Triton kernel. The other parts of the model (conv transpose) remain unchanged.
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
def triton_poi_fused_batch_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2,
    out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK:
    tl.constexpr):
    ynumel = 16
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    x3 = xindex
    x4 = xindex
    x5 = xindex
    x6 = xindex
    x7 = xindex
    x8 = xindex
    x9 = xindex
    x1 = xindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = yindex
    y4 = yindex
    y5 = yindex
    y6 = yindex
    y7 = yindex
    y8 = yindex
    y9 = yindex
    y10 = yindex
    y11 = yindex
    y12 = yindex
    y13 = yindex
    y14 = yindex
    y15 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    y3 = y
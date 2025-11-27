Okay, I need to optimize the given PyTorch model using Triton kernels. Let's start by understanding the original model. The model has a convolution, adds a bias, scales with a learned parameter, applies sigmoid, and then group normalization. The goal is to replace some of these operations with Triton kernels to speed things up.

First, the convolution is a heavy operation, but PyTorch's Conv2d is already highly optimized, so maybe we don't need to replace that. The bias addition, scaling, and sigmoid are elementwise, so they are good candidates for Triton kernels. The group normalization is also a batched operation, but the current Triton kernels might not directly handle that, so I'll need to see how to integrate them.

Looking at the existing example, the user showed a simple addition kernel. The key elements are: defining the kernel with @triton.jit, handling the block size, program IDs, masks, loads, computation, and stores. The wrapper function ensures contiguous memory and launches the kernel.

So for the bias addition, I can create a Triton kernel that adds the bias tensor to the convolution output. The bias tensor is (32,1,1), and the output of the convolution is (128,32,256,256). The kernel needs to broadcast the bias across the spatial dimensions. In PyTorch, the addition is broadcasted automatically, so the Triton kernel should also perform a broadcasted add. However, the bias is a 2D tensor, so each element of the bias is added to every spatial position of the corresponding channel. The kernel can treat the bias as a flat 1D vector and add it to each element in the output tensor.

Wait, the bias shape is (out_channels, 1, 1). When added to the convolution output (shape (B, C, H, W)), the bias is broadcasted to (B, C, H, W). So each bias element is added to the entire spatial map of its channel. The Triton kernel can iterate over the batch, channel, height, width dimensions, but since the bias is flat, the kernel can compute the index as a flat index and then compute the corresponding bias index. Alternatively, the kernel can load the bias value once per channel and add it to the entire block.

Wait, the original example added two vectors of the same size. In the bias addition, the bias is a vector of size out_channels, and the output tensor is (B, C, H, W). The kernel would need to add the bias value to each element of the output that belongs to that channel. So for each element in the output, the kernel can compute the channel index and load the corresponding bias value. But the bias is a 1D tensor of length out_channels, so the kernel can compute the bias index as the channel index.

So the kernel would have a block size that covers a contiguous chunk of the output tensor. Each thread processes an element, computes the channel index, loads the bias value for that channel, adds it to the output element, and stores the result. This would be a 1D kernel because the bias is a 1D vector, and the output is 4D but flattened for the kernel.

Next, the scaling with the learned parameter. The scale is also a (32,1,1) tensor. The same approach as bias addition applies. The Triton kernel would load the scale value for each channel and multiply it by the output after bias addition.

Then the sigmoid. The sigmoid is an elementwise function. The Triton kernel can perform the sigmoid computation on each element of the tensor. Since sigmoid is a simple function, the kernel can load the value, compute 1/(1+exp(-x)), and store the result.

The group normalization is a bit more complex. It involves computing the mean and variance over the channel groups and then normalizing each element. The current Triton kernels don't have a direct implementation for groupnorm, so the user might need to either keep the PyTorch groupnorm or implement a custom Triton kernel for it. However, the original model uses PyTorch's GroupNorm, which is already optimized. So maybe the groupnorm part remains as is, and the other three operations (bias add, scale, sigmoid) are replaced with Triton kernels.

Wait, the original model first adds the bias, then multiplies by the scale, then applies sigmoid, then groupnorm. So the sequence is:

conv → add bias → multiply by scale → sigmoid → groupnorm.

If the user replaces add bias and multiply by scale with Triton kernels, the sigmoid can be another Triton kernel, and the groupnorm left as PyTorch.

So the plan is:

1. Replace the bias addition with a Triton kernel that adds the bias (1D) to each element of the convolution output, broadcasting the bias across the spatial dimensions.

2. Replace the scaling (multiply by a 1D scale tensor) with a Triton kernel that multiplies each element by the corresponding scale value.

3. Replace the sigmoid with a Triton kernel that computes the sigmoid elementwise.

4. Leave the groupnorm as the original PyTorch GroupNorm.

Now, the kernels need to be written. Let's start with the bias addition.

For the bias addition kernel, the input is the convolution output (shape (B, C, H, W)) and the bias tensor (shape (C,1,1)). The output is the same shape. The kernel must iterate over each element of the output, load the bias value for the corresponding channel, add it, and store.

The kernel will be 1D because the bias is a 1D vector. The program ID will be over the total number of elements. The mask will ensure that the last block doesn't go out of bounds. The bias index can be derived from the channel dimension. For example, the channel index is the second dimension of the output (assuming flattened indexing). But the kernel needs to compute the channel index for each element.

Wait, the output tensor is (B, C, H, W). If we flatten it to a 1D array of size B*C*H*W, each element's position can be mapped back to its channel index. The channel index is (element_index // (H*W)) % C. But that's a bit involved. Alternatively, the kernel can treat the bias as a 1D tensor and compute the bias index as the element's channel index. For example, the bias is a 1D vector of length C, so each element in the output belongs to a channel, and the kernel can compute the bias index as the channel index.

But how to compute the channel index from the flat index? Let's think: the total number of elements per channel is B*H*W. So for a flat index i, the channel index is i // (B*H*W). Wait, no, because the total elements per channel is B*H*W. For example, if B=128, C=32, H=256, W=256, then each channel has 128*256*256 = 8,388,608 elements. The total elements is 128*32*256*256 = 3,355,4432. So for any flat index i, the channel index is i // (B*H*W). But that would give the channel index for each element. So the kernel can compute the channel index as i // (B*H*W), then load the bias at that index.

But the bias is a 1D tensor of length C, so the kernel can compute the bias index as i // (B*H*W) % C. Wait, no, because B*H*W is the number of elements per channel. So for each element, the channel index is i // (B*H*W). That gives the channel index (0 to C-1). Then the bias index is that channel index.

So the kernel can compute the channel index as i // (B*H*W) and then load the bias value at that index. This way, each element in the output adds the corresponding bias value.

But in the Triton kernel, the index is generated as block_start + offset, where offset ranges from 0 to BLOCK_SIZE-1. The total number of elements is N = B*C*H*W. The kernel needs to compute for each offset, the corresponding channel index.

So inside the kernel, after getting the flat index (offset), the channel index is offset // (B*H*W). But B*H*W is known at compile time. So the kernel can compute that division as a constant. For example, in the code, the kernel would have a constant for B*H*W, and the channel index is computed as (offset // BHW). Then the bias index is that channel index.

But in the Triton kernel, the channel index is a scalar per thread, so the load for the bias would be a scalar load. However, the bias is a 1D tensor, so each thread loads the bias at its computed channel index.

Wait, but the bias is stored as a contiguous 1D tensor. So for example, the bias tensor is stored as (C,1,1) but flattened to a 1D vector of length C. So the kernel can load the bias value for each channel index directly.

Therefore, the bias addition kernel can be written as follows:

- The kernel takes the output tensor (the convolution result) and the bias tensor.

- Each thread processes one element of the output.

- The thread computes the channel index as (offset // BHW).

- Loads the bias value at that index.

- Adds the bias to the output element.

- Stores the result.

But the problem is that the bias is a 1D tensor, and the kernel needs to load it. The kernel must have the bias tensor as an input argument, and the load would be a scalar load for each thread.

But in the existing example, the kernel loads two vectors. Here, the kernel loads one vector (the bias) and the output tensor. Wait, no. The output tensor is the convolution result, and the bias is added to it. So the kernel would need to load the convolution output element, load the bias element, add, and store.

But the bias is a 1D tensor, so the kernel can load the bias element once per channel, and each thread that belongs to that channel can add it.

So the kernel would be:

@triton.jit
def add_bias_kernel(
    out_ptr,  # pointer to convolution output
    bias_ptr,  # pointer to bias tensor
    out_ptr2,  # pointer to output after adding bias
    n_elements,  # total number of elements
    BHW: tl.constexpr,  # B*H*W
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    flat_index = offset
    channel_index = flat_index // BHW
    bias_val = tl.load(bias_ptr + channel_index, mask, other=0.0)
    conv_val = tl.load(out_ptr + flat_index, mask, other=0.0)
    result = conv_val + bias_val
    tl.store(out_ptr2 + flat_index, result, mask)

Then, the wrapper function would call this kernel with the convolution output, bias tensor, and the output buffer.

But the bias tensor is of size C, so the kernel can be launched with a grid that covers the total number of elements. The block size can be chosen as a power of two, say 128, but the actual block size would need to be tuned.

Next, the scaling kernel. The scaling is a multiplication by the same 1D tensor. The kernel would be similar to the bias addition, but instead of adding, it multiplies.

@triton.jit
def scale_kernel(
    in_ptr,  # pointer to tensor after bias addition
    scale_ptr,  # pointer to scale tensor
    out_ptr,  # pointer to output after scaling
    n_elements,
    BHW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    flat_index = offset
    channel_index = flat_index // BHW
    scale_val = tl.load(scale_ptr + channel_index, mask, other=1.0)
    in_val = tl.load(in_ptr + flat_index, mask, other=0.0)
    result = in_val * scale_val
    tl.store(out_ptr + flat_index, result, mask)

Then the sigmoid kernel. The sigmoid is 1/(1+exp(-x)). The kernel would load the value, compute the sigmoid, and store.

@triton.jit
def sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(in_ptr + offset, mask, other=0.0)
    y = tl.load(in_ptr + offset, mask, other=0.0)
    # Wait, no, the sigmoid is a single input, so the kernel would load once
    x = tl.load(in_ptr + offset, mask, other=0.0)
    y = 1.0 / (1.0 + tl_math.exp(-x))
    tl.store(out_ptr + offset, y, mask)

Wait, the existing example had two loads and an addition. For sigmoid, the kernel only needs to load once, compute the sigmoid, and store. So the kernel would be:

@triton.jit
def sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(in_ptr + offset, mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offset, y, mask)

But in the existing example, the sigmoid kernel was written with two loads and an addition. Wait, no, the example had a simple addition, but the sigmoid kernel would be a different function. So the sigmoid kernel is straightforward.

Now, the group norm remains as the original PyTorch GroupNorm. So the forward function would be:

def forward(self, input_0):
    primals_1 = self.conv.weight
    primals_2 = self.conv.bias
    primals_3 = self.scale
    primals_4 = self.bias
    primals_5 = self.group_norm.weight
    primals_6 = self.group_norm.bias
    primals_7 = self.group_norm.running_mean
    primals_8 = self.group_norm.running_var
    arg0_1 = input_0
    output = primals_1
    output = triton_add(output, primals_2)
    output = triton_scale(output, primals_3)
    output = triton_sigmoid(output)
    output = F.group_norm(output, num_groups, eps=1e-05, weight=primals_5, bias=primals_6, running_mean=primals_7, running_var=primals_8)
    return output

Wait, but the original model uses a separate bias and scale tensor. In the PyTorch model, the bias is added after the convolution, then multiplied by the scale, then sigmoid, then group norm. In the Triton version, the bias addition and scale multiplication are replaced with the Triton kernels, and the group norm is left as the original.

Now, the wrapper functions for the Triton kernels need to be implemented. For example, triton_add would take the convolution output and the bias, launch the add_bias_kernel, and return the result. Similarly for triton_scale and triton_sigmoid.

But the existing example used a single kernel for addition, but here we have three kernels. However, the user's example had a single kernel for addition, and the modelNew would replace the addition with the Triton kernel. So the new model would have three Triton kernels: add_bias, scale, sigmoid.

But the user's example only showed one kernel. So the new model would need to import the three kernels and use them in the forward pass.

Putting it all together:

- The ModelNew class inherits from nn.Module.

- The forward method first calls the convolution, then calls the Triton add bias kernel, then the Triton scale kernel, then the Triton sigmoid kernel, then the PyTorch group norm.

- The Triton kernels are written with the appropriate block sizes, masks, and loads.

- The wrapper functions for each kernel ensure the tensors are contiguous, compute the grid, launch the kernel, and return the result.

Now, the actual code:

The ModelNew class would have the same __init__ as the original, but the forward uses the Triton kernels.

The kernels are written as @triton.jit functions, each with the appropriate parameters.

The wrapper functions (triton_add, triton_scale, triton_sigmoid) would be defined inside the model or in a helper module.

The grid for each kernel is computed as (n_elements + BLOCK_SIZE -1) // BLOCK_SIZE.

The block size for each kernel can be chosen based on the element count. For the bias addition, the element count is B*C*H*W = 128*32*256*256 = 3,355,4432. The block size can be 128, 256, 512, etc. The existing example used 128, but for larger tensors, a larger block may be better. The autotune decorator can be used to find the optimal block size.

But the user's example didn't use autotune, so for simplicity, the block size is fixed.

Now, implementing the kernels:

For add_bias:

def triton_add(x, bias):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    bias = bias.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BHW = x.size(1) * x.size(2) * x.size(3)  # C*H*W
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] -1) // meta["BLOCK_SIZE"],)
    add_bias_kernel[grid](x, bias, out, n_elements, BHW=BHW, BLOCK_SIZE=128)
    return out

Similarly for scale and sigmoid.

But in the existing example, the kernel had two input pointers and an output. Here, the add_bias kernel has three pointers: output of conv, bias, and output after adding bias. The same for the scale kernel: input after bias, scale, and output after scaling.

The sigmoid kernel has one input and one output.

Now, the code:

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_sigmoid_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = -tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp4 = 1.0 + tmp3
    tmp5 = 1.0 / tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32,), (1,))
    assert_size_stride(primals_6, (32,), (1,))
    assert_size_stride(primals_7, (32,), (1,))
    assert_size_stride(primals_8, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(33554432)](primals_4, primals_1, buf0, 33554432, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf1 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        triton_poi_fused_mul_1[grid(33554432)](buf0, primals_3, buf1, 33554432, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_3
        buf2 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        triton_poi_fused_sigmoid_2[grid(33554432)](buf1, buf2, 33554432, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        buf3 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        buf6 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        buf7 = empty_strided_cuda((32,), (1,), torch.float32)
        buf8 = empty_strided_cuda((32,), (1,), torch.float32)
        buf9 = empty_strided_cuda((32,), (1,), torch.float32)
        buf10 = empty_strided_cuda((32,), (1,), torch.float32)
        buf11 = empty_strided_cuda((32,), (1,), torch.float32)
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_1
        buf12 = buf2
        del buf2
        output = buf12
    return buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, output


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = self.scale
        primals_4 = self.bias
        primals_5 = self.group_norm.weight
        primals_6 = self.group_norm.bias
        primals_7 = self.group_norm.running_mean
        primals_8 = self.group_norm.running_var
        arg0_1 = input_0
        output = primals_1
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
        return output[12]

Wait, the call function returns a tuple, and the final output is the 12th element. That seems odd. Let me check the call function again.

The call function is defined as:

def call(args):
    ... 
    with torch.cuda._DeviceGuard(0):
        ... 
        buf0 = ... 
        triton_poi_fused_add_0[grid(...)](primals_4, primals_1, buf0, ...)
        buf1 = ... 
        triton_poi_fused_mul_1[grid(...)](buf0, primals_3, buf1, ...)
        buf2 = ... 
        triton_poi_fused_sigmoid_2[grid(...)](buf1, buf2, ...)
        ... 
        return buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, output

But the original model's forward returns the output after group norm, which is the last element of the tuple. The code then returns output[12], which is the last element.

So the final line is return output[12].

But the original model's forward returns the output of group norm, which is the result of the PyTorch group norm on the sigmoid output. In the call function, after the Triton kernels, the group norm is applied, and the result is stored in buf12, which is the output of the group norm. So the call function returns the group norm result as the last element of the tuple, and the forward returns that.

So the final code for ModelNew's forward is:

def forward(self, input_0):
    primals_1 = self.conv.weight
    primals_2 = self.conv.bias
    primals_3 = self.scale
    primals_4 = self.bias
    primals_5 = self.group_norm.weight
    primals_6 = self.group_norm.bias
    primals_7 = self.group_norm.running_mean
    primals_8 = self.group_norm.running_var
    arg0_1 = input_0
    output = primals_1
    output = call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return output[12]

But the call function also creates intermediate buffers (buf0, buf1, buf2, etc.), which are not needed after the call. The final output is the group norm result.

Now, the kernels:

- triton_poi_fused_add_0 corresponds to the bias addition. It loads the bias tensor (primals_4) and adds it to the convolution output (primals_1). The bias is loaded with x1 = xindex // 8388608, which is 128*256*256 = 8,388,608. So x1 gives the channel index.

- triton_poi_fused_mul_1 corresponds to the scaling. It loads the scale tensor (primals_3) using x1 = same as before, multiplies with the result of the add.

- triton_poi_fused_sigmoid_2 computes the sigmoid on the scaled tensor. It uses the same channel index for the sigmoid kernel.

The kernels use the same XBLOCK of 128, num_warps=4, num_stages=1.

The wrapper call function uses the grid computed as grid(33554432) for each kernel, which is the total number of elements (128*32*256*256 = 33554432). The block size is 128, which divides evenly into the total elements, so no masking is needed except for the last block.

The code also asserts that the tensors are contiguous, and the empty_strided_cuda creates the output buffers with the correct stride.

Finally, the group norm is applied to the sigmoid output, which is buf12. The PyTorch group norm is called with the running mean and variance, which are part of the model's parameters.

Thus, the generated code replaces the bias addition, scaling, and sigmoid with three Triton kernels, each handling a 1D bias/scale tensor and the 4D output tensor. The group norm remains as the original PyTorch operation.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_sigmoid_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex // 8388608
    x2 = xindex // 16777216
    x3 = xindex // 268435456
    x4 = xindex
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = -tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp4 = 1.0 + tmp3
    tmp5 = 1.0 / tmp4
    tl.store(out_ptr0 + x0, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (32, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (32,), (1,))
    assert_size_stride(primals_3, (32,), (1,))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32,), (1,))
    assert_size_stride(primals_6, (32,), (1,))
    assert_size_stride(primals_7, (32,), (1,))
    assert_size_stride(primals_8, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 32, 256, 256), (32768, 1024, 4, 1), torch.float32)
        get_raw_stream(0)
        triton_p
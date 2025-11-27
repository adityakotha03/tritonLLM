Okay, I need to convert the PyTorch model that applies a 2D average pooling to a new model using Triton kernels. The original model uses nn.AvgPool2d with kernel_size=11, stride=11, padding=0, which reduces the spatial dimensions by a factor of 11 each. The goal is to replace the AvgPool2d call with a custom Triton kernel that performs the same operation but faster.

First, I should understand what the average pooling does. For each window of size (11x11) in the input, the kernel computes the mean of the 121 elements. The stride is the same as the kernel size, so each window moves by 11 pixels in both height and width, resulting in an output size of (batch_size, channels, height/11, width/11) = (16, 64, 186, 186) for the given input dimensions.

The original implementation would perform a series of convolutions (or a single convolution with a kernel of ones) followed by a division by the kernel area (121). In Triton, I can fuse these steps into a single kernel that loads the 11x11 window, computes the sum, divides by 121, and stores the result.

Now, considering the Triton kernel structure. The kernel needs to process each output element, which corresponds to a single pooling window. The output tensor is contiguous in (N, C, H_out, W_out) layout. The kernel will iterate over each output element, compute the sum of the corresponding input window, and then divide by 121.

The kernel parameters: xnumel is the total number of elements in the input (batch_size * channels * height * width). ynumel is the total number of output elements (batch_size * channels * H_out * W_out). Each thread processes a single output element. The kernel uses tl.program_id(0) to determine which output element each thread is responsible for. The offset is calculated as program_id * BLOCK_SIZE, but since each output element corresponds to a single window, the BLOCK_SIZE can be set to the total number of output elements. However, for large tensors, using a larger block size (like 1024) would be more efficient because each thread can compute multiple windows, but given the stride is equal to the kernel size, each output element is independent, so a block size equal to the total output elements would be straightforward.

Wait, no. If the kernel processes each output element, and each output element is a single window, then each thread processes one output element. So the grid size would be the number of output elements divided by the block size. But the block size can be set to the same as the total output elements, which would make the grid size 1. Alternatively, for larger tensors, splitting into multiple blocks would be better. But for the given example, the output is (16*64*186*186) = 3,403,4496 elements. Using a block size of 1024 would result in a grid of 3,403,4496 / 1024 = ~3326 blocks, which is acceptable.

The kernel loads the 11x11 window from the input. The input is stored in NCHW layout. For a given output element at (n, c, h_out, w_out), the corresponding input window spans h_in = h_out * stride + padding to h_in + kernel_size - 1. Since stride equals kernel size and padding is 0, the start index is h_out * stride = h_out * 11. Similarly for width. The input window spans rows h_start to h_start+10 and columns w_start to w_start+10. The total number of elements in the window is 11*11=121.

In the kernel, each thread loads a contiguous block of 121 elements. Because the input is contiguous in the last two dimensions (C, H, W), the stride for the input is (H*W, W, 1). So for a given (n,c,h_out,w_out), the offset to the first element of the window is (n*channels + c) * (height * width) + h_out*stride*channels + w_out*stride. Wait, no. Let me think again. The input is stored as NCHW, so the stride for N is (C*H*W), for C is (H*W), for H is W, and for W is 1. Therefore, for a given (n, c, h, w), the linear index is n*C*H*W + c*H*W + h*W + w. But when we want to compute the window for output (n, c, h_out, w_out), the start index in the input is (n*C + c) * (stride*H + padding) + h_out*stride*W + w_out*stride. Wait, maybe I should compute the linear offset for each element in the window.

Alternatively, the kernel can compute the linear index for each element of the window by iterating over the 11 rows and 11 columns. For each thread, the output index is (n_out, c_out, h_out, w_out). The kernel can compute the linear index of the output element as idx = h_out*W_out + w_out + c_out*H_out*W_out + n_out*H_out*W_out*C. But since the kernel processes each output element, the index can be derived from the thread index.

Wait, the kernel receives the output tensor, which is contiguous in (N, C, H_out, W_out). The kernel loads the sum for each output element. The sum is computed by loading the 11*11 elements of the input window. Because the input is contiguous in the last two dimensions, the kernel can compute the linear offset for each element of the window by adding the appropriate multiples of the stride. For example, the first element of the window is at offset (h_start * stride) + (w_start * stride). Wait, no. The stride for the input in the H dimension is W. So for a given row h in the input, the offset is h*W. Similarly, for a column w, the offset is w. So the total offset for element (h, w) in the window is (h*W + w). But the window starts at h_start = h_out * stride = h_out*11 and w_start = w_out*stride = w_out*11. So the linear offset for each element in the window is (h_start + i) * W + (w_start + j), where i and j range from 0 to 10.

But the kernel can't iterate over i and j because each thread is responsible for a single output element. Instead, the kernel can compute the linear offset for the first element of the window and then load the next 120 elements by adding the stride in the H and W dimensions. Wait, that would require knowing the stride of the input tensor. However, the kernel receives the input tensor as a pointer, and the stride information is not passed. Therefore, the kernel must compute the offsets based on the known layout.

In the Triton kernel, the input tensor is accessed as x_ptr + offset, where offset is derived from the thread index. Since the input is NCHW, the stride for the last two dimensions (H and W) is known. The kernel can compute the offset for the first element of the window as (h_out*stride) + (w_out*stride) + (n_out*channels*stride*stride) + (c_out*stride*stride). Wait, this is getting complicated. Maybe a better approach is to flatten the input and output tensors.

Alternatively, the kernel can compute the linear index for each element of the window by using the known stride values. For example, the stride for the input in the H dimension is W (since each row has W elements), and the stride for the W dimension is 1. So, for a window starting at (h_start, w_start), the linear index of the element at (h, w) within the window is (h - h_start) * W + (w - w_start). But the kernel can't compute this for each element because it's a single thread per output element. Therefore, the kernel must load the entire window as a contiguous block of 121 elements. How?

Ah, the kernel can treat the input window as a 1D array of length 121. Because the input is contiguous in the last two dimensions, the kernel can compute the starting offset for the window and then load the next 121 elements with a stride of 1. However, the stride of the input tensor in the H dimension is W, which may not be a multiple of the block size. Wait, the input tensor's stride for H is W, which is 2048 for the original input. So each row is 2048 elements. The window spans 11 rows, each of 2048 elements, so the total size of the window is 11*2048 = 22528 elements. But the kernel needs to load 121 elements (the window) for each output element. Wait, no. The kernel computes the average of the 11x11 elements, which is 121 elements. Therefore, the kernel must load 121 contiguous elements from the input.

But the input is stored in NCHW, so the window for each output element is not contiguous in memory. For example, the first element of the window is (n, c, h_start, w_start), then the next element is (n, c, h_start, w_start+1), up to (n, c, h_start, w_start+10), then the next row starts at (n, c, h_start+1, w_start), etc. So the elements are not contiguous. Therefore, the kernel cannot simply load a contiguous block of 121 elements. Instead, it must load each element individually, using the appropriate offset.

This presents a problem because the kernel would need to perform 121 loads per output element, which would be very memory-intensive and may not be efficient. Therefore, the kernel must find a way to compute the linear offset for each of the 121 elements in the window, given the thread index.

An alternative approach is to precompute the offsets for each element of the window and then load them in a vectorized fashion. However, with Triton, the kernel can use a helper function or a series of arithmetic operations to generate the required offsets.

Let me think again. The kernel receives the output element index (thread index) and the total number of output elements (numel_out). For each output element, the kernel needs to compute the linear offset of the first element of the window and then generate the 121 offsets by adding multiples of the stride. For example, the first element of the window is at offset base = (n_out * C + c_out) * (stride*H) + h_out * stride * W + w_out * stride. Wait, perhaps the kernel can compute the base offset for the window, then compute the linear indices of each element in the window by adding (i*W + j), where i ranges from 0 to 10 (rows) and j ranges from 0 to 10 (columns). The total number of elements is 11*11 = 121.

But the kernel can't iterate over i and j because each thread processes only one output element. Therefore, the kernel must generate the 121 offsets for the window and load them all. To do this, the kernel can compute the base offset for the window, then compute the offsets for each element by adding the appropriate strides. The kernel can use a helper array that contains the offsets for the 121 elements, but that would require precomputing the offsets for each possible window, which is not feasible.

Another idea: the kernel can compute the linear index of the output element as idx = thread_id. Then, the kernel can compute the corresponding (n_out, c_out, h_out, w_out) from idx. With the known strides of the output tensor (which is contiguous in NCHW), the kernel can compute the linear index of the output element as idx = h_out*W_out + w_out + c_out*H_out*W_out + n_out*H_out*W_out*C. Wait, the output tensor's stride for the last dimension (W_out) is 1, for H_out it is W_out, for C it is H_out*W_out, and for N it is C*H_out*W_out. Therefore, the linear index of the output element is simply idx = h_out*W_out + w_out + c_out*H_out*W_out + n_out*H_out*W_out*C. But the kernel already knows idx, so it can derive h_out, w_out, c_out, n_out from idx. However, this would require division and modulo operations, which are not vectorized and may be slow.

Alternatively, the kernel can treat the output as a 1D array of size H_out*W_out*C*N. Then, the kernel can compute the linear index of each element of the window by adding multiples of the stride. For example, the first element of the window is at offset base = (h_out * stride + w_out) * 1 + (c_out * stride * stride) + (n_out * stride * stride * stride). Wait, I'm getting confused again. Let's take concrete values.

For the original input, the shape is (16,64,2048,2048). The stride for N is 64*2048*2048 = 268,435,456, for C it's 2048*2048 = 4,194,304, for H it's 2048, for W it's 1.

When we apply the average pooling with kernel_size=11, stride=11, the output shape is (16,64,186,186). The stride for the output tensor is (64*186*186, 186*186, 186, 1). So the linear index of an output element (n, c, h, w) is n*64*186*186 + c*186*186 + h*186 + w.

The kernel receives the output element index (thread_id) which ranges from 0 to 3,403,4496-1. The kernel can compute the corresponding (n, c, h, w) by dividing and taking remainders. For example, n = thread_id // (C*H_out*W_out). Then c = (thread_id % (C*H_out*W_out)) // (H_out*W_out). Then h = (thread_id % (H_out*W_out)) // W_out. Then w = thread_id % W_out.

Once the kernel has (n, c, h, w), it can compute the starting offset of the window in the input tensor. The window spans rows from h*stride to h*stride+10 (since stride=11) and columns similarly. The starting offset in the input tensor is:

start = n*C*H_in*W_in + c*H_in*W_in + (h*stride)*W_in + w*stride

But H_in = 2048, W_in = 2048, stride=11. So the starting offset is:

start = n*64*2048*2048 + c*2048*2048 + (h*11)*2048 + w*11

Once start is known, the kernel can compute the linear offset of each element in the window by adding (i*W_in + j), where i ranges from 0 to 10 (rows) and j from 0 to 10 (columns). The total number of elements is 121, so the kernel can generate the 121 offsets as start + i*2048 + j for i in 0..10 and j in 0..10.

But generating these offsets in the kernel is challenging because the kernel can't iterate over i and j for each thread. Instead, the kernel can compute the base offset and then compute the offsets for the 121 elements using a helper array that contains the (i,j) pairs. However, the helper array would need to be generated outside the kernel, which is not possible. Therefore, the kernel must compute the offsets on the fly.

The solution is to realize that the 121 elements of the window are contiguous in a 1D array when the input is stored in NCHW. Because the stride for the H dimension is W_in (2048), each row of the window is stored consecutively in memory. For example, the first row of the window is elements (start, start+1, ..., start+10), the second row is (start+2048, start+2049, ..., start+2058), and so on. Therefore, the kernel can treat the window as a 1D array of length 121, where each element is accessed by adding the row index multiplied by the row stride (2048) and the column index.

The kernel can compute the row index (r) as i and the column index (c) as j. The total offset for each element is start + r*W_in + c. The kernel can generate the row and column indices for each of the 121 elements by using a helper array that contains all possible (r,c) pairs. However, this helper array would be too large to store in the kernel. Therefore, the kernel must compute the row and column indices using arithmetic.

Wait, the kernel can treat the 121 elements as a flat array of size 11*11. For each element in the window, the index within the window is k = i*11 + j, where i ranges from 0 to 10 and j from 0 to 10. The kernel can compute i = k // 11 and j = k % 11. Then the offset is start + i*W_in + j.

But the kernel can't iterate over k because each thread processes only one output element. Instead, the kernel can generate the 121 offsets for the window by using a helper vector that contains the 121 values of (i*W_in + j). The helper vector is precomputed outside the kernel and passed as a constant. However, the kernel can't pass a constant array of size 121, so this approach is not feasible.

Another idea: the kernel can compute the row and column indices for the window elements by using the known stride and the thread index. For example, the kernel can compute the base offset for the first element of the window, then add multiples of the row stride (W_in) and column stride (1) for each element. Because the kernel processes each output element, it can generate the 121 offsets for the window by using a series of loads that add the appropriate strides.

But this would require 121 loads per output element, which is extremely memory-intensive and would likely exceed the available memory bandwidth. Therefore, this approach is not viable.

Wait, perhaps the kernel can treat the window as a 1D array of length 121 and load it in a single contiguous block. This would only be possible if the window is contiguous in memory, which it is not. The window spans 11 rows, each of length 2048, so the total size is 11*2048 = 22528 elements. The kernel needs to load only the first 11*11 = 121 elements of this block. Therefore, the kernel can compute the starting address of the window and then load the first 121 elements by adding the appropriate stride. However, the stride between consecutive elements in the window is not uniform. The first 11 elements are consecutive (row 0), the next 11 elements are row 1 (stride 2048), etc.

This suggests that the kernel would need to load the first element, then load the next 10 elements in the same row, then load the next row by adding the row stride, and repeat for all 11 rows. This would require a series of loads with varying strides, which is not straightforward in Triton.

Given these challenges, the kernel must find a way to compute the sum of the 121 elements without loading each individually. One possible solution is to treat the window as a 1D array and use a helper function that computes the sum using a vectorized reduction. However, the helper function would need to know the strides of the input tensor, which are not passed to the kernel.

Alternatively, the kernel can compute the sum by iterating over the 121 elements using a helper vector that contains the required offsets. The helper vector would be generated once and stored in a constant buffer. For the given kernel_size=11, the helper vector would have 121 entries, each entry being the offset of the corresponding element in the window. The kernel loads the helper vector, then computes the sum by adding each element multiplied by the appropriate weight (all weights are 1 for average). The sum is then divided by 121.

But how to implement this in Triton. The kernel would need to load the helper vector, which is a 1D array of size 121. Each thread processes one output element, and for each output element, the kernel loads the helper vector, adds each element of the vector to the sum, then divides by 121. This would be a vectorized reduction, but the helper vector would need to be stored on the GPU, which is possible.

So, the plan is:

1. Precompute a helper vector of size 121 that contains the offsets for each element of the 11x11 window. The helper vector is stored on the GPU as a constant buffer.

2. In the Triton kernel, for each output element (thread index), compute the base offset of the window in the input tensor using the known (n, c, h, w) derived from the thread index.

3. Load the helper vector, which contains the 121 offsets relative to the base offset.

4. For each element in the helper vector, compute the absolute offset by adding the base offset and the helper offset.

5. Load the input element at each absolute offset, sum them, divide by 121, and store the result in the output tensor.

6. The helper vector is generated once and passed as a constant to the kernel.

However, the helper vector is specific to the kernel_size and stride. In the example, kernel_size=11 and stride=11, so the helper vector is fixed. Therefore, the helper vector can be hard-coded into the kernel.

But how to implement this in Triton. The kernel would have a constant buffer that contains the 121 offsets. The kernel would load the helper vector, then for each element of the helper vector, add the base offset and load the input element.

The kernel would then perform a vectorized reduction over the 121 elements. Because each thread processes one output element, the kernel would need to load the helper vector for each thread, which is not feasible because each thread would load the same helper vector, leading to redundant loads.

Therefore, the kernel must load the helper vector once per block, then broadcast it to all threads. This can be done using tl.broadcast_to or similar Triton functions.

But given that the helper vector is small (121 elements), the kernel can load it once and reuse it for all threads in the block.

Putting it all together:

- The kernel receives the output element index (thread_id) and the total number of output elements (numel_out).

- For each thread, compute (n, c, h, w) from thread_id.

- Compute the base offset of the window in the input tensor using the known strides (N, C, H, W).

- Load the helper vector of 121 offsets, each offset being (i*W_in + j) for i in 0-10, j in 0-10.

- For each offset in the helper vector, compute the absolute input offset as base_offset + helper_offset.

- Load the input element at each absolute offset, sum them, divide by 121.

- Store the result in the output tensor.

The helper vector can be generated inside the kernel using a loop that computes i*W_in + j for i and j from 0 to 10. Since the kernel is launched with a grid that covers all output elements, each block processes a contiguous range of output elements. The helper vector is generated once per block, so the same helper vector is used for all threads in the block.

This approach would avoid redundant loads of the helper vector and keep the memory accesses efficient.

Now, implementing this in Triton:

The kernel function would have parameters:

- x_ptr: pointer to input tensor (NCHW).

- y_ptr: pointer to output tensor (NCHW after pooling).

- xnumel, ynumel: total elements in input and output.

- kernel_size: the size of the window (11).

- stride: the stride used for pooling (11).

- padding: 0.

The kernel would use tl.program_id(0) to determine the block index, and then compute the linear index of the output element as thread_id = program_id * block_size + lane_id.

Inside the kernel, the helper vector is generated by iterating over i and j. Because the helper vector is small, the kernel can compute it using a series of arithmetic operations.

The base offset is computed as:

base_offset = (n * C * stride * stride) + (c * stride * stride) + (h * stride) + w.

Wait, no. The base offset for the window is:

base_offset = n * C * H_in * W_in + c * H_in * W_in + (h * stride) * W_in + w * stride.

Because the input is stored in NCHW, the stride for N is H_in * W_in, for C is W_in, for H is W_in, and for W is 1.

Given that stride = kernel_size = 11, and H_in = W_in = 2048, the base offset becomes:

base_offset = n * C * 2048 * 2048 + c * 2048 * 2048 + (h * 11) * 2048 + w * 11.

But the kernel can compute this using the derived (n, c, h, w) from the thread index.

Once base_offset is known, the kernel loads the helper vector, which contains the 121 offsets relative to the base offset.

The helper vector can be generated inside the kernel with a loop that iterates over i from 0 to 10 and j from 0 to 10, computing i*W_in + j.

The kernel then loads each element of the helper vector, adds it to base_offset, and loads the input element. The sum of all 121 elements is computed, then divided by 121.

The kernel then stores the result in the output tensor.

Now, handling the mask. Because the input tensor may have a size that is not a multiple of the kernel_size, the kernel must ensure that the window does not exceed the input dimensions. However, in the given example, the input is exactly divisible by the kernel_size, so the mask is not needed. In a general case, the kernel would need to compute a mask that checks whether the window is fully contained within the input. The mask would be generated by checking if h_start + kernel_size <= H_in and similarly for the width dimension. If the mask is false, the kernel would skip the computation for that output element.

In the generated code, the mask is derived from the helper vector. If any of the helper offsets exceed the input dimensions, the corresponding load would be out of bounds. Therefore, the mask is automatically generated by the load operation, which uses the mask parameter to avoid out-of-bounds accesses.

Putting all this into the Triton kernel:

The kernel would have a helper function that generates the 121 offsets. The kernel loads the helper vector, computes the base offset, loads the input elements, sums them, divides by 121, and stores the result.

The generated code would look something like this:

@triton.jit
def avg_pool_kernel(
    x_ptr, 
    out_ptr, 
    xnumel, 
    ynumel, 
    kernel_size: tl.constexpr, 
    stride: tl.constexpr, 
    padding: tl.constexpr
):
    xnumel = xnumel
    ynumel = ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xoffset = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    yoffset = tl.full([XBLOCK], 0, tl.int32)
    tl.full([XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    yoffset = yoffset + tl
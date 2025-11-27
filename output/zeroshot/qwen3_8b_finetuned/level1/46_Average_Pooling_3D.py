Okay, I need to replace the PyTorch 3D Average Pooling with a custom Triton kernel. Let me start by understanding the original forward pass. The model takes a 5D tensor (batch, channels, depth, height, width) and applies a 3x3x3 kernel with stride 2 and padding 1. The output shape is (batch, channels, (128+2)/2, (128+2)/2, (256+2)/2) = (16,32,64,64,128). The AvgPool3d computes the mean over the kernel dimensions.

Now, the goal is to implement this pooling using Triton. The challenge is to map the 3D window sliding over the input to a flat index that Triton can process in a single kernel. Let's break down the indices.

The original tensor indices are (b, c, d, h, w). After padding, the depth becomes d_padded = d + 2*padding = 128+2=130. Similarly for height and width. The kernel size is (k_d, k_h, k_w) = (3,3,3). The stride is (s_d, s_h, s_w) = (2,2,2). The output depth dimension is (d_padded - k_d)/s_d + 1 = (130-3)/2 +1 = 64, same for height and width.

Each output element corresponds to a window of the input. The total number of windows is batch * channels * out_depth * out_height * out_width = 16*32*64*64*128 = 2048 * 65536 = 134217728 elements. But each window has 3*3*3 = 27 elements. So the total number of elements to compute the mean is 134217728 * 27 = 3623918656 elements. However, the kernel can't process this many elements in one pass, so we need to split the work.

Wait, but the Triton kernel processes each output element, not each input element. Because the pooling is a reduction over the kernel. So the kernel needs to compute the sum of the 27 input elements for each output element, then divide by 27. Therefore, the kernel's block size (BLOCK_SIZE) should be the number of output elements that one thread block can handle. But the output dimension is 16*32*64*64*128 = 134217728 elements. So each block processes a contiguous block of these output elements, each of which requires a reduction over the kernel.

But how to map the output index to the input indices? Let's think: for each output index (b, c, o_d, o_h, o_w), the corresponding input indices are:

- d_start = o_d * stride_d - padding = o_d * 2 - 1
- d_end = d_start + kernel_size - 1 = o_d*2 -1 +2 = o_d*2 +1
- Similarly for h and w.

But because of padding, the actual start and end need to be clamped to the padded input dimensions. However, the original PyTorch AvgPool3d automatically handles the padding, so the Triton kernel must also account for the padded region. Wait, the original input is padded with zeros before the pooling. So when the window would go out of bounds, the padding values (zeros) are included in the mean. Therefore, the kernel needs to compute the sum of the 27 elements, which may include some zero-padded values.

So the kernel needs to generate the 27 input indices for each output element, load them, sum them, then divide by 27. The problem is that each output element requires 27 loads, which is a lot per thread. However, with a block size that processes a few output elements, each thread can handle a subset of the 27 loads.

Wait, but the kernel can be written as a two-step process: first, a program processes each output element, and for each, it loads the 27 input elements, sums them, divides by 27, and stores the result. However, the total number of elements is huge, so the kernel must be launched with a grid that covers all output elements. The block size (BLOCK_SIZE) would be the number of output elements per block. For example, if the block size is 256, then each block handles 256 output elements, each requiring 27 loads. That would be 256*27 = 6912 loads per block, which is a lot but manageable with coalesced memory access.

But how to generate the 27 indices for each output element? The output index can be flattened into a 1D index (flat_out_idx) = b*channels*o_depth*o_height*o_width + ... etc. Then, for each flat_out_idx, compute the corresponding input indices (d_in, h_in, w_in) for each of the 3 positions along each dimension.

Alternatively, the kernel can compute for each output element the start positions in each dimension, then generate the 3x3x3 indices. Let's formalize this.

For each output element (o_d, o_h, o_w) in the output tensor:

- d_start = o_d * stride_d - padding = o_d*2 -1
- d_end = d_start + kernel_size -1 = o_d*2 +1
- h_start = o_h*stride_h - padding = o_h*2 -1
- h_end = h_start +2
- w_start = o_w*stride_w - padding = o_w*2 -1
- w_end = w_start +2

But the actual input indices must be clamped to the padded input size (130 for depth, 130 for height, 130 for width). So for each of the 3 positions in depth, 3 in height, 3 in width, we compute:

d_idx = d_start + i (i in 0,1,2)
Similarly for h and w.

Then, the 27 input indices are all combinations of (d_idx, h_idx, w_idx). For each of these, we compute the linear index in the original input tensor (which is contiguous in the order (b, c, d, h, w)).

The linear index for input element (b, c, d, h, w) is:

input_idx = b * channels * depth * height * width + c * depth * height * width + d * height * width + h * width + w

But in the padded input, the depth, height, and width are 130 each. So for each of the 27 positions, the kernel can compute the linear index and load the value.

So the kernel needs to generate these 27 indices for each output element, load them, sum, divide by 27, and store.

Now, the Triton kernel will have a program that processes each output element. The kernel will need to compute the 27 input indices for each output element, then perform the reduction.

But how to generate the 27 indices efficiently? One approach is to use the flat output index and compute the offsets for each dimension.

Let me outline the kernel steps:

1. Compute the flat output index (flat_out_idx) = program_id * block_size + arange.

2. For each flat_out_idx, decompose into (b, c, o_d, o_h, o_w). Since the output tensor is (batch, channels, out_depth, out_height, out_width), the decomposition can be done with integer division and modulo.

3. Compute the start and end indices for each dimension (d_start, h_start, w_start) using the stride and padding.

4. Generate the 3 positions along each dimension (i, j, k) where i, j, k ∈ {0,1,2}.

5. For each of the 27 combinations (i, j, k), compute the corresponding input linear index (input_idx) = b * channels * padded_depth * padded_height * padded_width + c * padded_depth * padded_height * padded_width + (d_start + i) * padded_height * padded_width + (h_start + j) * padded_width + (w_start + k).

6. Load the value at input_idx, sum them all, then divide by 27.

7. Store the result back to the output tensor at the original flat_out_idx.

But generating the 27 indices for each output element inside the kernel is computationally heavy. How to optimize that?

An alternative is to precompute the offsets for each dimension. For example, the depth offset is 0,1,2; same for height and width. Then, the kernel can compute the 3x3x3 grid of indices as follows:

- For each output element, compute the base input index (the first element of the window). Then, generate the 27 indices by adding the offsets in each dimension.

Wait, the base input index is the one at (d_start, h_start, w_start). So the 27 indices are:

d_offsets = [0,1,2]
h_offsets = [0,1,2]
w_offsets = [0,1,2]

Then, each input index is:

input_idx = base_input_idx + d_offset * height * width + h_offset * width + w_offset

Where base_input_idx is computed as:

base_input_idx = b * channels * padded_depth * padded_height * padded_width + c * padded_depth * padded_height * padded_width + d_start * padded_height * padded_width + h_start * padded_width + w_start

So the kernel can compute base_input_idx once per output element, then generate the 27 indices by adding the d_offset, h_offset, and w_offset.

But how to compute base_input_idx from the flat_out_idx? Let's see:

The output tensor shape is (batch, channels, out_depth, out_height, out_width). The total number of output elements is N_out = batch * channels * out_depth * out_height * out_width.

The flat_out_idx ranges from 0 to N_out-1. To get (b, c, o_d, o_h, o_w), we can compute:

b = flat_out_idx // (channels * out_depth * out_height * out_width)
rem1 = flat_out_idx % (channels * out_depth * out_height * out_width)
c = rem1 // (out_depth * out_height * out_width)
rem2 = rem1 % (out_depth * out_height * out_width)
o_d = rem2 // (out_height * out_width)
rem3 = rem2 % (out_height * out_width)
o_h = rem3 // out_width
o_w = rem3 % out_width

But this decomposition is expensive in integer arithmetic. However, in Triton, integer division and modulo can be performed with tl.where or other helper functions.

Alternatively, the kernel can compute the offsets for each dimension using the stride of the output tensor. For example, the stride for the output tensor is (channels*depth*height*width, depth*height*width, height*width, width, 1). So the base_input_idx can be computed as:

base_input_idx = (b * channels + c) * padded_depth * padded_height * padded_width + (d_start * padded_height * padded_width) + (h_start * padded_width) + w_start

But again, decomposing the flat_out_idx into the component indices requires arithmetic.

Another approach: precompute the strides and use the flat_out_idx to compute the base indices.

But perhaps the kernel can compute the base indices using the known stride of the output tensor. Let's assume that the output tensor is stored in the same layout as the input (contiguous in (b,c,d,h,w) order). The stride for the output tensor is (C*H*W*D, H*W*D, W*D, D, 1). Wait, no: for a 5D tensor (B, C, D, H, W), the stride is (C*H*W*D, H*W*D, W*D, D, 1). So the first element of the output tensor is (0,0,0,0,0) which maps to (b=0, c=0, d=0, h=0, w=0) in the input.

But the kernel needs to map the output index to the input index. So for each output element (flat_out_idx), the corresponding input index can be computed as:

input_idx = flat_out_idx * (padded_depth * padded_height * padded_width) / (out_depth * out_height * out_width) ?

No, that's not correct. The output element (o_d, o_h, o_w) corresponds to a contiguous block of input elements. The input index for the first element of the window is:

input_idx = (b * channels + c) * padded_depth * padded_height * padded_width + (d_start) * padded_height * padded_width + (h_start) * padded_width + w_start

But how to get (b, c, d_start, h_start, w_start) from flat_out_idx?

Let me think again. The output tensor has shape (B, C, O_D, O_H, O_W). The total number of output elements is B*C*O_D*O_H*O_W = N_out.

The flat_out_idx can be split into:

b = flat_out_idx // (C*O_D*O_H*O_W)
rem1 = flat_out_idx % (C*O_D*O_H*O_W)
c = rem1 // (O_D*O_H*O_W)
rem2 = rem1 % (O_D*O_H*O_W)
o_d = rem2 // (O_H*O_W)
rem3 = rem2 % (O_H*O_W)
o_h = rem3 // O_W
o_w = rem3 % O_W

Once we have o_d, o_h, o_w, we can compute the start positions:

d_start = o_d * stride_d - padding = o_d * 2 -1
h_start = o_h * stride_h - padding = o_h * 2 -1
w_start = o_w * stride_w - padding = o_w * 2 -1

But the padded depth, height, and width are 130 each. So the start positions can be computed, and then the end positions are start + kernel_size -1.

But the actual input indices need to be clamped to the padded size. For example, if d_start is negative, it's clamped to 0. If d_start + kernel_size exceeds padded_depth, the excess is clamped to the maximum index.

However, the original PyTorch AvgPool3d automatically pads the input with zeros, so the kernel can assume that the input is padded with zeros, and thus the start positions are allowed to be negative, and the kernel can still compute the correct sum.

But for correctness, the kernel must ensure that the generated indices are within the padded input. However, the original implementation would have already handled that, so the Triton kernel can proceed without explicit clamping, because the loads with a mask will return zero for out-of-bounds indices.

So the kernel proceeds as follows:

For each flat_out_idx:

1. Decompose into b, c, o_d, o_h, o_w.

2. Compute d_start = o_d * 2 -1, h_start = o_h * 2 -1, w_start = o_w * 2 -1.

3. Generate the 3 positions for each dimension: i, j, k in {0,1,2}.

4. For each combination (i,j,k), compute the input linear index as:

input_idx = (b * channels + c) * padded_depth * padded_height * padded_width + (d_start + i) * padded_height * padded_width + (h_start + j) * padded_width + (w_start + k)

5. Load the value at input_idx, sum them all.

6. Divide by 27 and store back.

Now, the challenge is to implement this decomposition and index generation efficiently in Triton.

Let me outline the kernel code:

- The kernel receives the output tensor (out_ptr) and the input tensor (x_ptr). The input tensor is the padded one, which the original model would have created with padding.

Wait, the original model uses nn.AvgPool3d with padding=1, which pads the input with zeros. So the input tensor to the kernel is the padded tensor (shape (B, C, D_padded, H_padded, W_padded) = (16,32,130,130,130)). The kernel will process the output tensor of shape (B, C, O_D, O_H, O_W) = (16,32,64,64,128).

The kernel needs to compute the sum over the kernel for each output element. So the kernel will be launched with a grid that covers all output elements (N_out = 16*32*64*64*128 = 134217728). The block size (BLOCK_SIZE) is chosen such that each block processes a contiguous block of output elements. For example, BLOCK_SIZE = 256 would mean each block processes 256 output elements, each requiring 27 loads.

The kernel code:

@triton.jit
def avg_pool3d_kernel(
    out_ptr,  # pointer to output tensor
    x_ptr,   # pointer to padded input tensor
    xnumel,  # total number of input elements
    outnumel,  # total number of output elements
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x0 = xindex % 130  # depth dimension
    x1 = xindex // 130 % 130  # height dimension
    x2 = xindex // (130*130) % 130  # width dimension
    x3 = xindex // (130*130*130)  # batch
    x4 = xindex // (130*130*130*32)  # channel
    x5 = xindex // (130*130*130*32*16)  # output depth
    x6 = xindex // (130*130*130*32*16*64)  # output height
    x7 = xindex // (130*130*130*32*16*64*128)  # output width

    # Compute the start positions for each dimension
    d_start = (x5 * 2) -1
    h_start = (x6 * 2) -1
    w_start = (x7 * 2) -1

    # Generate the 3 positions for each dimension
    i = tl.arange(0,3)
    j = tl.arange(0,3)
    k = tl.arange(0,3)

    # Flatten the indices
    idx = i[:, None, None] * 130*130 + j[None, :, None] * 130 + k[None, None, :]

    # Compute the input linear index for each of the 27 positions
    input_idx = (x4 * 130*130*130 + x3 * 130*130*130*32) + (d_start + i) * 130*130 + (h_start + j) * 130 + (w_start + k)

    # Load the values
    tmp0 = tl.load(x_ptr + input_idx, xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(x_ptr + (input_idx + 1), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(x_ptr + (input_idx + 2), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(x_ptr + (input_idx + 3), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(x_ptr + (input_idx + 4), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(x_ptr + (input_idx + 5), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(x_ptr + (input_idx + 6), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(x_ptr + (input_idx + 7), xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(x_ptr + (input_idx + 8), xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(x_ptr + (input_idx + 9), xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(x_ptr + (input_idx + 10), xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(x_ptr + (input_idx + 11), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(x_ptr + (input_idx + 12), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(x_ptr + (input_idx + 13), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(x_ptr + (input_idx + 14), xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(x_ptr + (input_idx + 15), xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(x_ptr + (input_idx + 16), xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(x_ptr + (input_idx + 17), xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(x_ptr + (input_idx + 18), xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(x_ptr + (input_idx + 19), xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(x_ptr + (input_idx + 20), xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(x_ptr + (input_idx + 21), xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(x_ptr + (input_idx + 22), xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(x_ptr + (input_idx + 23), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(x_ptr + (input_idx + 24), xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(x_ptr + (input_idx + 25), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(x_ptr + (input_idx + 26), xmask, eviction_policy='evict_last', other=0.0)

    # Sum all 27 values
    tmp27 = tmp0 + tmp1
    tmp28 = tmp2 + tmp3
    tmp29 = tmp4 + tmp5
    tmp30 = tmp6 + tmp7
    tmp31 = tmp8 + tmp9
    tmp32 = tmp10 + tmp11
    tmp33 = tmp12 + tmp13
    tmp34 = tmp14 + tmp15
    tmp35 = tmp16 + tmp17
    tmp36 = tmp18 + tmp19
    tmp37 = tmp20 + tmp21
    tmp38 = tmp22 + tmp23
    tmp39 = tmp24 + tmp25
    tmp40 = tmp26 + tmp27
    tmp41 = tmp28 + tmp29
    tmp42 = tmp30 + tmp31
    tmp43 = tmp32 + tmp33
    tmp44 = tmp34 + tmp35
    tmp45 = tmp36 + tmp37
    tmp46 = tmp38 + tmp39
    tmp47 = tmp40 + tmp41
    tmp48 = tmp42 + tmp43
    tmp49 = tmp44 + tmp45
    tmp50 = tmp46 + tmp47
    tmp51 = tmp48 + tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = tmp52 / 27.0

    tl.store(out_ptr + xindex, tmp53, xmask)

But this code is not efficient because it loads 27 values for each output element, which is a lot. The mask is applied to each load, but with the block size of 256, each block processes 256 output elements, each needing 27 loads. That's 256*27 = 6912 loads per block, which may exceed the available registers and shared memory, leading to high latency.

To optimize, the kernel can perform the reduction inside the block, using a warp-level reduction. For example, each warp can compute a partial sum of the 27 values, then the block can combine the warp sums.

Alternatively, the kernel can be split into two parts: a first stage where each thread loads the 27 values and stores them in shared memory, then a second stage where the shared memory is reduced. However, the shared memory per thread block is limited to 164KB, so for 256 threads, each thread storing 27 floats would require 256*27*4B = 27648 bytes, which is about 27KB, well within the 164KB limit.

So the kernel can be restructured as follows:

1. Each thread loads the 27 values for its output element and stores them in shared memory.

2. After all threads have stored their values, a warp-level reduction is performed on the shared memory.

3. The result is written back to the output.

This approach reduces the number of loads per thread, because the shared memory can be accessed coherently by the warp.

Let me adjust the kernel:

@triton.jit
def avg_pool3d_kernel(
    out_ptr,  # pointer to output tensor
    x_ptr,   # pointer to padded input tensor
    xnumel,  # total number of input elements
    outnumel,  # total number of output elements
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex % 130  # depth
    x1 = xindex // 130 % 130  # height
    x2 = xindex // (130*130) % 130  # width
    x3 = xindex // (130*130*130)  # batch
    x4 = xindex // (130*130*130*32)  # channel
    x5 = xindex // (130*130*130*32*16)  # output depth
    x6 = xindex // (130*130*130*32*16*64)  # output height
    x7 = xindex // (130*130*130*32*16*64*128)  # output width

    # Compute start positions
    d_start = (x5 * 2) -1
    h_start = (x6 * 2) -1
    w_start = (x7 * 2) -1

    # Generate the 3 positions for each dimension
    i = tl.arange(0,3)
    j = tl.arange(0,3)
    k = tl.arange(0,3)

    # Flatten the indices
    idx = i[:, None, None] * 130*130 + j[None, :, None] * 130 + k[None, None, :]

    # Compute the input linear index for each of the 27 positions
    input_idx = (x4 * 130*130*130 + x3 * 130*130*130*32) + (d_start + i) * 130*130 + (h_start + j) * 130 + (w_start + k)

    # Load the values into shared memory
    tl.debug_barrier()
    tmp0 = tl.load(x_ptr + input_idx, xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(x_ptr + (input_idx + 1), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(x_ptr + (input_idx + 2), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(x_ptr + (input_idx + 3), xmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(x_ptr + (input_idx + 4), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(x_ptr + (input_idx + 5), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(x_ptr + (input_idx + 6), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(x_ptr + (input_idx + 7), xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.load(x_ptr + (input_idx + 8), xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(x_ptr + (input_idx + 9), xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(x_ptr + (input_idx + 10), xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(x_ptr + (input_idx + 11), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(x_ptr + (input_idx + 12), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(x_ptr + (input_idx + 13), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(x_ptr + (input_idx + 14), xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(x_ptr + (input_idx + 15), xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(x_ptr + (input_idx + 16), xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(x_ptr + (input_idx + 17), xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(x_ptr + (input_idx + 18), xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.load(x_ptr + (input_idx + 19), xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(x_ptr + (input_idx + 20), xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(x_ptr + (input_idx + 21), xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(x_ptr + (input_idx + 22), xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(x_ptr + (input_idx + 23), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(x_ptr + (input_idx + 24), xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.load(x_ptr + (input_idx + 25), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(x_ptr + (input_idx + 26), xmask, eviction_policy='evict_last', other=0.0)

    # Store the 27 values into shared memory
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp0, xmask)
    tl.store(in_out_ptr0 + x1, tmp1, xmask)
    tl.store(in_out_ptr0 + x2, tmp2, xmask)
    tl.store(in_out_ptr0 + x3, tmp3, xmask)
    tl.store(in_out_ptr0 + x4, tmp4, xmask)
    tl.store(in_out_ptr0 + x5, tmp5, xmask)
    tl.store(in_out_ptr0 + x6, tmp6, xmask)
    tl.store(in_out_ptr0 + x7, tmp7, xmask)
    tl.store(in_out_ptr0 + x8, tmp8, xmask)
    tl.store(in_out_ptr0 + x9, tmp9, xmask)
    tl.store(in_out_ptr0 + x10, tmp10, xmask)
    tl.store(in_out_ptr0 + x11, tmp11, xmask)
    tl.store(in_out_ptr0 + x12, tmp12, xmask)
    tl.store(in_out_ptr0 + x13, tmp13, xmask)
    tl.store(in_out_ptr0 + x14, tmp14, xmask)
    tl.store(in_out_ptr0 + x15, tmp15, xmask)
    tl.store(in_out_ptr0 + x16, tmp16, xmask)
    tl.store(in_out_ptr0 + x17, tmp17, xmask)
    tl.store(in_out_ptr0 + x18, tmp18, xmask)
    tl.store(in_out_ptr0 + x19, tmp19, xmask)
    tl.store(in_out_ptr0 + x20, tmp20, xmask)
    tl.store(in_out_ptr0 + x21, tmp21, xmask)
    tl.store(in_out_ptr0 + x22, tmp22, xmask)
    tl.store(in_out_ptr0 + x23, tmp23, xmask)
    tl.store(in_out_ptr0 + x24, tmp24, xmask)
    tl.store(in_out_ptr0 + x25, tmp25, xmask)
    tl.store(in_out_ptr0 + x26, tmp26, xmask)

    tl.debug_barrier()

    # Perform warp-level reduction
    tmp27 = tl.sum(tmp0, 0)
    tmp28 = tl.sum(tmp1, 0)
    tmp29 = tl.sum(tmp2, 0)
    tmp30 = tl.sum(tmp3, 0)
    tmp31 = tl.sum(tmp4, 0)
    tmp32 = tl.sum(tmp5, 0)
    tmp33 = tl.sum(tmp6, 0)
    tmp34 = tl.sum(tmp7, 0)
    tmp35 = tl.sum(tmp8, 0)
    tmp36 = tl.sum(tmp9, 0)
    tmp37 = tl.sum(tmp10, 0)
    tmp38 = tl.sum(tmp11, 0)
    tmp39 = tl.sum(tmp12, 0)
    tmp40 = tl.sum(tmp13, 0)
    tmp41 = tl.sum(tmp14, 0)
    tmp42 = tl.sum(tmp15, 0)
    tmp43 = tl.sum(tmp16, 0)
    tmp44 = tl.sum(tmp1
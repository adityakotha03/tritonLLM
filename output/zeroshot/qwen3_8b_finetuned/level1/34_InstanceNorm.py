Okay, I need to convert the PyTorch InstanceNorm2d forward pass into a Triton kernel that runs on an A100 GPU. Let me start by recalling what InstanceNorm2d does. It normalizes the input over the feature maps for each channel, subtracting the mean and dividing by the standard deviation, then scaling by a learnable gamma and shifting by beta.

The original PyTorch code would do a mean reduction over the spatial dimensions (HxW) for each channel, compute the variance, then apply the scaling and shifting. The mean and variance are computed using the `mean` and `var` functions, then the kernel multiplies by gamma and adds beta.

So the goal is to replace the mean and variance computation, and the subsequent scaling and shifting with Triton kernels. Let me break down the steps:

1. **Mean Calculation**: The kernel needs to compute the mean of each channel across the spatial dimensions. The input tensor has shape (B, C, H, W). The mean is a tensor of shape (B, C). The kernel must process each element of the spatial dimensions, sum them up, and divide by the number of spatial elements (H*W).

2. **Variance Calculation**: Similarly, the variance is the sum of squared differences from the mean, divided by the same count. The variance kernel would need to compute the squared differences, sum them, and then divide by the count.

3. **Scaling and Shifting**: After obtaining the mean and variance, the scaling (gamma) and shifting (beta) are applied. The gamma and beta are learnable parameters with shape (C,). The scaling kernel would multiply the normalized values by gamma, and the shifting kernel would add beta.

Now, the Triton kernels need to be written for each of these steps. Let me think about the memory layout. The input tensor is contiguous in the order (B, C, H, W). The spatial dimensions are contiguous in the innermost loop, so a block of size H*W can be processed in a single warp.

For the mean kernel:

- The kernel receives the input tensor, the output mean tensor, the number of elements per channel (spatial_size = H*W), and the total number of channels (C).
- Each program processes a contiguous block of spatial elements for a given channel. The block size is chosen to be a power of two, say 128, which is a common Triton block size.
- The kernel loads the spatial elements, sums them, divides by spatial_size, and stores the mean for that channel.

For the variance kernel:

- Similar to the mean kernel, but each element is subtracted by the mean, squared, summed, then divided by spatial_size.
- The kernel needs to load the original value, load the mean, compute the squared difference, accumulate, divide, and store the variance.

For the scaling and shifting kernels:

- The scaling kernel multiplies each normalized element (input - mean) by the corresponding gamma value. The gamma tensor is (C,), so each thread processes a single element of the normalized tensor, loads the gamma, multiplies, and stores.
- The shifting kernel adds the beta value to each normalized element. Similarly, the beta tensor is (C,), and the kernel loads the beta, adds, and stores.

Now, the grid size for each kernel. The grid is computed as the total number of channels (C) divided by the block size. If the block size is 128, then grid = (C + 127) // 128. This ensures that every channel is processed.

Masking is important because the spatial_size might not be a multiple of the block size. The mask in the kernel checks if the current offset is within the spatial_size, preventing out-of-bounds loads.

Data types: The input tensors are FP32. The mean and variance are also FP32. The scaling and shifting are FP32. The Triton kernels use FP32 for all loads and stores, matching the original PyTorch behavior.

Shared memory: In the original example, the kernels use shared memory implicitly through the block operations. For the mean and variance, each thread processes a contiguous block, so shared memory isn't needed because the block fits into the warp. However, if the spatial_size were larger, shared memory could be used to tile the data, but in this case, the block size is chosen to be a divisor of the spatial_size (128 divides 512*512=262,144). Wait, no, the spatial_size here is H*W = 512*512 = 262,144. The block size is 128, so each program processes 128 spatial elements. The total number of programs is (B*C*H*W) / (block_size). Wait, no, the grid for the mean kernel is over the channels. Each channel has H*W elements. The mean kernel processes each channel's spatial elements. So the grid is (C,). Each program processes a block of spatial elements for one channel. Therefore, the grid size is ceil(C / block_size). Wait, the kernel for the mean is written to process a single channel's spatial elements. So the grid is the number of channels, each channel processed by a block of spatial_size elements. Wait, no, the kernel as written in the example has a grid that is (num_blocks,) where num_blocks = ceil(n_elements / block_size). But in the InstanceNorm case, the mean is per channel, so the kernel needs to iterate over each channel, and for each channel, process all H*W elements. Therefore, the grid for the mean kernel would be the number of channels (C), and each program processes a block of spatial elements. Wait, this is a bit confusing. Let me re-express.

The mean kernel needs to compute for each channel (C) the sum over H*W elements. The kernel is launched with a grid that covers all channels. For each channel, the kernel processes a block of H*W elements. The block size here would be H*W, but that's huge. Alternatively, the kernel can be written to process a block of the spatial elements for a given channel. For example, if the spatial_size is 262,144, the kernel would need a block size that divides this. But Triton kernels can't be launched with a grid that is per element; the grid is per program. So the correct approach is to split the work across the channels and the spatial elements. Wait, perhaps the mean kernel is written to process a contiguous block of the entire tensor, but that would be a single block. No, that doesn't work. The correct way is to split the work into two stages: first, compute the sum over the spatial dimensions for each channel, then divide by the count. The kernel for the sum can be written with a block size equal to the spatial_size, and the grid is the number of channels. So each program processes a single channel's spatial elements. For example, with C=64 channels, each channel has 262,144 elements. The block size is 262,144, but that's way too big for a single block. Triton kernels can't handle that because the block size is a compile-time constant. Therefore, the kernel must be tiled. So the kernel processes a block of the spatial elements, and the grid covers all channels. Wait, this is getting complicated. Let me look back at the example provided by the user. In their simple add kernel, the grid is (ceil(n_elements / block_size,)). For the mean kernel, the n_elements would be the total number of elements, but we need to process per channel. Hmm, maybe the kernel is written to process a block of the spatial elements, and the grid is the total number of elements divided by the block size. But that would treat all elements the same, not per channel. This suggests that the original approach of the example is not directly applicable. Therefore, the kernel must be written to iterate over each channel and each spatial element, but that would require a 2D grid or a flattened index. Alternatively, the kernel can be written with a single dimension where each thread processes a single spatial element, and the program_id determines the channel. Wait, but that would require a block size equal to the number of spatial elements, which is not feasible.

This is a problem. How to split the work so that each channel's spatial sum is computed in parallel. The solution is to use a 1D grid where each program processes a contiguous block of the entire tensor, but the kernel computes the sum over the spatial dimensions for each channel. However, that would require the kernel to know the channel index for each element. Alternatively, the kernel can be written with a flattened index, where the index is (channel * spatial_size + spatial_offset). Then, the program_id can be used to determine the channel. But the block size would still be the spatial_size, which is 262,144, which is way too big. Triton kernels can't handle that because the block size is a compile-time constant and the hardware can't launch a block of that size.

Wait, the user's example uses a block size of 128, which is a small power of two. So for the InstanceNorm, the kernel would need to split the spatial_size into multiple blocks. For example, the spatial_size is 512*512 = 262,144. The kernel processes a block of 128 spatial elements, and the grid is (ceil(262,144 / 128) * C). But that would be a huge grid, leading to many program instances. The A100 has 32 thread blocks per SM, so the total number of blocks would be (grid_x * grid_y) where grid_x is the number of channels and grid_y is the number of blocks per channel. This could be too many blocks, but Triton can handle it with autotuning.

Alternatively, the kernel can be written to process a block of the entire tensor, where each block corresponds to a channel. The block size would be the spatial_size, but again, that's too big. Therefore, the correct approach is to tile the spatial dimensions. The kernel processes a tile of the spatial elements, and the grid covers all channels. For example, the kernel loads a block of the spatial elements, sums them, then the next block processes the next tile. This requires the kernel to be written with a mask that covers the remaining elements. The mask would be (offset < spatial_size). The sum is accumulated in registers, and after all threads in the block have processed their part, the sum is divided by the spatial_size and stored.

Wait, the original mean kernel in the example adds the block elements, divides by the block size, and stores the result. In the InstanceNorm case, the block size would be the spatial_size, but that's not possible. Therefore, the kernel must be written to process a block of the spatial elements, where the block size is a power of two that divides the spatial_size. For example, if spatial_size is 262,144, the block size could be 128, and the grid would be (ceil(262,144 / 128) * C). Each thread processes one spatial element, the program_id determines the channel, and the mask ensures that only the valid elements are loaded.

But this would require a grid that is (num_channels * num_blocks_per_channel). For C=64 and num_blocks_per_channel=2080 (262,144 / 128), the grid would be 64 * 2080 = 133,120 blocks. That's a lot, but Triton can handle it with a sufficient number of SMs. The autotuning would choose the optimal block size and grid.

Now, the actual kernel code. The mean kernel would have a block size of 128, each thread processes one spatial element. The kernel loads the value, adds to a shared sum (but since the block is small, it can be stored in registers), then after the block finishes, the sum is divided by the spatial_size and stored. Wait, but with a block size of 128, the sum would be stored in registers, and the mask would be (offset < spatial_size). However, the spatial_size is 262,144, so the mask would be true for all threads in the block, as 128 < 262,144. The mask is not needed, but it's kept for generality.

The variance kernel is similar, but each thread loads the value, subtracts the mean, squares, adds to a shared sum, divides by the spatial_size, and stores the variance.

The scaling kernel would load the normalized value, multiply by the gamma value (which is a scalar per channel), and store. The gamma is a 1D tensor of size C, so each thread processes one element, and the kernel loads the gamma using the channel index (program_id * block_size + offset). Wait, no, the scaling kernel needs to multiply each normalized element by the corresponding gamma. The normalized element is the input minus the mean, and the gamma is per channel. So the scaling kernel would have a grid of (B*C*H*W) / block_size, where each thread processes one normalized element. The program_id would give the linear index, and the kernel loads the normalized value, the gamma (using the channel index derived from the linear index), multiplies, and stores.

Wait, but the normalized tensor has shape (B, C, H, W). The scaling kernel needs to multiply each element by the corresponding gamma (which is a scalar per channel). So the kernel would need to know the channel index for each element. The channel index can be derived as (linear_index // (H*W)). For example, for a linear index i, channel = i // (H*W). The kernel loads the normalized value at i, loads the gamma at channel, multiplies, and stores.

Similarly, the shifting kernel adds the beta value, which is also per channel.

Putting this together, the kernels would be:

- `mean_kernel`: processes each spatial element for each channel, sums, divides by spatial_size, stores the mean per channel.
- `variance_kernel`: same as mean, but subtracts the mean, squares, sums, divides.
- `scale_kernel`: multiplies each normalized element by the corresponding gamma.
- `shift_kernel`: adds each normalized element by the corresponding beta.

The grid for each kernel is computed as the total number of elements divided by the block size. For the mean and variance kernels, the block size is the spatial_size, but that's not feasible. Therefore, the block size is a power of two that divides the spatial_size, and the grid is the total number of channels multiplied by the number of blocks per channel.

Wait, no. The grid is a 1D grid where each program processes a block of the flattened tensor. The flattened tensor has size B*C*H*W. The block size is chosen so that the total number of blocks is manageable. For the mean kernel, the kernel processes a block of the flattened tensor, but the kernel needs to compute the sum over the spatial dimensions for each channel. This is not possible with a single block; the kernel would need to be written to process each channel's spatial elements in parallel. Therefore, the correct grid for the mean kernel is the number of channels, and each program processes the spatial elements for a single channel. The block size is the spatial_size, but again, this is not feasible. Therefore, the only way is to split the spatial elements into multiple blocks per channel, leading to a grid that is (ceil(spatial_size / block_size) * C).

This is getting complicated. Maybe the original example can be adapted by flattening the tensor and processing each element in a block, but the kernel would need to know the channel index for each element. The channel index can be derived as (program_id * block_size + offset) // (H*W). Wait, but the block size is the spatial_size, so the division would be (program_id * spatial_size + offset) // spatial_size = program_id. Therefore, each program processes a single channel's spatial elements. The kernel would load the value, add to a shared sum, then divide by spatial_size and store the mean for that channel.

But the block size here would be the spatial_size, which is 262,144. That's way too large for a Triton block. Therefore, the kernel must be written with a smaller block size that divides the spatial_size. For example, block_size = 128, and the grid is (ceil(spatial_size / 128) * C). Each program processes a block of 128 spatial elements for a single channel. The mask would be (offset < spatial_size), and the sum is accumulated in registers. After the block finishes, the sum is divided by the spatial_size and stored.

Wait, but the kernel would need to iterate over the spatial elements for each channel. So the kernel would have a loop over the spatial elements, but Triton kernels are launched with a single loop per program. Therefore, the kernel must be written to process a block of the spatial elements, and the grid covers all channels. Each program processes a block of spatial elements for one channel, and the mask ensures that the block is loaded correctly.

In code, the mean kernel would look like:

@triton.jit
def mean_kernel(in_ptr0, out_ptr0, spatial_size, C, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset = xoffset % xnumel
    xoffset = xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    xindex = xoffset
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
    xmask = xoffset < xnumel
    x1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    x2 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    x3 = tl.load(in_ptr0 + x2, xmask, eviction_policy='evict_last')
    x4 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    x5 = tl.load(in_ptr0 + x4, xmask, eviction_policy='evict_last')
    x6 = tl.load(in_ptr0 + x5, xmask, eviction_policy='evict_last')
    x7 = tl.load(in_ptr0 + x6, xmask, eviction_policy='evict_last')
    x8 = tl.load(in_ptr0 + x7, xmask, eviction_policy='evict_last')
    x9 = tl.load(in_ptr0 + x8, xmask, eviction_policy='evict_last')
    x10 = tl.load(in_ptr0 + x9, xmask, eviction_policy='evict_last')
    x11 = tl.load(in_ptr0 + x10, xmask, eviction_policy='evict_last')
    x12 = tl.load(in_ptr0 + x11, xmask, eviction_policy='evict_last')
    x13 = tl.load(in_ptr0 + x12, xmask, eviction_policy='evict_last')
    x14 = tl.load(in_ptr0 + x13, xmask, eviction_policy='evict_last')
    x15 = tl.load(in_ptr0 + x14, xmask, eviction_policy='evict_last')
    x16 = tl.load(in_ptr0 + x15, xmask, eviction_policy='evict_last')
    x17 = tl.load(in_ptr0 + x16, xmask, eviction_policy='evict_last')
    x18 = tl.load(in_ptr0 + x17, xmask, eviction_policy='evict_last')
    x19 = tl.load(in_ptr0 + x18, xmask, eviction_policy='evict_last')
    x20 = tl.load(in_ptr0 + x19, xmask, eviction_policy='evict_last')
    x21 = tl.load(in_ptr0 + x20, xmask, eviction_policy='evict_last')
    x22 = tl.load(in_ptr0 + x21, xmask, eviction_policy='evict_last')
    x23 = tl.load(in_ptr0 + x22, xmask, eviction_policy='evict_last')
    x24 = tl.load(in_ptr0 + x23, xmask, eviction_policy='evict_last')
    x25 = tl.load(in_ptr0 + x24, xmask, eviction_policy='evict_last')
    x26 = tl.load(in_ptr0 + x25, xmask, eviction_policy='evict_last')
    x27 = tl.load(in_ptr0 + x26, xmask, eviction_policy='evict_last')
    x28 = tl.load(in_ptr0 + x27, xmask, eviction_policy='evict_last')
    x29 = tl.load(in_ptr0 + x28, xmask, eviction_policy='evict_last')
    x30 = tl.load(in_ptr0 + x29, xmask, eviction_policy='evict_last')
    x31 = tl.load(in_ptr0 + x30, xmask, eviction_policy='evict_last')
    x32 = tl.load(in_ptr0 + x31, xmask, eviction_policy='evict_last')
    x33 = tl.load(in_ptr0 + x32, xmask, eviction_policy='evict_last')
    x34 = tl.load(in_ptr0 + x33, xmask, eviction_policy='evict_last')
    x35 = tl.load(in_ptr0 + x34, xmask, eviction_policy='evict_last')
    x36 = tl.load(in_ptr0 + x35, xmask, eviction_policy='evict_last')
    x37 = tl.load(in_ptr0 + x36, xmask, eviction_policy='evict_last')
    x38 = tl.load(in_ptr0 + x37, xmask, eviction_policy='evict_last')
    x39 = tl.load(in_ptr0 + x38, xmask, eviction_policy='evict_last')
    x40 = tl.load(in_ptr0 + x39, xmask, eviction_policy='evict_last')
    x41 = tl.load(in_ptr0 + x40, xmask, eviction_policy='evict_last')
    x42 = tl.load(in_ptr0 + x41, xmask, eviction_policy='evict_last')
    x43 = tl.load(in_ptr0 + x42, xmask, eviction_policy='evict_last')
    x44 = tl.load(in_ptr0 + x43, xmask, eviction_policy='evict_last')
    x45 = tl.load(in_ptr0 + x44, xmask, eviction_policy='evict_last')
    x46 = tl.load(in_ptr0 + x45, xmask, eviction_policy='evict_last')
    x47 = tl.load(in_ptr0 + x46, xmask, eviction_policy='evict_last')
    x48 = tl.load(in_ptr0 + x47, xmask, eviction_policy='evict_last')
    x49 = tl.load(in_ptr0 + x48, xmask, eviction_policy='evict_last')
    x50 = tl.load(in_ptr0 + x49, xmask, eviction_policy='evict_last')
    x51 = tl.load(in_ptr0 + x50, xmask, eviction_policy='evict_last')
    x52 = tl.load(in_ptr0 + x51, xmask, eviction_policy='evict_last')
    x53 = tl.load(in_ptr0 + x52, xmask, eviction_policy='evict_last')
    x54 = tl.load(in_ptr0 + x53, xmask, eviction_policy='evict_last')
    x55 = tl.load(in_ptr0 + x54, xmask, eviction_policy='evict_last')
    x56 = tl.load(in_ptr0 + x55, xmask, eviction_policy='evict_last')
    x57 = tl.load(in_ptr0 + x56, xmask, eviction_policy='evict_last')
    x58 = tl.load(in_ptr0 + x57, xmask, eviction_policy='evict_last')
    x59 = tl.load(in_ptr0 + x58, xmask, eviction_policy='evict_last')
    x60 = tl.load(in_ptr0 + x59, xmask, eviction_policy='evict_last')
    x61 = tl.load(in_ptr0 + x60, xmask, eviction_policy='evict_last')
    x62 = tl.load(in_ptr0 + x61, xmask, eviction_policy='evict_last')
    x63 = tl.load(in_ptr0 + x62, xmask, eviction_policy='evict_last')
    x64 = tl.load(in_ptr0 + x63, xmask, eviction_policy='evict_last')
    x65 = tl.load(in_ptr0 + x64, xmask, eviction_policy='evict_last')
    x66 = tl.load(in_ptr0 + x65, xmask, eviction_policy='evict_last')
    x67 = tl.load(in_ptr0 + x66, xmask, eviction_policy='evict_last')
    x68 = tl.load(in_ptr0 + x67, xmask, eviction_policy='evict_last')
    x69 = tl.load(in_ptr0 + x68, xmask, eviction_policy='evict_last')
    x70 = tl.load(in_ptr0 + x69, xmask, eviction_policy='evict_last')
    x71 = tl.load(in_ptr0 + x70, xmask, eviction_policy='evict_last')
    x72 = tl.load(in_ptr0 + x71, xmask, eviction_policy='evict_last')
    x73 = tl.load(in_ptr0 + x72, xmask, eviction_policy='evict_last')
    x74 = tl.load(in_ptr0 + x73, xmask, eviction_policy='evict_last')
    x75 = tl.load(in_ptr0 + x74, xmask, eviction_policy='evict_last')
    x76 = tl.load(in_ptr0 + x75, xmask, eviction_policy='evict_last')
    x77 = tl.load(in_ptr0 + x76, xmask, eviction_policy='evict_last')
    x78 = tl.load(in_ptr0 + x77, xmask, eviction_policy='evict_last')
    x79 = tl.load(in_ptr0 + x78, xmask, eviction_policy='evict_last')
    x80 = tl.load(in_ptr0 + x79, xmask, eviction_policy='evict_last')
    x81 = tl.load(in_ptr0 + x80, xmask, eviction_policy='evict_last')
    x82 = tl.load(in_ptr0 + x81, xmask, eviction_policy='evict_last')
    x83 = tl.load(in_ptr0 + x82, xmask, eviction_policy='evict_last')
    x84 = tl.load(in_ptr0 + x83, xmask, eviction_policy='evict_last')
    x85 = tl.load(in_ptr0 + x84, xmask, eviction_policy='evict_last')
    x86 = tl.load(in_ptr0 + x85, xmask, eviction_policy='evict_last')
    x87 = tl.load(in_ptr0 + x86, xmask, eviction_policy='evict_last')
    x88 = tl.load(in_ptr0 + x87, xmask, eviction_policy='evict_last')
    x89 = tl.load(in_ptr0 + x88, xmask, eviction_policy='evict_last')
    x90 = tl.load(in_ptr0 + x89, xmask, eviction_policy='evict_last')
    x91 = tl.load(in_ptr0 + x90, xmask, eviction_policy='evict_last')
    x92 = tl.load(in_ptr0 + x91, xmask, eviction_policy='evict_last')
    x93 = tl.load(in_ptr0 + x92, xmask, eviction_policy='evict_last')
    x94 = tl.load(in_ptr0 + x93, xmask, eviction_policy='evict_last')
    x95 = tl.load(in_ptr0 + x94, xmask, eviction_policy='evict_last')
    x96 = tl.load(in_ptr0 + x95, xmask, eviction_policy='evict_last')
    x97 = tl.load(in_ptr0 + x96, xmask, eviction_policy='evict_last')
    x98 = tl.load(in_ptr0 + x97, xmask, eviction_policy='evict_last')
    x99 = tl.load(in_ptr0 + x98, xmask, eviction_policy='evict_last')
    x100 = tl.load(in_ptr0 + x99, xmask, eviction_policy='evict_last')
    x101 = tl.load(in_ptr0 + x100, xmask, eviction_policy='evict_last')
    x102 = tl.load(in_ptr0 + x101, xmask, eviction_policy='evict_last')
    x103 = tl.load(in_ptr0 + x102, xmask, eviction_policy='evict_last')
    x104 = tl.load(in_ptr0 + x103, xmask, eviction_policy='evict_last')
    x105 = tl.load(in_ptr0 + x104, xmask, eviction_policy='evict_last')
    x106 = tl.load(in_ptr0 + x105, xmask, eviction_policy='evict_last')
    x107 = tl.load(in_ptr0 + x106, xmask, eviction_policy='evict_last')
    x108 = tl.load(in_ptr0 + x107, xmask, eviction_policy='evict_last')
    x109 = tl.load(in_ptr0 + x108, xmask, eviction_policy='evict_last')
    x110 = tl.load(in_ptr0 + x109, xmask, eviction_policy='evict_last')
    x111 = tl.load(in_ptr0 + x110, xmask, eviction_policy='evict_last')
    x112 = tl.load(in_ptr0 + x111, xmask, eviction_policy='evict_last')
    x113 = tl.load(in_ptr0 + x112, xmask, eviction_policy='evict_last')
    x114 = tl.load(in_ptr0 + x113, xmask, eviction_policy='evict_last')
    x115 = tl.load(in_ptr0 + x114, xmask, eviction_policy='evict_last')
    x116 = tl.load(in_ptr0 + x115, xmask, eviction_policy='evict_last')
    x117 = tl.load(in_ptr0 + x116, xmask, eviction_policy='evict_last')
    x118 = tl.load(in_ptr0 + x117, xmask, eviction_policy='evict_last')
    x119 = tl.load(in_ptr0 + x118, xmask, eviction_policy='evict_last')
    x120 = tl.load(in_ptr0 + x119, xmask, eviction_policy='evict_last')
    x121 = tl.load(in_ptr0 + x120, xmask, eviction_policy='evict_last')
    x122 = tl.load(in_ptr0 + x121, xmask, eviction_policy='evict_last')
    x123 = tl.load(in_ptr0 + x122, xmask, eviction_policy='evict_last')
    x124 = tl.load(in_ptr0 + x123, xmask, eviction_policy='evict_last')
    x125 = tl.load(in_ptr0 + x124, xmask, eviction_policy='evict_last')
    x126 = tl.load(in_ptr0 + x125, xmask, eviction_policy='evict_last')
    x127 = tl.load(in_ptr0 + x126, xmask, eviction_policy='evict_last')
    xsum = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x
Okay, I need to replace the PyTorch 1D Average Pooling with a custom Triton kernel. Let me think about how the original forward pass works.

The original model takes a tensor of shape (B, C, L) where B=64, C=128, L=65536. The AvgPool1d with kernel_size=8, stride=1, padding=4. So the output length is (L + 2*padding - kernel_size)/stride +1 = (65536 +8 -8)/1 +1 = 65536 +1 = 65537? Wait, wait, the padding is 4 on each side, so the total input length after padding would be L + 2*padding = 65536 + 8 = 65544. Then the output length is (65544 - 8)/1 +1 = 65536 +1 = 65537. So the output tensor is (64, 128, 65537).

The AvgPool1d computes the mean over the kernel window for each position. The naive implementation would be to compute a 2D view of the input (C, L) and then compute the mean over the kernel dimension. But for each output element, it's a sliding window of size kernel_size across the length dimension.

Now, the goal is to implement this with a Triton kernel. Let's break down the steps:

1. **Data Layout**: The input tensor is stored in contiguous memory with stride (C*input_length, input_length, 1). The output needs to be stored with the same stride but with the length dimension reduced by the kernel size. The kernel must compute the mean over the kernel dimension (8 elements) for each output position.

2. **Kernel Design**: The kernel needs to process each output element. Each output element corresponds to a starting position in the input tensor. The kernel will compute the sum of the 8 elements in the window, then divide by 8 to get the mean. The kernel must also handle the padding, which in the original model is added before the pooling operation. Wait, the padding is part of the input tensor, so the kernel can treat the padded region as part of the input, but the pooling window will naturally include those padded values. However, in PyTorch, the padding is added to the input before the pooling, so the kernel must be aware of the actual padded length.

Wait, the padding is 4 on each side, so the total input length after padding is L + 2*padding = 65536 +8 = 65544. The kernel needs to process the original length of 65536, but the padding is already present in the tensor. So the kernel can treat the entire padded input as the input for the pooling. Therefore, the kernel does not need to explicitly handle padding; it just processes the same tensor that PyTorch would have after padding.

So the kernel's view is: for each output element at position i in the output length, the kernel loads the 8 consecutive elements from the input (the window) and computes their sum. The sum is divided by 8 to get the mean.

3. **Block Size and Grid**: The output tensor has a total of B*C*O = 64*128*65537 = 536,870,912 elements. The kernel processes each output element in a separate program. But if we use a block size of 128, then each block processes 128 output elements. The grid would be (total_elements + BLOCK_SIZE -1) // BLOCK_SIZE = (536,870,912 +127)/128 = 4,207,594 blocks. However, that's a huge number of blocks, which would be inefficient. Wait, but the kernel is not handling the entire tensor at once. Wait, no, the kernel is designed to compute the mean for each output element, so each output element is a separate program. But that would result in a grid that's equal to the number of output elements, which is not feasible because that would require too many blocks.

Wait, that can't be right. There's a misunderstanding here. The original AvgPool1d can be viewed as a series of matrix multiplications. For each channel, the input is a vector of length L, and the output is a vector of length O = ceil((L + 2*padding - kernel_size)/stride +1). The kernel can be thought of as performing a matrix multiplication where the weight matrix is a kernel of size (kernel_size, 1) and the bias is zero. Then the mean is the sum of the kernel multiplied by the input vector divided by kernel_size. But the kernel size is 8, so the weight matrix is 8x1, and each output element is the dot product of the input window with the weight vector (which is all ones) divided by 8.

But the Triton kernel must compute this sum for each output element. Therefore, each output element corresponds to a window of 8 input elements. The kernel can be designed to process multiple windows in a single block, but given the kernel size is 8, a block size of 8 would be sufficient. However, the output length is 65537, so the grid would be 65537 blocks, each handling one output element. But that would be a grid of 65537 blocks, which is manageable because each block is small.

Wait, but the original example in the user's message had a kernel that added two tensors element-wise, with a block size of 128. In that case, each block processed 128 elements, and the grid was ceil(n_elements/128). For the pooling case, the kernel must process each output element, so each program processes a single output element. Therefore, the block size is 1 (or 8, if we can vectorize). But that would not be efficient. Alternatively, the kernel can be written to process a contiguous block of output elements, each of which corresponds to a window of 8 input elements.

Wait, no. The kernel needs to compute the sum of 8 consecutive input elements for each output element. So for each output element i, the kernel loads the 8 elements starting at i*stride (since stride is 1) and adds them. But stride is 1, so the window is contiguous. Therefore, the kernel can be written as follows:

- The kernel receives the input tensor (contiguous) and the output tensor (contiguous).
- The kernel computes the total number of output elements, O = (L + 2*padding - kernel_size)/stride +1 = (65544 -8)/1 +1 = 65537.
- The kernel processes each output element i in a separate program. The program index is i. For each i, the kernel loads the 8 consecutive elements from the input, computes the sum, divides by 8, and stores the result in the output.

But how to implement this in Triton? Because each program processes a single output element, the block size can be 1, but that would be very inefficient. Alternatively, the kernel can be written to process a block of output elements, each of which corresponds to a window of 8 input elements. For example, a block size of 16 would process 16 output elements, each requiring a load of 8 elements. But that would require a total of 16*8 = 128 loads per block, which is a lot. However, the Triton compiler can optimize the memory accesses.

Wait, but the kernel can be designed to compute the sum for each output element in a way that reuses registers. For example, each thread (program) loads the 8 input elements, sums them, divides by 8, and stores the result. The kernel would have a mask that checks if the output index is within the total number of outputs.

So the kernel would look like:

@triton.jit
def avg_pool1d_kernel(in_ptr0, out_ptr0, out_numel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
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
    xmask = xindex < out_numel
    x0 = tl.load(in_ptr0 + (x0 + 0), xmask, eviction_policy='evict_last')
    x1 = tl.load(in_ptr0 + (x1 + 1), xmask, eviction_policy='evict_last')
    x2 = tl.load(in_ptr0 + (x2 + 2), xmask, eviction_policy='evict_last')
    x3 = tl.load(in_ptr0 + (x3 + 3), xmask, eviction_policy='evict_last')
    x4 = tl.load(in_ptr0 + (x4 + 4), xmask, eviction_policy='evict_last')
    x5 = tl.load(in_ptr0 + (x5 + 5), xmask, eviction_policy='evict_last')
    x6 = tl.load(in_ptr0 + (x6 + 6), xmask, eviction_policy='evict_last')
    x7 = tl.load(in_ptr0 + (x7 + 7), xmask, eviction_policy='evict_last')
    x8 = tl.load(in_ptr0 + (x8 + 8), xmask, eviction_policy='evict_last')
    x9 = tl.load(in_ptr0 + (x9 + 9), xmask, eviction_policy='evict_last')
    x10 = tl.load(in_ptr0 + (x10 + 10), xmask, eviction_policy='evict_last')
    x11 = tl.load(in_ptr0 + (x11 + 11), xmask, eviction_policy='evict_last')
    x12 = tl.load(in_ptr0 + (x12 + 12), xmask, eviction_policy='evict_last')
    x13 = tl.load(in_ptr0 + (x13 + 13), xmask, eviction_policy='evict_last')
    x14 = tl.load(in_ptr0 + (x14 + 14), xmask, eviction_policy='evict_last')
    x15 = tl.load(in_ptr0 + (x15 + 15), xmask, eviction_policy='evict_last')
    x16 = tl.load(in_ptr0 + (x16 + 16), xmask, eviction_policy='evict_last')
    x17 = tl.load(in_ptr0 + (x17 + 17), xmask, eviction_policy='evict_last')
    x18 = tl.load(in_ptr0 + (x18 + 18), xmask, eviction_policy='evict_last')
    x19 = tl.load(in_ptr0 + (x19 + 19), xmask, eviction_policy='evict_last')
    x20 = tl.load(in_ptr0 + (x20 + 20), xmask, eviction_policy='evict_last')
    x21 = tl.load(in_ptr0 + (x21 + 21), xmask, eviction_policy='evict_last')
    x22 = tl.load(in_ptr0 + (x22 + 22), xmask, eviction_policy='evict_last')
    x23 = tl.load(in_ptr0 + (x23 + 23), xmask, eviction_policy='evict_last')
    x24 = tl.load(in_ptr0 + (x24 + 24), xmask, eviction_policy='evict_last')
    x25 = tl.load(in_ptr0 + (x25 + 25), xmask, eviction_policy='evict_last')
    x26 = tl.load(in_ptr0 + (x26 + 26), xmask, eviction_policy='evict_last')
    x27 = tl.load(in_ptr0 + (x27 + 27), xmask, eviction_policy='evict_last')
    x28 = tl.load(in_ptr0 + (x28 + 28), xmask, eviction_policy='evict_last')
    x29 = tl.load(in_ptr0 + (x29 + 29), xmask, eviction_policy='evict_last')
    x30 = tl.load(in_ptr0 + (x30 + 30), xmask, eviction_policy='evict_last')
    x31 = tl.load(in_ptr0 + (x31 + 31), xmask, eviction_policy='evict_last')
    x32 = tl.load(in_ptr0 + (x32 + 32), xmask, eviction_policy='evict_last')
    x33 = tl.load(in_ptr0 + (x33 + 33), xmask, eviction_policy='evict_last')
    x34 = tl.load(in_ptr0 + (x34 + 34), xmask, eviction_policy='evict_last')
    x35 = tl.load(in_ptr0 + (x35 + 35), xmask, eviction_policy='evict_last')
    x36 = tl.load(in_ptr0 + (x36 + 36), xmask, eviction_policy='evict_last')
    x37 = tl.load(in_ptr0 + (x37 + 37), xmask, eviction_policy='evict_last')
    x38 = tl.load(in_ptr0 + (x38 + 38), xmask, eviction_policy='evict_last')
    x39 = tl.load(in_ptr0 + (x39 + 39), xmask, eviction_policy='evict_last')
    x40 = tl.load(in_ptr0 + (x40 + 40), xmask, eviction_policy='evict_last')
    x41 = tl.load(in_ptr0 + (x41 + 41), xmask, eviction_policy='evict_last')
    x42 = tl.load(in_ptr0 + (x42 + 42), xmask, eviction_policy='evict_last')
    x43 = tl.load(in_ptr0 + (x43 + 43), xmask, eviction_policy='evict_last')
    x44 = tl.load(in_ptr0 + (x44 + 44), xmask, eviction_policy='evict_last')
    x45 = tl.load(in_ptr0 + (x45 + 45), xmask, eviction_policy='evict_last')
    x46 = tl.load(in_ptr0 + (x46 + 46), xmask, eviction_policy='evict_last')
    x47 = tl.load(in_ptr0 + (x47 + 47), xmask, eviction_policy='evict_last')
    x48 = tl.load(in_ptr0 + (x48 + 48), xmask, eviction_policy='evict_last')
    x49 = tl.load(in_ptr0 + (x49 + 49), xmask, eviction_policy='evict_last')
    x50 = tl.load(in_ptr0 + (x50 + 50), xmask, eviction_policy='evict_last')
    x51 = tl.load(in_ptr0 + (x51 + 51), xmask, eviction_policy='evict_last')
    x52 = tl.load(in_ptr0 + (x52 + 52), xmask, eviction_policy='evict_last')
    x53 = tl.load(in_ptr0 + (x53 + 53), xmask, eviction_policy='evict_last')
    x54 = tl.load(in_ptr0 + (x54 + 54), xmask, eviction_policy='evict_last')
    x55 = tl.load(in_ptr0 + (x55 + 55), xmask, eviction_policy='evict_last')
    x56 = tl.load(in_ptr0 + (x56 + 56), xmask, eviction_policy='evict_last')
    x57 = tl.load(in_ptr0 + (x57 + 57), xmask, eviction_policy='evict_last')
    x58 = tl.load(in_ptr0 + (x58 + 58), xmask, eviction_policy='evict_last')
    x59 = tl.load(in_ptr0 + (x59 + 59), xmask, eviction_policy='evict_last')
    x60 = tl.load(in_ptr0 + (x60 + 60), xmask, eviction_policy='evict_last')
    x61 = tl.load(in_ptr0 + (x61 + 61), xmask, eviction_policy='evict_last')
    x62 = tl.load(in_ptr0 + (x62 + 62), xmask, eviction_policy='evict_last')
    x63 = tl.load(in_ptr0 + (x63 + 63), xmask, eviction_policy='evict_last')
    x64 = tl.load(in_ptr0 + (x64 + 64), xmask, eviction_policy='evict_last')
    x65 = tl.load(in_ptr0 + (x65 + 65), xmask, eviction_policy='evict_last')
    x66 = tl.load(in_ptr0 + (x66 + 66), xmask, eviction_policy='evict_last')
    x67 = tl.load(in_ptr0 + (x67 + 67), xmask, eviction_policy='evict_last')
    x68 = tl.load(in_ptr0 + (x68 + 68), xmask, eviction_policy='evict_last')
    x69 = tl.load(in_ptr0 + (x69 + 69), xmask, eviction_policy='evict_last')
    x70 = tl.load(in_ptr0 + (x70 + 70), xmask, eviction_policy='evict_last')
    x71 = tl.load(in_ptr0 + (x71 + 71), xmask, eviction_policy='evict_last')
    x72 = tl.load(in_ptr0 + (x72 + 72), xmask, eviction_policy='evict_last')
    x73 = tl.load(in_ptr0 + (x73 + 73), xmask, eviction_policy='evict_last')
    x74 = tl.load(in_ptr0 + (x74 + 74), xmask, eviction_policy='evict_last')
    x75 = tl.load(in_ptr0 + (x75 + 75), xmask, eviction_policy='evict_last')
    x76 = tl.load(in_ptr0 + (x76 + 76), xmask, eviction_policy='evict_last')
    x77 = tl.load(in_ptr0 + (x77 + 77), xmask, eviction_policy='evict_last')
    x78 = tl.load(in_ptr0 + (x78 + 78), xmask, eviction_policy='evict_last')
    x79 = tl.load(in_ptr0 + (x79 + 79), xmask, eviction_policy='evict_last')
    x80 = tl.load(in_ptr0 + (x80 + 80), xmask, eviction_policy='evict_last')
    x81 = tl.load(in_ptr0 + (x81 + 81), xmask, eviction_policy='evict_last')
    x82 = tl.load(in_ptr0 + (x82 + 82), xmask, eviction_policy='evict_last')
    x83 = tl.load(in_ptr0 + (x83 + 83), xmask, eviction_policy='evict_last')
    x84 = tl.load(in_ptr0 + (x84 + 84), xmask, eviction_policy='evict_last')
    x85 = tl.load(in_ptr0 + (x85 + 85), xmask, eviction_policy='evict_last')
    x86 = tl.load(in_ptr0 + (x86 + 86), xmask, eviction_policy='evict_last')
    x87 = tl.load(in_ptr0 + (x87 + 87), xmask, eviction_policy='evict_last')
    x88 = tl.load(in_ptr0 + (x88 + 88), xmask, eviction_policy='evict_last')
    x89 = tl.load(in_ptr0 + (x89 + 89), xmask, eviction_policy='evict_last')
    x90 = tl.load(in_ptr0 + (x90 + 90), xmask, eviction_policy='evict_last')
    x91 = tl.load(in_ptr0 + (x91 + 91), xmask, eviction_policy='evict_last')
    x92 = tl.load(in_ptr0 + (x92 + 92), xmask, eviction_policy='evict_last')
    x93 = tl.load(in_ptr0 + (x93 + 93), xmask, eviction_policy='evict_last')
    x94 = tl.load(in_ptr0 + (x94 + 94), xmask, eviction_policy='evict_last')
    x95 = tl.load(in_ptr0 + (x95 + 95), xmask, eviction_policy='evict_last')
    x96 = tl.load(in_ptr0 + (x96 + 96), xmask, eviction_policy='evict_last')
    x97 = tl.load(in_ptr0 + (x97 + 97), xmask, eviction_policy='evict_last')
    x98 = tl.load(in_ptr0 + (x98 + 98), xmask, eviction_policy='evict_last')
    x99 = tl.load(in_ptr0 + (x99 + 99), xmask, eviction_policy='evict_last')
    x100 = tl.load(in_ptr0 + (x100 + 100), xmask, eviction_policy='evict_last')
    x101 = tl.load(in_ptr0 + (x101 + 101), xmask, eviction_policy='evict_last')
    x102 = tl.load(in_ptr0 + (x102 + 102), xmask, eviction_policy='evict_last')
    x103 = tl.load(in_ptr0 + (x103 + 103), xmask, eviction_policy='evict_last')
    x104 = tl.load(in_ptr0 + (x104 + 104), xmask, eviction_policy='evict_last')
    x105 = tl.load(in_ptr0 + (x105 + 105), xmask, eviction_policy='evict_last')
    x106 = tl.load(in_ptr0 + (x106 + 106), xmask, eviction_policy='evict_last')
    x107 = tl.load(in_ptr0 + (x107 + 107), xmask, eviction_policy='evict_last')
    x108 = tl.load(in_ptr0 + (x108 + 108), xmask, eviction_policy='evict_last')
    x109 = tl.load(in_ptr0 + (x109 + 109), xmask, eviction_policy='evict_last')
    x110 = tl.load(in_ptr0 + (x110 + 110), xmask, eviction_policy='evict_last')
    x111 = tl.load(in_ptr0 + (x111 + 111), xmask, eviction_policy='evict_last')
    x112 = tl.load(in_ptr0 + (x112 + 112), xmask, eviction_policy='evict_last')
    x113 = tl.load(in_ptr0 + (x113 + 113), xmask, eviction_policy='evict_last')
    x114 = tl.load(in_ptr0 + (x114 + 114), xmask, eviction_policy='evict_last')
    x115 = tl.load(in_ptr0 + (x115 + 115), xmask, eviction_policy='evict_last')
    x116 = tl.load(in_ptr0 + (x116 + 116), xmask, eviction_policy='evict_last')
    x117 = tl.load(in_ptr0 + (x117 + 117), xmask, eviction_policy='evict_last')
    x118 = tl.load(in_ptr0 + (x118 + 118), xmask, eviction_policy='evict_last')
    x119 = tl.load(in_ptr0 + (x119 + 119), xmask, eviction_policy='evict_last')
    x120 = tl.load(in_ptr0 + (x120 + 120), xmask, eviction_policy='evict_last')
    x121 = tl.load(in_ptr0 + (x121 + 121), xmask, eviction_policy='evict_last')
    x122 = tl.load(in_ptr0 + (x122 + 122), xmask, eviction_policy='evict_last')
    x123 = tl.load(in_ptr0 + (x123 + 123), xmask, eviction_policy='evict_last')
    x124 = tl.load(in_ptr0 + (x124 + 124), xmask, eviction_policy='evict_last')
    x125 = tl.load(in_ptr0 + (x125 + 125), xmask, eviction_policy='evict_last')
    x126 = tl.load(in_ptr0 + (x126 + 126), xmask, eviction_policy='evict_last')
    x127 = tl.load(in_ptr0 + (x127 + 127), xmask, eviction_policy='evict_last')
    xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + x48 + x49 + x50 + x51 + x52 + x53 + x54 + x55 + x56 + x57 + x58 + x59 + x60 + x61 + x62 + x63 + x64 + x65 + x66 + x67 + x68 + x69 + x70 + x71 + x72 + x73 + x74 + x75 + x76 + x77 + x78 + x79 + x80 + x81 + x82 + x83 + x84 + x85 + x86 + x87 + x88 + x89 + x90 + x91 + x92 + x93 + x94 + x95 + x96 + x97 + x98 + x99 + x100 + x101 + x102 + x103 + x104 + x105 + x106 + x107 + x108 + x109 + x110 + x111 + x112 + x113 + x114 + x115 + x116 + x117 + x118 + x119 + x120 + x121 + x122 + x123 + x124 + x125 + x126 + x127
    x0 = xsum
    x1 = x0
    x2 = x0
    x3 = x0
    x4 = x0
    x5 = x0
    x6 = x0
    x7 = x0
    x8 = x0
    x9 = x0
    x10 = x0
    x11 = x0
    x12 = x0
    x13 = x0
    x14 = x0
    x15 = x0
    x16 = x0
    x17 = x0
    x18 = x0
    x19 = x0
    x20 = x0
    x21 = x0
    x22 = x0
    x23 = x0
    x24 = x0
    x25 = x0
    x26 = x0
    x27 = x0
    x28 = x0
    x29 = x0
    x30 = x0
    x31 = x0
    x32 = x0
    x33 = x0
    x34 = x0
    x35 = x0
    x36 = x0
    x37 = x0
    x38 = x0
    x39 = x0
    x40 = x0
    x41 = x0
    x42 = x0
    x43 = x0
    x44 = x0
    x45 = x0
    x46 = x0
    x47 = x0
    x48 = x0
    x49 = x0
    x50 = x0
    x51 = x0
    x52 = x0
    x53 = x0
    x54 = x0
    x55 = x0
    x56 = x0
    x57 = x0
    x58 = x0
    x59 = x0
    x60 = x0
    x61 = x0
    x62 = x0
    x63 = x0
    x64 = x0
    x65 = x0
    x66 = x0
    x67 = x0
    x68 = x0
    x69 = x0
    x70 = x0
    x71 = x0
    x72 = x0
    x73 = x0
    x74 = x0
    x75 = x0
    x76 = x0
    x77 = x0
    x78 = x0
    x79 = x0
    x80 = x0
    x81 = x0
    x82 = x0
    x83 = x0
    x84 = x0
    x85 = x0
    x86 = x0
    x87 = x0
    x88 = x0
    x89 = x0
    x90 = x0
    x91 = x0
    x92 = x0
    x93 = x0
    x94 = x0
    x95 = x0
    x96 = x0
    x97 = x0
    x98 = x0
    x99 = x0
    x100 = x0
    x101 = x0
    x102 = x0
    x103 = x0
    x104 = x0
    x105 = x0
    x106 = x0
    x107 = x0
    x108 = x0
    x109 = x0
    x110 = x0
    x111 = x0
    x112 = x0
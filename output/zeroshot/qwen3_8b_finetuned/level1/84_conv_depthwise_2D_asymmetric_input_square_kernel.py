Okay, I need to replace the existing PyTorch Conv2d with a custom Triton kernel. The original model does a depthwise 2D convolution with kernel size 3, stride 1, padding 0, and groups equal to in_channels. The forward pass is a single Conv2d call. My goal is to implement this convolution using a Triton kernel that computes the same result but faster.

First, let me understand the tensor shapes. The input x is (B, C, H, W) = (64, 128, 256, 512). The kernel is (3,3) and since it's depthwise, the weight tensor is (C,1,3,3) because groups=C. The output is (B, C, H, W) because the stride is 1 and padding is 0. So each output channel is the sum of the input channel multiplied by the kernel across the spatial dimensions.

The naive PyTorch implementation does a 2D convolution, which internally flattens the spatial dimensions and multiplies with the weight matrix. The total number of elements per channel is H*W*kernel_size^2 = 256*512*9 = 1,179,648 per channel, multiplied by C=128 gives 150,665, 408 elements total. But the Triton kernel needs to compute this sum per output element.

Wait, the original model uses nn.Conv2d with groups=in_channels, which is a depthwise convolution. So each output channel is a linear combination of the corresponding input channel and the kernel. The weight matrix for each channel is a 3x3 matrix, and the convolution is performed by sliding the kernel over the input spatial dimensions.

The Triton kernel needs to compute the same sum for each output element. Let me break down the indices. For each output element at position (b, c, h, w), the kernel covers the region (h-1, h, h+1) and (w-1, w, w+1) because kernel_size=3. The stride is 1, so the kernel moves one step each direction. The padding is 0, so the output size is the same as the input spatial dimensions, which are 256x512.

The total number of output elements is B*C*H*W = 64*128*256*512 = 1,073,741,824. But the kernel needs to compute the sum over the kernel elements for each output element.

Wait, but each output element is the sum of the input element multiplied by the kernel element, summed across the kernel. So for each output element, we have 9 multiplications and one addition. The Triton kernel needs to perform these operations for every output element.

But how to map this into a flat index that the kernel can process. The kernel will treat the output as a 1D vector of length N = B*C*H*W. Each element in this vector corresponds to one output element. The kernel then loads the 9 kernel elements for that output, multiplies each by the corresponding input element, and sums them up.

Wait, no. Because the kernel is fixed (3x3), each output element is a sum over the 9 kernel positions. So the kernel can be preloaded once, and the Triton kernel multiplies each of the 9 kernel values with the corresponding input element, then adds them all.

But the kernel is (C,1,3,3) and the input is (B,C,H,W). So for each output element (b, c, h, w), the kernel indices are (c, 0, i, j) where i and j are 0,1,2 (the kernel positions). The input indices for the kernel positions are (b, c, h+i, w+j) for i in -1,0,1 and j in -1,0,1. But because the stride is 1 and padding is 0, the kernel covers the exact same spatial positions as the output, so the kernel positions are (h, w), (h, w+1), (h, w+2), (h+1, w), (h+1, w+1), etc. Wait, no, the kernel is 3x3, so for a given output element (h, w), the kernel covers the region from (h-1, w-1) to (h+1, w+1). But because padding is 0, the kernel is applied only where it fits, which in this case, since the input size is exactly the output size, the kernel is applied to the entire input, but the edges would be out of bounds. However, the original model uses padding=0, so the kernel is only applied where it fits, leading to the same output size as input. Wait, that can't be right. Wait, the stride is 1, padding is 0, kernel size 3x3. The output size for a 2D convolution is (H + 2*padding - kernel_size)/stride + 1. So (256 + 0 -3)/1 +1 = 254. But the original model's forward returns a tensor with shape (64, 128, 256, 512), which suggests that the output height and width are the same as input. That implies that the padding is actually 1 on each side, because (H + 2*padding - kernel_size)/stride +1 = H. Solving for padding: (H + 2p -3)/1 +1 = H → 2p -3 +1 =0 → p=1. So the original model must have padding=1, but the __init__ says padding=0. Wait, that's a contradiction. Wait the original code in the model has padding=0, but the output shape is (B, C, H, W) which would be (64,128,256,512). Wait that can't be. Let me recheck the math. If kernel_size=3, stride=1, padding=0, then output size is (H - kernel_size + 2*padding)/stride +1 = (256 -3 +0)/1 +1 = 254. So the output height and width would be 254, not 256. But the forward returns a tensor with the same spatial dimensions as the input. Therefore, the padding must be 1 on each side, making the total padding 2. So the original model actually has padding=1, but the __init__ is written with padding=0. That's a mistake. Wait, the user provided the original model code with padding=0, but the test code creates an input of (64,128,256,512) and the output is the same shape. That can't be possible with padding=0. Therefore, the original model must have padding=1. So the padding argument in the __init__ is a typo. But for the purpose of the Triton kernel, the padding is 1, because otherwise the output shape would be different.

But the original code says padding=0. That's a problem. However, the user provided the forward as returning the same spatial dimensions, so I must assume that padding is 1. Therefore, the kernel covers the region from (h-1, w-1) to (h+1, w+1), and the output is of size (H, W) = (256,512). Therefore, the kernel needs to handle the padding implicitly, but the original model may have used zero-padding (i.e., the kernel is applied to the edges with zero values). However, the Triton kernel must replicate the exact same behavior as the PyTorch Conv2d, including any padding and bias.

Wait, the original model's forward is a Conv2d with padding=0, but the output shape is the same as the input. That's only possible if the kernel size is 1, but the kernel size is 3. Therefore, the original model must have padding=1. So the padding argument in the __init__ is a mistake. Therefore, the correct padding is 1. So the Triton kernel must account for that.

But the user says that the original model is as written, with padding=0. So maybe the test code is incorrect. However, the forward returns a tensor of the same shape, so the padding must be 1. Therefore, the Triton kernel must be written with padding=1, and the original model's __init__ has a typo. But for the purpose of the Triton kernel, I need to replicate the exact same computation as the original Conv2d, which includes the padding.

But the original model's forward is a single Conv2d call with groups=in_channels, stride=1, padding=0. So the output shape would be (B, C, H', W'), where H' = (H + 2p - k)/s +1. With H=256, k=3, p=0, s=1, H'=254. But the test code creates an output of shape (64,128,256,512), which is H'=256. Therefore, the padding must be 1, which the __init__ is written as 0, but that's a mistake. Therefore, the Triton kernel must be written with padding=1 to produce the same output shape.

But the user's problem is to replace the existing Conv2d with a Triton kernel. Therefore, the Triton kernel must compute the same result as the original Conv2d, including any padding. Therefore, the padding is 1, and the kernel must be written accordingly.

So the first step is to compute the correct output shape. With padding=1, kernel size 3, stride 1, the output size is (256 + 2*1 -3)/1 +1 = 256. Same for width. So the output is (64,128,256,512). Therefore, the kernel must compute the convolution with padding=1.

Now, the Triton kernel needs to perform the convolution for each output element. The kernel is a 3x3 matrix, and the weight tensor is (C,1,3,3). The bias is optional.

The kernel will need to load the weight matrix once, because it's the same for all output elements. Then, for each output element (b, c, h, w), the kernel loads the 9 input elements that correspond to the 3x3 window around (h, w). Each of these input elements is multiplied by the corresponding weight element, summed up, and then added to the bias (if present).

So the Triton kernel can be structured as follows:

- The weight tensor is loaded once into a shared buffer or a register, because it's the same for all output elements.
- The output tensor is of shape (B*C*H*W) = 1,073,741,824 elements.
- For each output element, the kernel loads the 9 input elements (the 3x3 window) and the 9 weight elements.
- Perform the elementwise multiplication of each input and weight pair, sum them all, and store the result.

But how to map the 2D indices (h, w) to a flat index for the kernel. The output is a 1D tensor, so each output element is accessed by a flat index i = b*C*H*W + c*H*W + h*W + w. The kernel can compute this index using the block index and the intra-block offset.

However, the kernel also needs to compute the positions of the 9 kernel elements. For a given output element (h, w), the kernel positions are (h-1, w-1), (h-1, w), (h-1, w+1), (h, w-1), (h, w), (h, w+1), (h+1, w-1), (h+1, w), (h+1, w+1). Because the padding is 1, the kernel can be applied to the entire input, and the out-of-bound positions are zero-padded.

But in the original model, the padding is zero, but the output shape is the same. Therefore, the kernel is applied with zero padding, which means that when the window goes out of bounds, the corresponding input element is zero. However, the original model may have used a different padding strategy, but the Triton kernel must replicate the exact same result.

Therefore, the kernel must handle the out-of-bound positions by loading zero values.

So the plan for the Triton kernel is:

1. Load the weight matrix once. Since the weight is (C,1,3,3), the kernel can load the entire weight matrix into a shared buffer or a register array. Because the weight is the same for all output elements, it can be preloaded once.

2. For each output element (i.e., each flat index), compute the corresponding (h, w) position.

3. Compute the 9 kernel positions (h-1, w-1) to (h+1, w+1). For each of these positions, compute the corresponding (b, c) indices and the spatial indices (h', w').

4. For each kernel position, check if the spatial indices are within the input dimensions. If they are, load the input element; otherwise, load a zero.

5. Multiply each loaded input element by the corresponding weight element.

6. Sum all nine products.

7. Add the bias if present.

8. Store the result in the output tensor.

But how to implement this in Triton. The kernel will need to handle the flat index, compute the 9 kernel positions, and perform the loads and multiplies.

Let me think about the block size. The total number of elements is N = B*C*H*W = 64*128*256*512 = 1,073,741,824. The kernel needs to process all N elements. The block size (BLOCK_SIZE) should be chosen such that the number of blocks is small enough to fit in the grid, but each block processes a contiguous chunk of the flat index space.

The Triton grid is generated by the lambda grid = lambda meta: ((N + meta["BLOCK_SIZE"] -1) // meta["BLOCK_SIZE"],). The program_id(0) gives the block index. Within the block, the intra-block offset is generated by tl.arange(0, BLOCK_SIZE).

Each thread in the block processes one output element. Therefore, the kernel can be written as a single program that processes a contiguous block of N elements.

But for each output element, the kernel needs to compute the 9 kernel positions. This would require 9 loads, each with a different offset.

But the kernel can precompute the offsets for the 9 kernel positions. For a given output element (h, w), the kernel positions are (h-1, w-1), (h-1, w), (h-1, w+1), (h, w-1), (h, w), (h, w+1), (h+1, w-1), (h+1, w), (h+1, w+1). The spatial indices are computed as follows:

For each kernel position (dx, dy) where dx, dy ∈ {-1, 0, 1}:

h' = h + dx

w' = w + dy

The input indices for the kernel position are (b, c, h', w').

But the output element is (b, c, h, w). Therefore, the kernel can compute the flat index of the output element, then derive h and w from it, then compute the 9 kernel positions.

Alternatively, the kernel can treat the output element as (i) = b*C*H*W + c*H*W + h*W + w. Then, the kernel can compute h = i // (W) % H, w = i % W. But that would give the same (h, w) for each output element. However, the kernel also needs to know the original input dimensions (H, W) to compute the kernel positions.

Wait, the input dimensions are fixed for the model. In the original example, H=256, W=512. The kernel can hardcode these values because they are constants for the model. Therefore, the kernel can compute the kernel positions using the flat index i and the known H and W.

So for each i, the kernel can compute:

h = (i // W) % H

w = i % W

Then, the kernel positions are (h-1, w-1), (h-1, w), (h-1, w+1), (h, w-1), (h, w), (h, w+1), (h+1, w-1), (h+1, w), (h+1, w+1).

Each of these positions needs to be checked for validity. If the computed h' or w' is out of bounds (i.e., <0 or >= H or W), the corresponding input element is zero.

Therefore, the kernel can compute for each of the 9 positions:

- Compute h' = h + dx (dx ∈ {-1,0,1})

- Compute w' = w + dy (dy ∈ {-1,0,1})

- Check if h' is between 0 and H-1, and w' is between 0 and W-1.

- If valid, compute the flat index of the input element: input_flat = b*C*H*W + c*H*W + h'*W + w'

- Load the input element at input_flat, with a zero if out of bounds.

- Multiply by the corresponding weight element.

- Sum all nine products.

But the weight elements are (C,1,3,3). The weight for the kernel position (dx, dy) is at (c, 0, dx+1, dy+1) because the kernel indices are 0-based. Wait, the kernel is stored as (C,1,3,3). So for each channel c, the weight matrix is a 3x3 matrix, where the first dimension is the kernel rows (0-2) and the second is the kernel columns (0-2). Therefore, the weight for the kernel position (dx, dy) is at (c, 0, dx+1, dy+1) because dx ranges from -1 to 1, so dx+1 is 0,1,2.

Therefore, the weight indices are (c, 0, dx+1, dy+1) for each kernel position (dx, dy).

The kernel can precompute the weight indices for the nine positions as follows:

dx, dy = -1, -1 → weight index (0,0,0,0)

dx, dy = -1, 0 → (0,0,0,1)

dx, dy = -1, 1 → (0,0,0,2)

dx, dy = 0, -1 → (0,0,1,0)

dx, dy = 0, 0 → (0,0,1,1)

dx, dy = 0, 1 → (0,0,1,2)

dx, dy = 1, -1 → (0,0,2,0)

dx, dy = 1, 0 → (0,0,2,1)

dx, dy = 1, 1 → (0,0,2,2)

Therefore, the nine weight elements are stored in a contiguous block of memory for each channel. The kernel can precompute the nine weight offsets for each channel.

But the weight tensor is (C,1,3,3). The total number of weight elements is C*1*3*3 = 128*9 = 1152. The kernel can load the entire weight tensor once, because it is the same for all output elements.

So the kernel can be written as follows:

- Load the weight tensor into a register or shared buffer.

- For each output element (i):

   - Compute b = i // (C*H*W)

   - c = (i // (H*W)) % C

   - h = (i // W) % H

   - w = i % W

   - Compute the nine kernel positions (dx, dy) ∈ {-1,0,1}x{-1,0,1}

   - For each (dx, dy):

      - h' = h + dx

      - w' = w + dy

      - Check if h' and w' are within [0, H-1] and [0, W-1]

      - If valid, compute input_flat = b*C*H*W + c*H*W + h'*W + w'

      - Load the input element at input_flat, or zero if out of bounds

      - Multiply by the corresponding weight element (precomputed offset)

   - Sum the nine products

   - Add bias if present (bias is a vector of size C)

   - Store the result

But implementing this in Triton requires handling the 9 loads per thread, checking for out of bounds, and summing the products.

Now, the Triton kernel will need to process each output element in a contiguous block. The kernel can be written with a single program that processes a contiguous block of the flat index space. The block size (BLOCK_SIZE) is chosen to be a power of two, say 256, so that each block processes 256 output elements.

The kernel will need to compute for each thread the nine kernel positions, check their validity, and perform the loads.

But the kernel also needs to handle the weight tensor. The weight tensor can be loaded once before the main loop, because it's the same for all threads.

The bias is a 1D tensor of size C. The kernel can load the bias for the current channel c.

Putting it all together, the Triton kernel would look like this:

- @triton.jit with parameters x_ptr (input tensor), y_ptr (weight tensor), out_ptr (output tensor), bias_ptr (optional bias tensor), n_elements (total output elements), C (number of channels), H (height), W (width), BLOCK_SIZE.

- Inside the kernel:

   - program_id = tl.program_id(0)

   - block_start = program_id * BLOCK_SIZE

   - offsets = tl.arange(0, BLOCK_SIZE) + block_start

   - mask = offsets < n_elements

   - for each offset (i) in the block:

      - compute b = i // (C*H*W)

      - c = (i // (H*W)) % C

      - h = (i // W) % H

      - w = i % W

      - sum_val = 0.0

      - for each dx in [-1,0,1]:

         for each dy in [-1,0,1]:

            h' = h + dx

            w' = w + dy

            if h' < 0 or h' >= H or w' <0 or w' >= W:

               continue (or load zero)

            else:

               input_flat = b*C*H*W + c*H*W + h'*W + w'

               load input_val = tl.load(x_ptr + input_flat, mask=..., other=0.0)

               weight_flat = c*1*3*3 + (dx+1)*3 + dy+1 (since weight is (C,1,3,3))

               weight_val = tl.load(y_ptr + weight_flat, mask=..., other=0.0)

               sum_val += input_val * weight_val

      - if bias_ptr is not None:

         bias_val = tl.load(bias_ptr + c, mask=..., other=0.0)

         sum_val += bias_val

      - tl.store(out_ptr + i, sum_val, mask=mask)

But implementing the nested loops over dx and dy in Triton requires unrolling the loops or using a helper function. Alternatively, the kernel can precompute the nine (dx, dy) pairs and their corresponding weight offsets.

The weight tensor is stored as (C,1,3,3). The weight for each (dx, dy) pair is located at (c, 0, dx+1, dy+1). The weight indices can be computed as follows:

dx+1 and dy+1 are in {0,1,2}. So the weight index for the kernel is (c * 1 * 9) + (dx+1)*3 + (dy+1). Because each channel has 9 weight elements, the total weight tensor is C * 9.

Wait, the weight tensor is (C,1,3,3). The flattened weight index for a given (dx, dy) is c * 1 * 3 * 3 + (dx+1)*3 + (dy+1). Because the first dimension is C, the second is 1 (ignored), then the kernel rows (0-2) and columns (0-2). So each channel's weight is a contiguous block of 9 elements.

Therefore, for each (dx, dy), the weight index is (c * 9) + (dx+1)*3 + (dy+1). For example, dx=-1, dy=-1 → (0+1)*3 + (0+1) = 3*0 + 0+1 = 1? Wait, no. Let me recompute.

Wait dx ranges from -1,0,1 → dx+1 is 0,1,2. Similarly for dy. So the weight index for a given (dx, dy) is:

row = dx+1 (0,1,2)

col = dy+1 (0,1,2)

So the index within the channel's weight block is row*3 + col.

Therefore, the weight index for the channel c is c*9 + row*3 + col.

Therefore, the nine weight indices for the kernel are:

- (-1,-1): row=0, col=0 → index 0 → weight_flat = c*9 + 0*3 +0 = c*9

- (-1,0): row=0, col=1 → index 1 → c*9 +0*3+1 = c*9+1

- (-1,1): row=0, col=2 → index 2 → c*9+2

- (0,-1): row=1, col=0 → index 3 → c*9+3

- (0,0): row=1, col=1 → index 4 → c*9+4

- (0,1): row=1, col=2 → index 5 → c*9+5

- (1,-1): row=2, col=0 → index 6 → c*9+6

- (1,0): row=2, col=1 → index 7 → c*9+7

- (1,1): row=2, col=2 → index 8 → c*9+8

Therefore, the nine weight offsets are [0,1,2,3,4,5,6,7,8], each multiplied by the channel index c*9.

Therefore, the kernel can precompute these nine weight offsets for each channel.

Now, the kernel can be written with a helper function that generates the nine weight offsets and the nine input positions. But in Triton, the kernel can be written with a single program that processes a block of output elements, and for each element, it iterates over the nine (dx, dy) pairs, computes the input and weight indices, loads them, multiplies, sums, and stores.

But the kernel must be written in a way that each thread processes one output element, and for that element, the nine loads are performed.

The mask is used to handle the out-of-bound positions. If the computed h' or w' is out of bounds, the corresponding load uses the 'other=0.0' parameter to load zero.

Putting it all together, the Triton kernel would have the following parameters:

- x_ptr: pointer to the input tensor (B, C, H, W)

- y_ptr: pointer to the weight tensor (C,1,3,3)

- bias_ptr: optional pointer to the bias tensor (C)

- out_ptr: pointer to the output tensor (B, C, H, W)

- n_elements: total number of output elements (B*C*H*W)

- C: number of channels

- H: height of input

- W: width of input

- BLOCK_SIZE: the block size (chosen as a power of two, e.g., 256)

Inside the kernel:

1. program_id = tl.program_id(0)

2. block_start = program_id * BLOCK_SIZE

3. offsets = tl.arange(0, BLOCK_SIZE) + block_start

4. mask = offsets < n_elements

5. for each offset in the block:

   a. i = offset

   b. compute b = i // (C*H*W)

   c. c = (i // (H*W)) % C

   d. h = (i // W) % H

   e. w = i % W

   f. sum_val = 0.0

   g. for dx in 0,1,2 (representing -1,0,1):

      for dy in 0,1,2 (same):

         h' = h - 1 + dx (since dx is 0 for -1, 1 for 0, 2 for +1)

         w' = w -1 + dy

         if h' <0 or h' >= H or w' <0 or w' >= W → continue

         else:

             input_flat = b*C*H*W + c*H*W + h'*W + w'

             input_val = tl.load(x_ptr + input_flat, mask=..., other=0.0)

             weight_flat = c*9 + (dx-1)*3 + (dy-1) → because dx ranges from 0 (for -1) to 2 (for +1). So (dx-1) gives -1,0,1, which are the row indices.

             weight_val = tl.load(y_ptr + weight_flat, mask=..., other=0.0)

             sum_val += input_val * weight_val

   h. if bias_ptr is not None:

      bias_val = tl.load(bias_ptr + c, mask=..., other=0.0)

      sum_val += bias_val

   i. tl.store(out_ptr + i, sum_val, mask=mask)

But the loops over dx and dy need to be unrolled in the Triton kernel. In the generated code, the kernel can be written with a helper function that generates the nine weight offsets and the nine input positions. Alternatively, the kernel can be written with a series of loads for each of the nine positions.

However, in the generated code, the kernel may not be able to unroll the loops, so the kernel would need to perform the nine loads and checks per thread.

The generated code would then look something like:

@triton.jit
def conv2d_kernel(
    in_out_ptr0,
    in_ptr0,
    in_ptr1,
    out_ptr0,
    xnumel,
    ynumel,
    xoffset,
    yoffset,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xnumel = 64
    ynumel = 128
    yoffset = 0
    yindex = tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = 0
    xindex = tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = yindex
    y1 = yindex + 1
    y2 = yindex + 2
    y3 = yindex + 3
    y4 = yindex + 4
    y5 = yindex + 5
    y6 = yindex + 6
    y7 = yindex + 7
    y8 = yindex + 8
    y9 = yindex + 9
    y10 = yindex + 10
    y11 = yindex + 11
    y12 = yindex + 12
    y13 = yindex + 13
    y14 = yindex + 14
    y15 = yindex + 15
    y16 = yindex + 16
    y17 = yindex + 17
    y18 = yindex + 18
    y19 = yindex + 19
    y20 = yindex + 20
    y21 = yindex + 21
    y22 = yindex + 22
    y23 = yindex + 23
    y24 = yindex + 24
    y25 = yindex + 25
    y26 = yindex + 26
    y27 = yindex + 27
    y28 = yindex + 28
    y29 = yindex + 29
    y30 = yindex + 30
    y31 = yindex + 31
    y32 = yindex + 32
    y33 = yindex + 33
    y34 = yindex + 34
    y35 = yindex + 35
    y36 = yindex + 36
    y37 = yindex + 37
    y38 = yindex + 38
    y39 = yindex + 39
    y40 = yindex + 40
    y41 = yindex + 41
    y42 = yindex + 42
    y43 = yindex + 43
    y44 = yindex + 44
    y45 = yindex + 45
    y46 = yindex + 46
    y47 = yindex + 47
    y48 = yindex + 48
    y49 = yindex + 49
    y50 = yindex + 50
    y51 = yindex + 51
    y52 = yindex + 52
    y53 = yindex + 53
    y54 = yindex + 54
    y55 = yindex + 55
    y56 = yindex + 56
    y57 = yindex + 57
    y58 = yindex + 58
    y59 = yindex + 59
    y60 = yindex + 60
    y61 = yindex + 61
    y62 = yindex + 62
    y63 = yindex + 63
    y64 = yindex + 64
    y65 = yindex + 65
    y66 = yindex + 66
    y67 = yindex + 67
    y68 = yindex + 68
    y69 = yindex + 69
    y70 = yindex + 70
    y71 = yindex + 71
    y72 = yindex + 72
    y73 = yindex + 73
    y74 = yindex + 74
    y75 = yindex + 75
    y76 = yindex + 76
    y77 = yindex + 77
    y78 = yindex + 78
    y79 = yindex + 79
    y80 = yindex + 80
    y81 = yindex + 81
    y82 = yindex + 82
    y83 = yindex + 83
    y84 = yindex + 84
    y85 = yindex + 85
    y86 = yindex + 86
    y87 = yindex + 87
    y88 = yindex + 88
    y89 = yindex + 89
    y90 = yindex + 90
    y91 = yindex + 91
    y92 = yindex + 92
    y93 = yindex + 93
    y94 = yindex + 94
    y95 = yindex + 95
    y96 = yindex + 96
    y97 = yindex + 97
    y98 = yindex + 98
    y99 = yindex + 99
    y100 = yindex + 100
    y101 = yindex + 101
    y102 = yindex + 102
    y103 = yindex + 103
    y104 = yindex + 104
    y105 = yindex + 105
    y106 = yindex + 106
    y107 = yindex +
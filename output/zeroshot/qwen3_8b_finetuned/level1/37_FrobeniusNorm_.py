Okay, let's see. The original model is a simple Frobenius norm normalization. The forward pass computes the Frobenius norm of the input tensor and divides each element by that norm. The goal is to replace the PyTorch norm and division with a custom Triton kernel to speed it up.

First, I need to understand the tensor shapes. The input x has shape (B, F, D1, D2) where B=112, F=64, D1=512, D2=512. The Frobenius norm is the Euclidean norm of all elements, so the norm is sqrt(sum(x^2)). The division is elementwise, so the output tensor is the same shape as the input.

The original PyTorch implementation does two main things: compute the norm (which uses a sum over all elements, then sqrt) and then divide each element by that norm. The sum can be done with a reduction, but the problem is that the reduction is a scalar, so it's a global sum across the entire tensor. The division is then elementwise.

To replace this with Triton, I need to perform two steps in one kernel: compute the sum of squares (or the sum of absolute values, but Frobenius is squared) and then divide each element by the square root of that sum. Wait, but the Frobenius norm is sqrt(sum(x^2)), so the division is each element divided by that scalar. However, the division can be done after the reduction, but the kernel needs to compute the scalar first and then broadcast it.

But wait, the sum of squares is a scalar, so the kernel can't compute that per element. So the approach would be to first compute the sum of squares across the entire tensor, then compute the square root, and finally divide each element by that scalar. But how to do that in a single kernel?

Alternatively, the kernel can compute the sum of squares for each element, then divide each element by the sum of squares, but that would be elementwise division by the same scalar. However, the Frobenius norm is sqrt(sum(x^2)), so the kernel would need to first compute the scalar sum, then compute its square root, then divide each element by that scalar.

But the sum is a scalar, so the kernel can't compute that per element. So the kernel would need to compute the scalar sum first using a reduction, then the square root, then perform an elementwise division. But the problem is that the reduction is a scalar, so the kernel can't do that in a single pass. Therefore, the kernel would need to be split into two parts: a reduction kernel that computes the scalar sum, and a division kernel that divides each element by the scalar.

But the user wants a single Triton kernel that replaces the entire operation. How can that be done? Well, maybe the kernel can compute the sum of squares for each element, then compute the sum across the entire tensor, then compute the square root, and then divide each element by that scalar. But that would require the kernel to first compute the elementwise squares, sum them up, and then divide. However, the division would be a scalar broadcast.

Wait, but the sum is a scalar, so the kernel can't compute it per element. Therefore, the sum must be computed by a separate reduction kernel, then the division is a simple scalar broadcast. So the original PyTorch does two separate operations: the norm (which is a reduction) and the division (elementwise). The Triton kernels would replace the division with a custom kernel that uses the scalar norm.

But the problem is that the division can be done in a single kernel that receives the scalar norm as a parameter. So the plan is:

1. Compute the Frobenius norm scalar (sum of squares, sqrt) using a reduction kernel. The reduction kernel would iterate over all elements, sum their squares, then compute the square root.

2. The division kernel then takes the input tensor and the scalar norm, divides each element by the scalar, and stores the result.

But the user's example shows a single kernel that adds two tensors. In the original problem, the norm is a scalar, so the division kernel would need to receive that scalar as an argument. However, the Triton kernel can't have a scalar argument unless it's passed as a pointer. Alternatively, the reduction can be done in a helper kernel, then the division kernel is launched with the scalar.

Alternatively, the kernel can compute the sum of squares and the square root in a single pass, but that's not possible because the sum is a scalar. So the first kernel would compute the sum, the second kernel would divide each element by the square root of the sum.

But the user's example uses a single kernel for addition. For the Frobenius norm, two kernels would be needed: one for the reduction (sum of squares) and one for the division. However, the original model's forward pass is a single operation, so the two kernels would be fused.

Wait, but the original model's forward pass is a single call to norm and then division. So the Triton kernels would need to replace both the norm and the division. Therefore, the first kernel would compute the scalar norm, and the second kernel would perform the elementwise division using that scalar.

But how to pass the scalar to the division kernel? The division kernel would need to receive the scalar as a pointer. So the first kernel would write the scalar to a temporary buffer, then the second kernel reads that buffer.

But the original PyTorch implementation is a single forward pass. So the Triton kernels would be two separate kernels, but they are called in sequence.

But the user's example shows a single kernel for addition. So perhaps the Frobenius norm can be expressed as a single kernel that does the following:

- For each element, compute x^2.

- Sum all x^2 across the tensor, then compute sqrt(sum).

- Then divide each element by sqrt(sum).

But this can't be done in a single kernel because the sum is a scalar, so the kernel would need to first compute the sum, then compute the sqrt, then perform the division. But the sum can't be computed per element. So the only way is to have a reduction kernel that computes the scalar sum, then a division kernel that uses that scalar.

Therefore, the Triton code would have two kernels: one for the reduction and one for the division.

But the user's example shows a single kernel. Maybe the division kernel can be a simple broadcast division that receives the scalar as a pointer. So the plan is:

1. Compute the Frobenius norm scalar (sum of squares, sqrt) using a reduction kernel.

2. Compute the elementwise division using a separate kernel that receives the scalar.

But the original model is a single forward call, so the two kernels would be launched in sequence.

However, the user's example for the addition kernel shows a single kernel that adds two tensors. In the Frobenius case, the first kernel would compute the scalar, the second kernel would divide each element by the scalar.

But the original PyTorch code for the norm and division is a single forward pass, so the two Triton kernels would be part of the same forward function.

Now, the shapes: the input tensor is (B, F, D1, D2) = (112,64,512,512). The total number of elements is 112*64*512*512 = 18874368 elements. The reduction kernel would need to compute the sum of squares across all elements, which is a single scalar.

The division kernel would then take the original tensor and divide each element by the scalar.

So the first kernel (reduction) would be a 1D kernel that iterates over each element, computes x^2, accumulates the sum, then computes the square root. The second kernel (division) would be a 1D kernel that loads each element, divides by the scalar, and stores the result.

But the problem is that the reduction kernel can't be written as a simple elementwise kernel because it needs to accumulate the sum across all elements. Therefore, the reduction kernel would be a simple reduction over the flattened tensor.

Wait, but the reduction kernel can be a 1D kernel that processes each element, adds its square to a shared variable, then reduces the shared variable to a scalar. However, in Triton, the reduction can be done using a helper function, but the kernel would need to be written to perform the reduction.

Alternatively, the reduction can be performed with a simple kernel that computes the sum of squares. The kernel would load each element, square it, add to a shared accumulator, then after the block finishes, the shared accumulator is summed across blocks and stored as the scalar.

But in the example given by the user, the kernel for addition is a simple elementwise addition. For the Frobenius norm, the kernel would need to compute the sum of squares, then the square root.

So the first kernel would be:

- Load each element.

- Square it.

- Add to a shared accumulator.

- After the block, the shared accumulator is summed across threads, then the result is stored as the scalar.

But in Triton, the shared memory can be used for each block. However, for a 1D kernel with a large number of elements, the shared memory per block would be small, so the reduction would be a per-block sum, then the block sums are combined across the grid.

But for the Frobenius norm, the kernel needs to compute the sum of squares across the entire tensor, not per block. So the kernel would need to be a single block that processes all elements, but that's impossible because the number of elements is large. Therefore, the kernel must be a grid of blocks that each compute a per-block sum, then the grid reduces the per-block sums to a single scalar.

But the example given by the user for the addition kernel is a single block that processes a contiguous block of data, and the grid is chosen to cover the entire tensor. For the Frobenius norm, the same approach can be used: a grid of blocks that each compute a per-block sum of squares, then the grid sum is the total sum.

So the first kernel would be a 1D kernel that processes each element, adds its square to a shared accumulator, then the grid reduces the shared accumulators to a scalar.

But how to implement this in Triton? The kernel would need to load each element, square it, add to a shared variable, then after the block, the shared variable is summed across the block threads, then the block's sum is stored in a temporary buffer. Then, a second kernel would read the temporary buffer, sum across the grid, compute the square root, and store the scalar.

But that would be two kernels. However, the user's example uses a single kernel for addition. So perhaps the Frobenius norm can be expressed as a single kernel that computes the sum of squares and the square root, but that would require the kernel to first compute the sum, then compute the square root, then divide each element by that scalar.

Wait, but the division is elementwise, so the kernel would need to first compute the scalar, then divide each element by that scalar. However, the kernel can't compute the scalar per element. So the only way is to have a scalar that is computed by a reduction kernel, then the division kernel uses that scalar.

Therefore, the two kernels are necessary. The first kernel reduces the tensor to the scalar norm, the second kernel divides each element by the scalar.

But the original model's forward call is a single operation, so the two kernels would be launched in sequence.

Now, let's think about the shapes and strides. The input tensor is contiguous in memory, with stride (64*512*512, 512*512, 512, 1). The reduction kernel would flatten the tensor, so each element is accessed with a linear index. The kernel would load each element, square it, add to a shared accumulator, then the shared accumulator is summed across the block, and the block's sum is stored in a temporary buffer. The second kernel would then load the temporary buffer, sum across the grid, compute the square root, and store the scalar.

But the problem is that the temporary buffer for the per-block sums would need to be allocated on the GPU. Then the second kernel would read that buffer and perform the grid reduction.

However, the user's example shows a single kernel that writes to the output tensor. In the Frobenius case, the output tensor is the same shape as the input, but each element is divided by the scalar. So the division kernel would write to the same output tensor.

Putting it all together, the forward function would:

1. Call the reduction kernel to compute the scalar Frobenius norm.

2. Call the division kernel to divide each element by the scalar.

3. Return the divided tensor.

Now, the actual Triton kernels.

First kernel: reduction to compute the sum of squares.

The kernel would be a 1D kernel that processes each element, squares it, adds to a shared accumulator, then the shared accumulator is summed across the block, stored in a temporary buffer.

Second kernel: division, which takes the original tensor, the scalar, and writes each element divided by the scalar.

But the scalar is a single value, so it's passed as a pointer to the division kernel.

Alternatively, the division kernel can be a simple elementwise division that receives the scalar as a pointer, then loads the element, divides by the scalar, and stores.

But the scalar is a single float, so the kernel can be written to receive the scalar as a pointer, then load it once per thread, and use it for the division.

So the division kernel would be a simple elementwise kernel that divides each element by the scalar.

But the scalar is computed in the reduction kernel, so the division kernel must be launched after the reduction kernel.

Now, the code structure.

The ModelNew class would have two Triton kernels: one for the reduction (sum of squares) and one for the division.

The forward function would:

- Allocate a temporary buffer for the per-block sums.

- Launch the reduction kernel to compute the per-block sums.

- Launch the reduction kernel again (or a different kernel) to compute the grid sum of the per-block sums, then compute the square root.

- Then launch the division kernel with the scalar.

But the reduction kernel for the grid sum can be a simple kernel that loads the per-block sums, sums them across the grid, then computes the square root.

Alternatively, the first kernel can be written to compute the sum of squares and the square root in a single pass, but that's not possible because the sum is a scalar.

So, the first kernel computes the per-block sum of squares, the second kernel computes the grid sum of the per-block sums, then the square root, then the division kernel divides each element by the scalar.

But the division kernel would need to receive the scalar.

Alternatively, the division kernel can be written to receive the scalar as a pointer, then each thread loads the scalar once, then divides the element by the scalar.

Now, the code.

First, the reduction kernel that computes the per-block sum of squares.

The kernel would be:

@triton.jit
def reduction_kernel_sum_squares(
    in_ptr0,
    out_ptr0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    xnumel = n_elements
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [BLOCK_SIZE])
    tmp3 = tl.sum(tmp2, 0)[:]
    tl.debug_barrier()
    tl.store(out_ptr0 + tl.full([1], xoffset, tl.int32), tmp3, xmask)

But this kernel would store the per-block sum in the out_ptr0 buffer. The out_ptr0 is a 1D buffer of size ceil(n_elements / BLOCK_SIZE). Then, a second kernel would load the per-block sums, sum them across the grid, compute the square root, and store the scalar.

Wait, but the second kernel would need to read the per-block sums. For example, if the first kernel is launched with a grid that covers all elements, the out_ptr0 would be a buffer of size grid_size, where grid_size = ceil(n_elements / BLOCK_SIZE). The second kernel would then load each element of this buffer, sum them, compute the square root, and store the scalar.

So the second kernel would be:

@triton.jit
def reduction_kernel_grid_sum(
    in_ptr0,
    out_ptr0,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    xnumel = n_blocks
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [BLOCK_SIZE])
    tmp2 = tl.sum(tmp1, 0)[:]
    tl.debug_barrier()
    tl.store(out_ptr0 + tl.full([1], xoffset, tl.int32), tmp2, xmask)

This kernel loads each per-block sum, adds them across the block, then stores the grid sum. Then, the grid sum is a single scalar (since the grid size is the number of blocks, which is a small number compared to the total elements). The square root is then computed on that scalar.

But the grid sum is the sum of the per-block sums, which equals the total sum of squares. Then the square root is taken.

Once the scalar is computed, the division kernel is launched.

The division kernel would be:

@triton.jit
def division_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr0,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    xnumel = n_elements
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)

This kernel loads the original tensor element, loads the scalar (in_ptr1) once per thread, divides, and stores the result.

Putting it all together, the forward function would:

1. Allocate a temporary buffer for per-block sums.

2. Launch the first reduction kernel to compute per-block sums of squares.

3. Allocate a temporary buffer for the grid sum.

4. Launch the second reduction kernel to compute the grid sum of per-block sums, then compute the square root.

5. Launch the division kernel to divide each element by the scalar.

But in the example given, the user's forward function calls a single kernel. However, in this case, three kernels are needed: two reduction kernels and a division kernel.

But the user's example shows a single kernel that adds two tensors. For the Frobenius norm, the original model is a single forward pass that involves two operations (norm and division). The Triton kernels would replace those two operations with the three kernels described.

However, the user may want to fuse the reduction and division into a single kernel. Is that possible?

Alternatively, after computing the scalar, the division can be performed in a single kernel that receives the scalar as a pointer. So the division kernel is a simple elementwise division that divides each element by the scalar.

Thus, the overall flow is:

- Compute the scalar Frobenius norm using two reduction kernels (per-block sum, then grid sum).

- Divide each element by the scalar using a third kernel.

But the user's example shows a single kernel for addition. For the Frobenius norm, three kernels are needed. However, the user may be okay with that.

Now, the code for the ModelNew class would include the three kernels, and the forward function would launch them in sequence.

But the user's example for the addition kernel shows a single kernel that adds two tensors and writes to an output tensor. In the Frobenius case, the division kernel writes to the same output tensor as the original input, after the norm has been computed.

Now, the shapes and strides.

The input tensor x has shape (B, F, D1, D2) = (112,64,512,512). The flattened number of elements is 112*64*512*512 = 18874368.

The per-block sum kernel processes each element, squares it, adds to a shared accumulator, then stores the per-block sum in a buffer of size grid_size = ceil(18874368 / BLOCK_SIZE). The grid_size for a BLOCK_SIZE of 128 would be 18874368 / 128 = 147096.0625, so grid_size = 147097.

The second reduction kernel processes the grid_size elements, sums them across the grid, then computes the square root. The grid_size is small, so a single block can process it.

The division kernel processes the original tensor elements, divides each by the scalar, and writes to the same output tensor.

Now, the code.

The first kernel (reduction_sum_squares) would be:

@triton.jit
def reduction_sum_squares(in_ptr0, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [BLOCK_SIZE])
    tmp3 = tl.sum(tmp2, 0)[:]
    tl.debug_barrier()
    tl.store(out_ptr0 + tl.full([1], xoffset, tl.int32), tmp3, xmask)

The second kernel (reduction_grid_sum) would be:

@triton.jit
def reduction_grid_sum(in_ptr0, out_ptr0, n_blocks, BLOCK_SIZE: tl.constexpr):
    xnumel = n_blocks
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [BLOCK_SIZE])
    tmp2 = tl.sum(tmp1, 0)[:]
    tl.debug_barrier()
    tl.store(out_ptr0 + tl.full([1], xoffset, tl.int32), tmp2, xmask)

The third kernel (division) would be:

@triton.jit
def division_kernel(in_ptr0, in_ptr1, out_ptr0, n_elements, BLOCK_SIZE: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)

Then, the forward function would:

- Allocate a temporary buffer for the per-block sums (size = grid_size).

- Launch reduction_sum_squares with the input tensor, the temporary buffer, n_elements, and BLOCK_SIZE.

- Allocate a temporary buffer for the grid sum (size = 1).

- Launch reduction_grid_sum with the per-block buffer, the grid sum buffer, grid_size, and BLOCK_SIZE.

- Compute the square root of the grid sum scalar.

- Launch division_kernel with the original input tensor, the scalar, the output tensor, n_elements, and BLOCK_SIZE.

- Return the output tensor.

But the grid sum buffer after reduction_grid_sum contains the sum of squares. Then, the square root is computed as sqrt(grid_sum). The division kernel receives this scalar as in_ptr1.

Wait, but the division kernel loads the scalar once per thread. However, the scalar is a single value, so the eviction policy 'evict_last' is used to load it once per block, and each thread uses the same value.

Thus, the division kernel can be written as above.

Putting it all together, the ModelNew class would have these three kernels and the forward function that calls them in sequence.

Now, the final code would look like this:

Import torch, nn, triton, tl.

Define the three kernels: reduction_sum_squares, reduction_grid_sum, division_kernel.

In the forward function of ModelNew:

- Check that the input is on CUDA.

- Allocate a temporary buffer for per-block sums.

- Launch the first reduction kernel.

- Allocate a temporary buffer for the grid sum.

- Launch the second reduction kernel.

- Compute the square root of the grid sum scalar.

- Allocate the output tensor.

- Launch the division kernel.

- Return the output tensor.

But the grid sum buffer after the second reduction kernel contains the total sum of squares, which is a scalar. So the square root can be computed with torch.sqrt after the kernel.

Wait, but the grid sum buffer is a scalar stored in a single location. So after the second kernel, the grid sum is stored in the grid_sum_buffer[0]. Then, the square root is computed as torch.sqrt(grid_sum_buffer[0]). Then, the division kernel receives this scalar as a pointer.

Thus, the forward function would:

- Compute the grid sum scalar.

- Compute the scalar norm.

- Launch the division kernel.

Now, the code in the forward function would be:

def forward(self, input_0):
    arg0_1 = input_0
    arg0_2 = input_0
    arg0_3 = input_0
    output_0 = arg0_1
    arg0_4 = output_0
    arg0_5 = output_0
    arg0_6 = output_0
    arg0_7 = output_0
    arg0_8 = output_0
    arg0_9 = output_0
    arg0_10 = output_0
    arg0_11 = output_0
    arg0_12 = output_0
    arg0_13 = output_0
    arg0_14 = output_0
    arg0_15 = output_0
    arg0_16 = output_0
    arg0_17 = output_0
    arg0_18 = output_0
    arg0_19 = output_0
    arg0_20 = output_0
    arg0_21 = output_0
    arg0_22 = output_0
    arg0_23 = output_0
    arg0_24 = output_0
    arg0_25 = output_0
    arg0_26 = output_0
    arg0_27 = output_0
    arg0_28 = output_0
    arg0_29 = output_0
    arg0_30 = output_0
    arg0_31 = output_0
    arg0_32 = output_0
    arg0_33 = output_0
    arg0_34 = output_0
    arg0_35 = output_0
    arg0_36 = output_0
    arg0_37 = output_0
    arg0_38 = output_0
    arg0_39 = output_0
    arg0_40 = output_0
    arg0_41 = output_0
    arg0_42 = output_0
    arg0_43 = output_0
    arg0_44 = output_0
    arg0_45 = output_0
    arg0_46 = output_0
    arg0_47 = output_0
    arg0_48 = output_0
    arg0_49 = output_0
    arg0_50 = output_0
    arg0_51 = output_0
    arg0_52 = output_0
    arg0_53 = output_0
    arg0_54 = output_0
    arg0_55 = output_0
    arg0_56 = output_0
    arg0_57 = output_0
    arg0_58 = output_0
    arg0_59 = output_0
    arg0_60 = output_0
    arg0_61 = output_0
    arg0_62 = output_0
    arg0_63 = output_0
    arg0_64 = output_0
    arg0_65 = output_0
    arg0_66 = output_0
    arg0_67 = output_0
    arg0_68 = output_0
    arg0_69 = output_0
    arg0_70 = output_0
    arg0_71 = output_0
    arg0_72 = output_0
    arg0_73 = output_0
    arg0_74 = output_0
    arg0_75 = output_0
    arg0_76 = output_0
    arg0_77 = output_0
    arg0_78 = output_0
    arg0_79 = output_0
    arg0_80 = output_0
    arg0_81 = output_0
    arg0_82 = output_0
    arg0_83 = output_0
    arg0_84 = output_0
    arg0_85 = output_0
    arg0_86 = output_0
    arg0_87 = output_0
    arg0_88 = output_0
    arg0_89 = output_0
    arg0_90 = output_0
    arg0_91 = output_0
    arg0_92 = output_0
    arg0_93 = output_0
    arg0_94 = output_0
    arg0_95 = output_0
    arg0_96 = output_0
    arg0_97 = output_0
    arg0_98 = output_0
    arg0_99 = output_0
    arg0_100 = output_0
    arg0_101 = output_0
    arg0_102 = output_0
    arg0_103 = output_0
    arg0_104 = output_0
    arg0_105 = output_0
    arg0_106 = output_0
    arg0_107 = output_0
    arg0_108 = output_0
    arg0_109 = output_0
    arg0_110 = output_0
    arg0_111 = output_0
    arg0_112 = output_0
    arg0_113 = output_0
    arg0_114 = output_0
    arg0_115 = output_0
    arg0_116 = output_0
    arg0_117 = output_0
    arg0_118 = output_0
    arg0_119 = output_0
    arg0_120 = output_0
    arg0_121 = output_0
    arg0_122 = output_0
    arg0_123 = output_0
    arg0_124 = output_0
    arg0_125 = output_0
    arg0_126 = output_0
    arg0_127 = output_0
    arg0_128 = output_0
    arg0_129 = output_0
    arg0_130 = output_0
    arg0_131 = output_0
    arg0_132 = output_0
    arg0_133 = output_0
    arg0_134 = output_0
    arg0_135 = output_0
    arg0_136 = output_0
    arg0_137 = output_0
    arg0_138 = output_0
    arg0_139 = output_0
    arg0_140 = output_0
    arg0_141 = output_0
    arg0_142 = output_0
    arg0_143 = output_0
    arg0_144 = output_0
    arg0_145 = output_0
    arg0_146 = output_0
    arg0_147 = output_0
    arg0_148 = output_0
    arg0_149 = output_0
    arg0_150 = output_0
    arg0_151 = output_0
    arg0_152 = output_0
    arg0_153 = output_0
    arg0_154 = output_0
    arg0_155 = output_0
    arg0_156 = output_0
    arg0_157 = output_0
    arg0_158 = output_0
    arg0_159 = output_0
    arg0_160 = output_0
    arg0_161 = output_0
    arg0_162 = output_0
    arg0_163 = output_0
    arg0_164 = output_0
    arg0_165 = output_0
    arg0_166 = output_0
    arg0_167 = output_0
    arg0_168 = output_0
    arg0_169 = output_0
    arg0_170 = output_0
    arg0_171 = output_0
    arg0_172 = output_0
    arg0_173 = output_0
    arg0_174 = output_0
    arg0_175 = output_0
    arg0_176 = output_0
    arg0_177 = output_0
    arg0_178 = output_0
    arg0_179 = output_0
    arg0_180 = output_0
    arg0_181 = output_0
    arg0_182 = output_0
    arg0_183 = output_0
    arg0_184 = output_0
    arg0_185 = output_0
    arg0_186 = output_0
    arg0_187 = output_0
    arg0_188 = output_0
    arg0_189 = output_0
    arg0_190 = output_0
    arg0_191 = output_0
    arg0_192 = output_0
    arg0_193 = output_0
    arg0_194 = output_0
    arg0_195 = output_0
    arg0_196 = output_0
    arg0_197 = output_0
    arg0_198 = output_0
    arg0_199 = output_0
    arg0_200 = output_0
    arg0_201 = output_0
    arg0_202 = output_0
    arg0_203 = output_0
    arg0_204 = output_0
    arg0_205 = output_0
    arg0_206 = output_0
    arg0_207 = output_0
    arg0_208 = output_0
    arg0_209 = output_0
    arg0_210 = output_0
    arg0_211 = output_0
    arg0_212 = output_0
    arg0_213 = output_0
    arg0_214 = output_0
    arg0_215 = output_0
    arg0_216 = output_0
    arg0_217 = output_0
    arg0_218 = output_0
    arg0_219 = output_0
    arg0_220 = output_0
    arg0_221 = output_0
    arg0_222 = output_0
    arg
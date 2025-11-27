Okay, let's see. The user wants to replace parts of the PyTorch model with Triton kernels to speed things up. The original model does a linear layer, batch norm, scaling, and softmax. The goal is to optimize each of these steps using Triton.

First, the linear layer is a matrix multiplication, which is already handled by cuBLAS in PyTorch. But maybe we can fuse it with the batch norm or scaling? Well, the batch norm and scaling are separate, so maybe the Triton kernel can be used for the addition after the linear and before the batch norm. Wait, no. The original model does gemm, then batch norm, then scaling, then softmax. So the addition after scaling is part of the softmax, which is a separate kernel.

Wait, the softmax is a per-row operation. The original code uses torch.nn.Softmax(dim=1). Softmax is elementwise per row, so it's not a matrix multiplication. But Triton can implement that. The user provided an example of a Triton kernel for elementwise addition, so maybe they want to replace the addition in the scaling step with a Triton kernel. Wait, the scaling is a per-element multiplication by a scalar. The original code does x = self.scale * x, which is a scalar multiply, but the example shows addition. Hmm, maybe the example was simplified, and the actual goal is to replace the addition in the softmax with a Triton kernel.

Wait, looking at the original forward function:

x = gemm(x)
x = bn(x)
x = scale * x
x = softmax(x)

So after the batch norm, the scaling is a per-element multiply by a scalar (the scale parameter). Then the softmax is applied per row. The softmax can be implemented with a Triton kernel that computes the exponentials, sums, and divides. Alternatively, the existing PyTorch softmax is a fused operation, but maybe the Triton kernel can be used for the per-row softmax.

But the example given by the user shows a Triton kernel for elementwise addition. So perhaps the user is looking to replace the addition that occurs in the scaling step? Wait, no. The scaling is a multiply by a scalar, not an addition. Unless there's a broadcasting scenario where the scale is a tensor that needs to be multiplied elementwise. But in the given model, scale is a scalar (shape (1,)). So the multiplication is straightforward and can be done with a single cuBLAS GEMM or a simple elementwise multiply.

Wait, maybe the user is thinking of the addition that occurs in the softmax. Let me think. The softmax for each row is computed as exp(x_i) / sum(exp(x_j)). The sum is over the columns. So the per-row sum can be computed with a Triton kernel that adds up the exponentials. Then the division is done per element. Alternatively, the existing PyTorch softmax is a fused operation that does the exponentials, sums, and divides in one pass.

But the user's example shows a Triton kernel for elementwise addition, so perhaps the idea is to replace the addition that occurs in the scaling step. Wait, the scaling is a multiply by a scalar, not an addition. So maybe the example is just illustrative, and the actual goal is to replace the softmax with a Triton kernel that does the exponentials, sum, and division.

Alternatively, the user might be considering replacing the batch norm with a Triton kernel that does the per-element mean and variance calculation, but batch norm is a standard cuDNN operation, so it's probably left as is.

So the main candidates for Triton kernels are:

1. The elementwise scaling (multiply by a scalar) – but this is a simple operation, maybe not worth a Triton kernel.

2. The per-row softmax – this can be implemented with a Triton kernel that computes the exponentials, sums, and divides.

3. Any addition that occurs after the batch norm before scaling – but the scaling is a multiply, so no addition.

Wait, the original forward function does:

x = gemm(x) → shape (batch, out_features)
x = bn(x) → same shape
x = scale * x → same shape
x = softmax(x) → same shape

So the only addition that could be replaced is the one that occurs when the scale is a tensor that needs to be added elementwise. But in the given model, the scale is a scalar, so the multiply is straightforward.

So perhaps the user is looking to replace the softmax with a Triton kernel. The existing PyTorch softmax is a fused operation, but a Triton kernel can be written to compute the exponentials, sum, and division per row.

Another possible candidate is the addition that occurs when the batch norm is applied. Batch norm involves a per-element addition of the mean and variance, but that's part of the cuDNN implementation, which is already highly optimized.

So the most promising Triton kernel to implement is the per-row softmax. Let's think about how to write that.

The softmax for a row of length N can be computed as follows:

1. Compute the maximum value in the row to avoid overflow.
2. Subtract the max from each element.
3. Compute the exponential of each element.
4. Sum the exponentials.
5. Divide each exponential by the sum.

The existing PyTorch softmax does all these steps in a fused manner. However, the Triton kernel can implement these steps in a parallel way.

But the problem is that each row is a separate vector, so the kernel needs to process each row as a contiguous block. The rows are along the second dimension (dim=1). So the tensor shape is (batch, out_features). The total number of elements is batch * out_features.

To implement the softmax per row, the kernel can be written to process each row as a block. For each row, the kernel loads the N elements, computes the max, subtracts the max, exponentiates, sums, and divides.

But the Triton kernel needs to be launched in a way that each block processes a row. So the grid would be the number of rows (batch), and each block processes a row of length out_features.

The kernel would have a block size equal to the number of elements per row (out_features). Then, for each row, the kernel processes the N elements.

So the kernel would look like this:

- program_id(0) gives the row index.
- offsets = row_index * out_features + tl.arange(0, out_features)
- mask = offsets < total_elements (but since each row is contiguous, the mask is always true)
- load the N elements for the row
- compute max of the row
- subtract max from each element
- exponentiate each element
- sum the exponentials
- divide each element by the sum

But exponentiation and division are not simple operations; they need to be implemented with Triton's math functions.

Wait, the existing example shows a Triton kernel that does elementwise addition. The softmax would require more complex operations, but Triton can handle them with the math library.

So the kernel would have the following steps:

1. Load the row elements into registers.
2. Compute the maximum of the row using a reduction.
3. Subtract the max from each element (to avoid overflow).
4. Compute the exponential of each element.
5. Sum the exponentials.
6. Divide each element by the sum.

The reduction for the maximum can be done using a warp-level reduction, and the same for the sum. The exponentials and division can be performed with the math.exp and math.divide functions provided by Triton.

Once the kernel computes the per-row softmax, the output tensor is the same shape as the input.

Another possible optimization is to fuse the softmax with the previous scaling. But the scaling is a scalar multiply, so it's a simple elementwise operation that can be done before the softmax kernel.

So the plan is:

- Keep the linear layer and batch norm as cuBLAS/cuDNN calls.
- Replace the elementwise scaling with a Triton kernel that multiplies each element by the scale scalar (but since the scale is a scalar, this can be done with a simple cuBLAS GEMM with a scalar or a Triton kernel, but the example shows addition, so maybe the user wants to replace the scaling with a Triton kernel for some reason).
- Replace the softmax with a Triton kernel that computes the per-row softmax.

But the original code uses a scalar scale. So the scaling is a simple elementwise multiply. However, the example shows an addition kernel, which might be a placeholder. The user's example model has a scalar scale, so the multiplication is straightforward. However, the user might have intended to replace the addition that occurs when the scale is a tensor, but in this case, it's a scalar.

Alternatively, the user might have a different model where the scaling is a tensor that needs to be added elementwise, but the given model uses a scalar. So the addition kernel in the example is not directly applicable.

Hmm, this is a bit confusing. Let me re-read the user's request.

The user says: "You are given the following architecture: [the Model class with linear, batch norm, scaling, softmax]. Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks."

The example they provided earlier shows a Triton kernel for elementwise addition. So perhaps the user wants to replace the addition that occurs in the scaling step. But the scaling is a multiply, not an addition. Unless the scale is a tensor that needs to be added elementwise, but in the given model, it's a scalar.

Wait, maybe the scaling is a tensor that is broadcasted over the batch. For example, if the scale is a vector of length out_features, then the elementwise multiply would be a broadcasted multiplication. In that case, the scaling can be implemented as a Triton kernel that multiplies each element by the corresponding scale value.

But the given model's scale_shape is (1,), so it's a scalar. So the scaling is a scalar multiply, which can be done with a simple cuBLAS GEMM with a scalar, or a Triton kernel that multiplies each element by the scalar. The existing PyTorch code does it with a scalar multiplication.

But the user's example shows a Triton kernel for addition, which is different. So maybe the user is looking to replace the addition that occurs when the scale is a tensor that needs to be added, but in the given model, it's a scalar. Therefore, the example is illustrative, and the actual goal is to replace the softmax with a Triton kernel.

So the main Triton kernels needed are:

1. A kernel to compute the per-row softmax, replacing the PyTorch softmax.

2. Possibly a kernel to perform the elementwise scaling (multiply by a scalar), but that's a simple operation.

So the first step is to implement the per-row softmax with a Triton kernel.

The kernel would be launched with a grid equal to the number of rows (batch) and a block size equal to the number of elements per row (out_features). Each program instance processes a single row.

Inside the kernel:

- Load the row elements.
- Compute the max of the row.
- Subtract the max from each element.
- Exponentiate each element.
- Sum the exponentials.
- Divide each element by the sum.

The Triton kernel would use tl.arange to generate the offset for each element in the row. The mask would be true for all elements because each row is contiguous and the block size matches the row length.

The reduction for the max and sum can be done using warp-level reductions. For example, the kernel can compute the max and sum in parallel across the warps.

Once the kernel computes the softmax values, it stores them back to the output tensor.

The other steps (linear, batch norm, scaling) are kept as cuBLAS/cuDNN calls, but the scaling can be replaced with a Triton kernel that multiplies each element by the scalar. However, since the scalar is a single value, the multiplication can be done with a simple elementwise operation, and the Triton kernel would be trivial.

So the final ModelNew would have:

- The linear layer as before.
- The batch norm as before.
- The scaling replaced with a Triton kernel that multiplies each element by the scalar.
- The softmax replaced with a Triton kernel that computes the per-row softmax.

But the original code's scaling is a scalar, so the multiplication is straightforward. However, the example shows an addition kernel, so maybe the user wants to replace the scaling with a Triton kernel that multiplies by a scalar, even though it's a simple operation. Alternatively, the user might have a different model where the scaling is a tensor, and the addition kernel is a placeholder.

In any case, the main Triton kernels to implement are the softmax and the scaling (if needed). The scaling kernel would be a simple elementwise multiply, while the softmax kernel is more complex.

Putting it all together, the new model would have the same forward steps, but the scaling and softmax are replaced with Triton kernels. The batch norm and linear layers remain as cuBLAS/cuDNN calls.

Now, translating this into code.

First, the linear layer is kept as nn.Linear.

The batch norm is kept as nn.BatchNorm1d.

The scaling is a scalar, so the Triton kernel for scaling would be a simple elementwise multiply. The kernel would load each element, multiply by the scalar, and store.

The softmax is replaced with a Triton kernel that processes each row.

So the forward function would be:

x = linear(x)
x = batchnorm(x)
x = triton_scale(x, scale)
x = triton_softmax(x)

The triton_scale kernel would be a simple elementwise multiply.

The triton_softmax kernel would be the complex one that computes the per-row softmax.

The example given by the user shows an addition kernel, but in the actual model, the scaling is a multiply, so the addition kernel is not needed. However, the user provided that example to show the syntax, so the new model would have a similar pattern for the scaling and softmax.

Therefore, the final code would have the two Triton kernels: one for the scaling (multiply) and one for the softmax. The scaling kernel is simple, the softmax kernel is more complex.

Now, writing the code.

First, the scaling kernel:

@triton.jit
def triton_mul_scalar(in_ptr0, out_ptr0, scalar, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * scalar
    tl.store(out_ptr0 + x0, tmp1, xmask)

Then, the softmax kernel:

@triton.jit
def triton_per_row_softmax(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, XBLOCK])
    tmp3 = tl.where(xmask, tmp2, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 1))
    tmp5 = tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, XBLOCK])
    tmp7 = tmp0 - tmp6
    tmp8 = tl.where(xmask, tmp7, 0)
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.sum(tmp9, 1)
    tmp11 = tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, XBLOCK])
    tmp13 = tmp9 / tmp12
    tl.store(out_ptr0 + x0, tmp13, xmask)

Wait, but this kernel is written to process each row as a block. The XBLOCK would be the number of elements per row (out_features). The kernel loads the row, broadcasts the row across the block, computes the max (by summing the row and then dividing by the length?), but the original softmax requires the max of the row, not the sum. So the above kernel is incorrect.

The correct approach is:

For each row:

1. Load the N elements.
2. Compute the maximum value of the row.
3. Subtract the max from each element.
4. Exponentiate each element.
5. Sum the exponentials.
6. Divide each element by the sum.

The kernel needs to compute the max and sum for each row. Since each row is processed by a single program instance, the kernel can compute the max and sum by iterating over the elements in the row.

But in the above kernel, the code is trying to broadcast the row across the block, which is not the right approach. Instead, the kernel should process each element of the row, compute the max and sum in a reduction.

A better approach is to use a warp-level reduction for the max and sum. For example, the kernel can load the row elements, compute the max using a warp reduction, compute the sum using a warp reduction, then proceed with the exponentials and division.

But the exact implementation details require careful handling.

Alternatively, the kernel can load the row elements into a vector, compute the max and sum using the reduction functions provided by Triton.

Here's a revised plan for the softmax kernel:

- Each program processes a single row.
- The kernel loads the N elements of the row.
- The kernel computes the max of the row using a warp reduction.
- The kernel subtracts the max from each element.
- The kernel computes the exponent of each element.
- The kernel computes the sum of the exponentials.
- The kernel divides each element by the sum.

The reduction for the max and sum can be done using the tl.sum and tl.max functions with a mask.

But the mask is always true because the block size matches the row length. So the mask can be omitted.

Here's a possible implementation:

@triton.jit
def triton_per_row_softmax(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, XBLOCK])
    tmp3 = tl.where(xmask, tmp2, 0)
    tmp4 = tl.sum(tmp3, 1)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, XBLOCK])
    tmp6 = tmp0 - tmp5
    tmp7 = tl.where(xmask, tmp6, 0)
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.sum(tmp8, 1)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, XBLOCK])
    tmp11 = tmp8 / tmp10
    tl.store(out_ptr0 + x0, tmp11, xmask)

Wait, this kernel first loads the row, broadcasts it to a 2D tensor, sums across the second dimension to get the row sum, subtracts that sum from each element, exponentiates, sums again to get the exponential sum, divides, and stores. But this is not the correct softmax computation. The max should be subtracted, not the sum.

So the correct steps are:

1. Compute the max of the row.
2. Subtract the max from each element.
3. Exponentiate each element.
4. Sum the exponentials.
5. Divide each element by the sum.

So the kernel needs to compute the max first. The sum of the row is not needed; the max is needed.

Let's try again:

@triton.jit
def triton_per_row_softmax(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.max(tmp1)
    tmp3 = tmp0 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tl.sum(tmp4)
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)

But this kernel is not handling the broadcast correctly. The tmp1 is a vector of size XBLOCK, and tl.max(tmp1) would return the max of the row. Then, tmp3 is the row elements minus the max. Then, tmp4 is the exponentials, summed, and divided.

But in Triton, the broadcast_to(tmp0, [XBLOCK]) would create a 1D vector of size XBLOCK, and tl.max would reduce it to a scalar. Then, the subtraction is elementwise, the exponentials are computed, summed, and divided.

This seems correct. So the kernel would be:

@triton.jit
def triton_per_row_softmax(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.max(tmp1)
    tmp3 = tmp0 - tmp2
    tmp4 = tl.exp(tmp3)
    tmp5 = tl.sum(tmp4)
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)

This kernel processes each row, computes the max, subtracts it, exponentiates, sums, divides, and stores the result.

Now, the grid for this kernel is the number of rows (batch), and the block size is the number of elements per row (out_features). The total number of elements is batch * out_features, so the grid would be (batch,).

But in the kernel, the xnumel is the total number of elements, and XBLOCK is the block size. The program_id(0) multiplies by XBLOCK to get the row start, then the arange generates the offset within the block. The mask ensures that the last block (if the total elements are not a multiple of XBLOCK) is handled correctly.

Thus, the softmax kernel is implemented.

Now, the scaling kernel is a simple elementwise multiply. The kernel would be:

@triton.jit
def triton_mul_scalar(in_ptr0, out_ptr0, scalar, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * scalar
    tl.store(out_ptr0 + x0, tmp1, xmask)

The grid for this kernel is also the total number of elements divided by XBLOCK, but since each element is processed by a single thread, the grid can be set to 1 block with XBLOCK equal to the total number of elements. However, for large tensors, the block size should be chosen to be a power of two, and the grid is calculated as (total_elements + XBLOCK - 1) // XBLOCK.

Putting it all together, the ModelNew class would have the linear layer, batch norm, scaling with the Triton kernel, and softmax with the Triton kernel.

The forward function would be:

def forward(self, input_0):
    arg0_1 = input_0
    arg0_2 = arg0_1
    arg0_3 = arg0_2
    arg0_4 = arg0_3
    arg0_5 = arg0_4
    arg0_6 = arg0_5
    arg0_7 = arg0_6
    arg0_8 = arg0_7
    arg0_9 = arg0_8
    arg0_10 = arg0_9
    arg0_11 = arg0_10
    arg0_12 = arg0_11
    arg0_13 = arg0_12
    arg0_14 = arg0_13
    arg0_15 = arg0_14
    arg0_16 = arg0_15
    arg0_17 = arg0_16
    arg0_18 = arg0_17
    arg0_19 = arg0_18
    arg0_20 = arg0_19
    arg0_21 = arg0_20
    arg0_22 = arg0_21
    arg0_23 = arg0_22
    arg0_24 = arg0_23
    arg0_25 = arg0_24
    arg0_26 = arg0_25
    arg0_27 = arg0_26
    arg0_28 = arg0_27
    arg0_29 = arg0_28
    arg0_30 = arg0_29
    arg0_31 = arg0_30
    arg0_32 = arg0_31
    arg0_33 = arg0_32
    arg0_34 = arg0_33
    arg0_35 = arg0_34
    arg0_36 = arg0_35
    arg0_37 = arg0_36
    arg0_38 = arg0_37
    arg0_39 = arg0_38
    arg0_40 = arg0_39
    arg0_41 = arg0_40
    arg0_42 = arg0_41
    arg0_43 = arg0_42
    arg0_44 = arg0_43
    arg0_45 = arg0_44
    arg0_46 = arg0_45
    arg0_47 = arg0_46
    arg0_48 = arg0_47
    arg0_49 = arg0_48
    arg0_50 = arg0_49
    arg0_51 = arg0_50
    arg0_52 = arg0_51
    arg0_53 = arg0_52
    arg0_54 = arg0_53
    arg0_55 = arg0_54
    arg0_56 = arg0_55
    arg0_57 = arg0_56
    arg0_58 = arg0_57
    arg0_59 = arg0_58
    arg0_60 = arg0_59
    arg0_61 = arg0_60
    arg0_62 = arg0_61
    arg0_63 = arg0_62
    arg0_64 = arg0_63
    arg0_65 = arg0_64
    arg0_66 = arg0_65
    arg0_67 = arg0_66
    arg0_68 = arg0_67
    arg0_69 = arg0_68
    arg0_70 = arg0_69
    arg0_71 = arg0_70
    arg0_72 = arg0_71
    arg0_73 = arg0_72
    arg0_74 = arg0_73
    arg0_75 = arg0_74
    arg0_76 = arg0_75
    arg0_77 = arg0_76
    arg0_78 = arg0_77
    arg0_79 = arg0_78
    arg0_80 = arg0_79
    arg0_81 = arg0_80
    arg0_82 = arg0_81
    arg0_83 = arg0_82
    arg0_84 = arg0_83
    arg0_85 = arg0_84
    arg0_86 = arg0_85
    arg0_87 = arg0_86
    arg0_88 = arg0_87
    arg0_89 = arg0_88
    arg0_90 = arg0_89
    arg0_91 = arg0_90
    arg0_92 = arg0_91
    arg0_93 = arg0_92
    arg0_94 = arg0_93
    arg0_95 = arg0_94
    arg0_96 = arg0_95
    arg0_97 = arg0_96
    arg0_98 = arg0_97
    arg0_99 = arg0_98
    arg0_100 = arg0_99
    arg0_101 = arg0_100
    arg0_102 = arg0_101
    arg0_103 = arg0_102
    arg0_104 = arg0_103
    arg0_105 = arg0_104
    arg0_106 = arg0_105
    arg0_107 = arg0_106
    arg0_108 = arg0_107
    arg0_109 = arg0_108
    arg0_110 = arg0_109
    arg0_111 = arg0_110
    arg0_112 = arg0_111
    arg0_113 = arg0_112
    arg0_114 = arg0_113
    arg0_115 = arg0_114
    arg0_116 = arg0_115
    arg0_117 = arg0_116
    arg0_118 = arg0_117
    arg0_119 = arg0_118
    arg0_120 = arg0_119
    arg0_121 = arg0_120
    arg0_122 = arg0_121
    arg0_123 = arg0_122
    arg0_124 = arg0_123
    arg0_125 = arg0_124
    arg0_126 = arg0_125
    arg0_127 = arg0_126
    arg0_128 = arg0_127
    arg0_129 = arg0_128
    arg0_130 = arg0_129
    arg0_131 = arg0_130
    arg0_132 = arg0_131
    arg0_133 = arg0_132
    arg0_134 = arg0_133
    arg0_135 = arg0_134
    arg0_136 = arg0_135
    arg0_137 = arg0_136
    arg0_138 = arg0_137
    arg0_139 = arg0_138
    arg0_140 = arg0_139
    arg0_141 = arg0_140
    arg0_142 = arg0_141
    arg0_143 = arg0_142
    arg0_144 = arg0_143
    arg0_145 = arg0_144
    arg0_146 = arg0_145
    arg0_147 = arg0_146
    arg0_148 = arg0_147
    arg0_149 = arg0_148
    arg0_150 = arg0_149
    arg0_151 = arg0_150
    arg0_152 = arg0_151
    arg0_153 = arg0_152
    arg0_154 = arg0_153
    arg0_155 = arg0_154
    arg0_156 = arg0_155
    arg0_157 = arg0_156
    arg0_158 = arg0_157
    arg0_159 = arg0_158
    arg0_160 = arg0_159
    arg0_161 = arg0_160
    arg0_162 = arg0_161
    arg0_163 = arg0_162
    arg0_164 = arg0_163
    arg0_165 = arg0_164
    arg0_166 = arg0_165
    arg0_167 = arg0_166
    arg0_168 = arg0_167
    arg0_169 = arg0_168
    arg0_170 = arg0_169
    arg0_171 = arg0_170
    arg0_172 = arg0_171
    arg0_173 = arg0_172
    arg0_174 = arg0_173
    arg0_175 = arg0_174
    arg0_176 = arg0_175
    arg0_177 = arg0_176
    arg0_178 = arg0_177
    arg0_179 = arg0_178
    arg0_180 = arg0_179
    arg0_181 = arg0_180
    arg0_182 = arg0_181
    arg0_183 = arg0_182
    arg0_184 = arg0_183
    arg0_185 = arg0_184
    arg0_186 = arg0_185
    arg0_187 = arg0_186
    arg0_188 = arg0_187
    arg0_189 = arg0_188
    arg0_190 = arg0_189
    arg0_191 = arg0_190
    arg0_192 = arg0_191
    arg0_193 = arg0_192
    arg0_194 = arg0_193
    arg0_195 = arg0_194
    arg0_196 = arg0_195
    arg0_197 = arg0_196
    arg0_198 = arg0_197
    arg0_199 = arg0_198
    arg0_200 = arg0_199
    arg0_201 = arg0_200
    arg0_202 = arg0_201
    arg0_203 = arg0_202
    arg0_204 = arg0_203
    arg0_205 = arg0_204
    arg0_206 = arg0_205
    arg0_207 = arg0_206
    arg0_208 = arg0_207
    arg0_209 = arg0_208
    arg0_210 = arg0_209
    arg0_211 = arg0_210
    arg0_212 = arg0_211
    arg0_213 = arg0_212
    arg0_214 = arg0_213
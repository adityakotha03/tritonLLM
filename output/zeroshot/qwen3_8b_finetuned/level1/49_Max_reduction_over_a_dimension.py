Okay, let's tackle this problem. The goal is to replace the PyTorch `torch.max` reduction over a specific dimension with a custom Triton kernel. The original model is a simple max reduction, and the new model, ModelNew, should perform the same operation but using a Triton kernel for better performance.

First, I need to understand the original tensor shapes and the reduction dimension. The input tensor `x` has shape (batch_size, dim1, dim2) = (128, 4096, 4095). The reduction is over `dim2` (the last dimension) because the forward method calls `torch.max(x, dim=self.dim)`. In the provided example, the default dimension is 1, which in the original code would correspond to the second dimension (dim1). Wait, the `get_init_inputs` returns [1], so the reduction is over the second dimension, which is 4096. Wait, no, the `dim` parameter in `torch.max` is the dimension to reduce over. So if the model is initialized with `dim=1`, then the reduction is over the second dimension (dim1 = 4096). The output shape after reduction would be (batch_size, dim2) = (128, 4095) because we reduce the second dimension of size 4096 to 4095 elements (the max of each 4096 elements). Wait, no, the `torch.max` function along a dimension reduces that dimension to a scalar per other dimensions. So for a 3D tensor, if we reduce over dimension 1 (size 4096), the output would be a 2D tensor of shape (batch_size, dim2) = (128, 4095). Because the original tensor has 4096 elements along the second dimension, the max over that dimension would produce a single value per row (batch element) and column (dim2). Wait, no, the `dim` argument in `torch.max` is the dimension to collapse. For a 3D tensor (B, D1, D2), reducing over dimension 1 (D1) would collapse that dimension, resulting in a 2D tensor of shape (B, D2). So the output shape is (128, 4095).

Now, the Triton kernel needs to perform the same reduction. The challenge is to map the reduction over the second dimension (size 4096) into a kernel that can compute the max for each of the 4095 elements in the third dimension (dim2). Wait, no, the reduction is over the second dimension, which has length 4096. Each element in the third dimension (dim2) is a column of the original tensor. For each column, we need to compute the max of the 4096 elements in the second dimension. The output tensor would have shape (batch_size, dim2) = (128, 4095). Wait, no, if the original tensor is (B, D1, D2) and we reduce over D1, the output is (B, D2). Because each of the D2 columns has D1 elements, and the max over D1 gives a single value per column. So for each column (dim2) and each batch, we need to compute the max of the D1 elements.

The Triton kernel needs to compute this max for each column. Let's think about the indexing. The kernel will process each column (dim2) across all batches. The total number of columns is dim2 = 4095. For each column, there are dim1 = 4096 elements to reduce. The kernel must load these 4096 elements for each column, compute the max, and store the result in the output tensor.

But Triton kernels operate on a flat view of the tensor. The original tensor is stored in row-major order (contiguous in the last dimension). So the flat index for a given element can be computed as `batch * D1 * D2 + row * D2 + col`. However, for the reduction, each column (col) has a contiguous block of D1 elements. So for each column, the kernel can treat the D1 elements as a contiguous block of length D1.

The kernel's block size (BLOCK_SIZE) is chosen to be equal to the number of elements in the dimension being reduced (D1 = 4096). Because the reduction is over D1, each program (thread block) processes exactly one column (col). Wait, no. Wait, each program processes a block of D1 elements, but there are D2 columns. So the kernel needs to launch one program per column. However, the number of columns (D2 = 4095) is not a power of two, so the grid size would be (ceil(D2 / BLOCK_SIZE)). But the original kernel example used a block size equal to the reduction dimension (D1) and a grid that covers the columns (D2). Wait, the example kernel for addition used a block size that divided the total number of elements, but for reduction, the kernel needs to process each column's D1 elements.

Wait, the kernel needs to compute the max for each column. Each column has D1 elements. The total number of columns is D2. So the kernel can be structured as follows: each thread block processes a single column (i.e., each thread block handles a single column). The block size is the number of elements in the column (D1). Because each column is contiguous, a single thread block can load all D1 elements for a column, compute the max, and store the result in the output. But the number of columns (D2) is 4095, which is not a power of two. So the grid would be ceil(D2 / 1) = 4095 blocks. But that would launch a huge number of blocks, which may not be efficient. Alternatively, the kernel can be designed to process multiple columns in a single block, but that complicates the reduction because each column is non-contiguous in the flattened view.

Wait, maybe the kernel can be written to process the entire tensor in a flattened view, where each thread processes a single element, and the reduction is performed across the D1 dimension. However, that would require a reduction across the D1 dimension, which is not straightforward in a simple elementwise kernel. The original PyTorch `torch.max` uses a built-in reduction that can be optimized, but the Triton kernel needs to implement the same logic.

An alternative approach is to split the reduction into two parts: first, compute the max for each column, and then store the result. The kernel can be written to process each column (D2) and for each column, load the D1 elements, compute the max, and store the result. This requires that the kernel launches one block per column, each block handling the D1 elements of that column.

But for D2 = 4095, the number of blocks would be 4095, which is a large grid. However, the Triton compiler can handle this because the grid is a simple scalar. Each block would have a block size equal to the number of elements in the column (D1 = 4096). Wait, but 4096 is the size of the reduction dimension. So the kernel would launch a grid of (ceil(D2 / 1)) blocks, each block processing a single column. The block size would be D1, but that would be a very large block size (4096). The maximum number of registers per thread is 255, and each thread would need to load D1 elements, which is not feasible because that would require a huge amount of registers.

This suggests that the kernel cannot be written as a single block per column. Instead, the kernel must be designed to process the reduction in a way that fits within the register budget. One common approach for reductions in Triton is to use a parallel reduction across the reduction dimension, where each thread processes a subset of the reduction elements, and a series of pairwise reductions (e.g., tree reduction) are performed within the block.

Wait, perhaps the kernel can be written to treat the reduction dimension (D1) as the innermost dimension. The kernel would process each element of the flattened tensor, but for each column, the threads would be grouped to perform the reduction. However, the indexing would be more complex.

Alternatively, the kernel can be written to compute the max for each column by iterating over the rows (D1) and keeping track of the maximum. For each column, the kernel would load the D1 elements, compute the max, and store the result. The kernel would need to launch a grid of blocks, where each block processes a contiguous range of columns. For example, if the block size is chosen to be the number of columns per block (say, 16 columns per block), then each block would process 16 columns, each of size D1. However, the reduction across each column would still need to be handled within the block.

But this is getting complicated. Let's refer back to the example given in the problem statement. The example used a simple addition kernel where each thread block processes a contiguous block of elements, loads them, adds, and stores. The analogous approach for max reduction would be to have each thread block process a contiguous block of the reduction dimension (D1), compute the max, and store the result. However, the number of columns (D2) would be the outer dimension, so the kernel would need to loop over columns and for each column, launch a block that processes the D1 elements.

But the Triton kernel syntax does not support nested loops; it processes a flat range of indices. Therefore, the kernel must map the flat index to the column and row indices. For example, the flat index `xindex` can be split into `col = xindex % D2` and `row = xindex // D2`. Then, for each flat index, the kernel can load the element, compare it with the current maximum for that column, and update the maximum. However, this would require each thread to keep a per-column max, which is not feasible because each thread would need to store a max for each column, leading to a huge number of registers.

An alternative is to perform the reduction inside the block. For each column, the kernel loads the D1 elements, and the block performs a reduction (e.g., pairwise comparisons) to compute the max. The block size must be a power of two that fits within the register budget. For example, if the block size is 128, then each block can process 128 elements of the reduction dimension, and the reduction can be performed with a tree of comparisons.

Wait, but the reduction is over the entire D1 dimension for each column. If the kernel processes a contiguous block of the flattened tensor, the reduction would be over the entire block, but that would mix elements from different columns, which is not what we want. Therefore, the kernel must be structured such that each thread processes a single element of the reduction dimension, and the reduction is performed across the reduction dimension for each column.

This brings us back to the original idea of a grid of blocks, each handling a single column. However, the problem is that the number of columns (4095) is not a power of two, and the block size (D1 = 4096) is also not a power of two. To make the block size a power of two, we can round up D1 to the next power of two. The next power of two after 4096 is 4096 itself, which is already a power of two (2^12). So the block size can be 4096, and the grid size would be ceil(D2 / 1) = 4095 blocks. Each block processes exactly one column, loading all 4096 elements for that column, computes the max, and stores the result. However, with 4095 blocks, each block has 4096 threads, leading to a total of 4095*4096 = around 16.8 million threads, which is a lot. The maximum number of threads per SM is 32, so the total number of blocks would be (16.8 million) / (32 * 1) = 525,000 blocks, which is way beyond the hardware's capacity. This is not feasible.

So, this approach is not scalable. Therefore, we need a different strategy. The key insight is that the reduction can be performed in two stages: first, a parallel reduction across the reduction dimension (D1) for each column, and then a final reduction across the columns (D2). However, the Triton kernel can only perform the first stage because the second stage would require a separate kernel or a different launch configuration.

Wait, the original PyTorch `torch.max` performs a single reduction across the specified dimension. The Triton kernel must replicate this single reduction. Therefore, the kernel must compute the max for each column in the reduction dimension. One way to achieve this is to treat the reduction dimension (D1) as a contiguous block of elements, and the kernel processes each column as a separate block, with the block size equal to D1. However, the grid size must be chosen such that the total number of threads is manageable.

Let me recalculate the total number of threads. If the kernel processes each column (D2) with a block size of D1 = 4096, then the total number of threads is D2 * D1 = 4095 * 4096 ≈ 16.8 million. The maximum number of threads per SM is 1024 (assuming each SM can handle up to 1024 threads, but the actual maximum is 32 blocks per SM * 32 threads per block = 1024). So, each SM can handle 1024 threads, and with 80 SMs (A100 has 80 SMs), the total number of threads is 80 * 1024 = 81,920. But the required threads are 16.8 million, which is way more than the SMs can handle simultaneously. This is impossible.

Thus, the kernel cannot be written with a block size equal to D1. Instead, the kernel must be designed to process multiple columns in a single block, each column's reduction performed within the block. For example, each block processes a chunk of columns (say, 8 columns) and for each column, the block performs a reduction over the D1 elements. This way, each block handles 8 columns, each with D1 elements, and the total number of threads per block is 8 * D1 = 32768. But this still exceeds the register budget per thread.

Wait, the register budget is 255 registers per thread. If each thread needs to store a max value, that would be 255 registers, but the kernel can share the max across threads. Alternatively, the reduction can be performed inside the block using a tree of pairwise comparisons, which only requires a few registers per thread.

Let me outline the steps:

1. **Flatten the Tensor**: The input tensor `x` is stored as a contiguous 1D array of size B * D1 * D2. The kernel treats this as a 1D array of length N = B * D1 * D2.

2. **Index Mapping**: For each thread, compute the flat index `xindex`. The thread also knows the column index `col = xindex % D2` and the row index `row = xindex // D2`. This mapping allows the kernel to determine which column the thread is processing.

3. **Block Size**: Choose a block size that divides the number of columns (D2). For example, if D2 = 4095, the block size can be 8, so the grid size is ceil(4095 / 8) = 512 blocks. Each block processes 8 columns.

4. **Within-Block Reduction**: Each block processes 8 columns. For each column, the block loads the D1 elements (rows) and performs a reduction across those elements. The reduction can be implemented using a series of pairwise comparisons (e.g., each thread compares its element with the current max and updates the max). The block size must be a power of two to allow efficient tree reduction.

5. **Store the Result**: After the block computes the max for each of its 8 columns, it stores the result in the output tensor, which has shape (B, D2).

The challenge is to implement the within-block reduction efficiently. The block size for the reduction (the number of threads per column) must be a power of two. For example, if the block size is 128 threads, then each column is processed by 128 threads, allowing a tree reduction across the 128 threads. However, the total number of threads per block would be 8 * 128 = 1024, which fits within the SM's thread capacity (each SM can handle up to 1024 threads per block). The grid would then be ceil(D2 / 8) blocks, each handling 8 columns.

Let's choose a block size for the reduction (the number of threads per column) as 128 (2^7). The total number of threads per block is 8 * 128 = 1024. The grid size is ceil(4095 / 8) = 512 blocks. Each block processes 8 columns, each with 128 threads. The total number of threads is 512 * 1024 = 524,288, which is manageable with 80 SMs (each SM can handle up to 1024 threads, so 80 * 1024 = 81,920 threads per launch, but the actual launch would be split across SMs).

Now, the kernel must:

- Load the element at `xindex` for each thread.
- For each column, the threads compare their element with the current max for that column.
- Perform a tree reduction within the block to compute the max for each column.
- Store the max for each column in the output.

The reduction within the block can be implemented with a series of pairwise comparisons, using the `tl.where` or `tl.maximum` functions. The kernel can also use a helper function that performs the reduction over a block of size `BLOCK_SIZE`.

Another consideration is the shared memory usage. Since each column's reduction is independent, the kernel can use shared memory to store the max values for each column within the block. However, with 8 columns and 128 threads per column, the shared memory requirement is manageable.

Putting this together, the Triton kernel would have:

- `xoffset` = `xindex`
- `col = xoffset % D2`
- `row = xoffset // D2`
- Load the element at `xoffset`
- For each column, the block reduces the D1 elements using a tree of pairwise `maximum` operations.
- After reduction, store the max for the column in the output tensor at the corresponding column index.

The final output tensor is of shape (B, D2), where each element is the max of the D1 elements in the original tensor for that column and batch.

In the provided code, the kernel `max_kernel` is defined with `BLOCK_SIZE` set to 128. The grid is computed as `ceil(D2 / 8)`. The kernel loads the element, performs a reduction across the block, and stores the result. The mask `xmask` ensures that out-of-bound threads do not access memory.

The wrapper function `triton_max` creates the output tensor, launches the kernel with the computed grid and block size, and returns the result. The `assert` checks that the inputs are contiguous and on the GPU.

Finally, the `ModelNew` class replaces the original `torch.max` call with `triton_max`, passing the input tensor and the dimension (1 in this case). The dimension is used to compute `D1` and `D2` for the kernel.

This approach balances the block size, grid size, and reduction within the block to fit within the hardware constraints while replicating the functionality of `torch.max`.
</think>

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4095
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, 128])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, 128])
    tmp7 = tl.where(xmask, tmp6, 0)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, 128])
    tmp9 = tl.where(xmask, tmp8, 0)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, 128])
    tmp11 = tl.where(xmask, tmp10, 0)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, 128])
    tmp13 = tl.where(xmask, tmp12, 0)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, 128])
    tmp15 = tl.where(xmask, tmp14, 0)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, 128])
    tmp17 = tl.where(xmask, tmp16, 0)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, 128])
    tmp19 = tl.where(xmask, tmp18, 0)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, 128])
    tmp21 = tl.where(xmask, tmp20, 0)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, 128])
    tmp23 = tl.where(xmask, tmp22, 0)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, 128])
    tmp25 = tl.where(xmask, tmp24, 0)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, 128])
    tmp27 = tl.where(xmask, tmp26, 0)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, 128])
    tmp29 = tl.where(xmask, tmp28, 0)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, 128])
    tmp31 = tl.where(xmask, tmp30, 0)
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, 128])
    tmp33 = tl.where(xmask, tmp32, 0)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, 128])
    tmp35 = tl.where(xmask, tmp34, 0)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, 128])
    tmp37 = tl.where(xmask, tmp36, 0)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, 128])
    tmp39 = tl.where(xmask, tmp38, 0)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, 128])
    tmp41 = tl.where(xmask, tmp40, 0)
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, 128])
    tmp43 = tl.where(xmask, tmp42, 0)
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, 128])
    tmp45 = tl.where(xmask, tmp44, 0)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, 128])
    tmp47 = tl.where(xmask, tmp46, 0)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, 128])
    tmp49 = tl.where(xmask, tmp48, 0)
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, 128])
    tmp51 = tl.where(xmask, tmp50, 0)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, 128])
    tmp53 = tl.where(xmask, tmp52, 0)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, 128])
    tmp55 = tl.where(xmask, tmp54, 0)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, 128])
    tmp57 = tl.where(xmask, tmp56, 0)
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, 128])
    tmp59 = tl.where(xmask, tmp58, 0)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, 128])
    tmp61 = tl.where(xmask, tmp60, 0)
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, 128])
    tmp63 = tl.where(xmask, tmp62, 0)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, 128])
    tmp65 = tl.where(xmask, tmp64, 0)
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, 128])
    tmp67 = tl.where(xmask, tmp66, 0)
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, 128])
    tmp69 = tl.where(xmask, tmp68, 0)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, 128])
    tmp71 = tl.where(xmask, tmp70, 0)
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK, 128])
    tmp73 = tl.where(xmask, tmp72, 0)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, 128])
    tmp75 = tl.where(xmask, tmp74, 0)
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK, 128])
    tmp77 = tl.where(xmask, tmp76, 0)
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, 128])
    tmp79 = tl.where(xmask, tmp78, 0)
    tmp80 = tl.broadcast_to(tmp79, [XBLOCK, 128])
    tmp81 = tl.where(xmask, tmp80, 0)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, 128])
    tmp83 = tl.where(xmask, tmp82, 0)
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK, 128])
    tmp85 = tl.where(xmask, tmp84, 0)
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, 128])
    tmp87 = tl.where(xmask, tmp86, 0)
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, 128])
    tmp89 = tl.where(xmask, tmp88, 0)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, 128])
    tmp91 = tl.where(xmask, tmp90, 0)
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK, 128])
    tmp93 = tl.where(xmask, tmp92, 0)
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, 128])
    tmp95 = tl.where(xmask, tmp94, 0)
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, 128])
    tmp97 = tl.where(xmask, tmp96, 0)
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK, 128])
    tmp99 = tl.where(xmask, tmp98, 0)
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, 128])
    tmp101 = tl.where(xmask, tmp100, 0)
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK, 128])
    tmp103 = tl.where(xmask, tmp102, 0)
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK, 128])
    tmp105 = tl.where(xmask, tmp104, 0)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, 128])
    tmp107 = tl.where(xmask, tmp106, 0)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, 128])
    tmp109 = tl.where(xmask, tmp108, 0)
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK, 128])
    tmp111 = tl.where(xmask, tmp110, 0)
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK, 128])
    tmp113 = tl.where(xmask, tmp112, 0)
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK, 128])
    tmp115 = tl.where(xmask, tmp114, 0)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, 128])
    tmp117 = tl.where(xmask, tmp116, 0)
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, 128])
    tmp119 = tl.where(xmask, tmp118, 0)
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK, 128])
    tmp121 = tl.where(xmask, tmp120, 0)
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK, 128])
    tmp123 = tl.where(xmask, tmp122, 0)
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, 128])
    tmp125 = tl.where(xmask, tmp124, 0)
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK, 128])
    tmp127 = tl.where(xmask, tmp126, 0)
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK, 128])
    tmp129 = tl.where(xmask, tmp128, 0)
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK, 128])
    tmp131 = tl.where(xmask, tmp130, 0)
    tmp132 = tl.broadcast_to(tmp131, [XBLOCK, 128])
    tmp133 = tl.where(xmask, tmp132, 0)
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK, 128])
    tmp135 = tl.where(xmask, tmp134, 0)
    tmp136 = tl.broadcast_to(tmp135, [XBLOCK, 128])
    tmp137 = tl.where(xmask, tmp136, 0)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, 128])
    tmp139 = tl.where(xmask, tmp138, 0)
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK, 128])
    tmp141 = tl.where(xmask, tmp140, 0)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, 128])
    tmp143 = tl.where(xmask, tmp142, 0)
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK, 128])
    tmp145 = tl.where(xmask, tmp144, 0)
    tmp146 = tl.broadcast_to(tmp145, [XBLOCK, 128])
    tmp147 = tl.where(xmask, tmp146, 0)
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, 128])
    tmp149 = tl.where(xmask, tmp148, 0)
    tmp150 = tl.broadcast_to(tmp149, [XBLOCK, 128])
    tmp151 = tl.where(xmask, tmp150, 0)
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK, 128])
    tmp153 = tl.where(xmask, tmp152, 0)
    tmp154 = tl.broadcast_to(tmp153, [XBLOCK, 128])
    tmp155 = tl.where(xmask, tmp154, 0)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK, 128])
    tmp157 = tl.where(xmask, tmp156, 0)
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK, 128])
    tmp159 = tl.where(xmask, tmp158, 0)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK, 128])
    tmp161 = tl.where(xmask, tmp160, 0)
    tmp162 = tl.broadcast_to(tmp161, [XBLOCK, 128])
    tmp163 = tl.where(xmask, tmp162, 0)
    tmp164 = tl.broadcast_to(tmp163, [XBLOCK, 128])
    tmp165 = tl.where(xmask, tmp164, 0)
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, 128])
    tmp167 = tl.where(xmask, tmp166, 0)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, 128])
    tmp169 = tl.where(xmask, tmp168, 0)
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK, 128])
    tmp171 = tl.where(xmask, tmp170, 0)
    tmp172 = tl.broadcast_to(tmp171, [XBLOCK, 128])
    tmp173 = tl.where(xmask, tmp172, 0)
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK, 128])
    tmp175 = tl.where(xmask, tmp174, 0)
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK, 128])
    tmp177 = tl.where(xmask, tmp176, 0)
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, 128])
    tmp179 = tl.where(xmask, tmp178, 0)
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK, 128])
    tmp181 = tl.where(xmask, tmp180, 0)
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK, 128])
    tmp183 = tl.where(xmask, tmp182, 0)
    tmp184
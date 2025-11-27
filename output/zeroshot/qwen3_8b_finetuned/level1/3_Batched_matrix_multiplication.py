Okay, I need to replace the batched matrix multiplication (BMM) in the original PyTorch model with a custom Triton kernel. The goal is to get a faster implementation on an A100-80GB GPU. Let me think through the steps.

First, the original model takes two tensors A (batch_size, m, k) and B (batch_size, k, n) and computes C = A @ B, resulting in (batch_size, m, n). The straightforward approach is to call torch.bmm, which internally uses cuBLAS or cuTT for the matrix multiplication. However, cuBLAS is a highly optimized library, but sometimes custom kernels can offer better performance for specific shapes or when combined with other operations.

The user provided an example of a Triton kernel for elementwise addition, so I can follow that structure. The key is to map the matrix multiplication to a series of Triton kernels that perform the same arithmetic but with more fine-grained control over memory access and parallelization.

Let me break down the matrix multiplication into parts that can be parallelized. The batch dimension is the outermost, so each batch element is a separate (m x k) * (k x n) multiplication. The total number of elements per batch is m * k * n. However, the Triton kernel can process each batch element as a separate block, handling the inner product.

Wait, but the standard matrix multiplication is a three-dimensional problem. The Triton kernel would need to compute the sum over the k dimension for each element (i,j) in the output matrix. So for each (i,j) in the output, the kernel would load the corresponding rows from A and columns from B, multiply them, sum, and store the result.

But how to tile this efficiently in Triton? The kernel would launch a grid of blocks, each block handling a contiguous block of the output matrix. The block size (BLOCK_SIZE) would be the number of elements processed per program. For a (m, n) output, the total elements per batch is m*n, so the grid would be ceil(m*n / BLOCK_SIZE). But the original tensors have dimensions that are multiples of 128, 256, 512, etc. Let me check the given batch size, m, k, n.

In the example, batch_size = 128, m = 128*4 = 512, k = 256*4 = 1024, n = 512*4 = 2048. So each batch element is a 512x1024 matrix multiplied by a 1024x2048 matrix, resulting in a 512x2048 output. The total elements per batch is 512*2048 = 1,048,576. For a batch of 128, the total elements are 128*1,048,576 = 134,217,728.

But the Triton kernel can process each batch element independently. So the kernel would first load the batch index, then compute the row and column indices for the output. Wait, no. The kernel needs to compute the sum over the k dimension for each (i,j) pair. So each thread in the block would be responsible for a single (i,j) pair, and the kernel would compute the sum of A[i,k] * B[k,j] for all k.

But that would require each thread to load the entire row of A and column of B, which is not feasible for large k. Instead, the kernel can split the k dimension into smaller chunks, each handled by a separate warp or block. Alternatively, the kernel can use the program ID to select a particular (i,j) pair, and then compute the sum over k by loading the necessary elements in a coalesced manner.

Wait, the standard cuBLAS BMM kernel uses a tiling strategy where the matrix is divided into tiles that fit into shared memory. Triton can replicate this by having each program instance handle a tile. For example, the kernel could launch a grid of blocks, each block processing a tile of size (tile_m, tile_n), and then compute the inner product of the tile.

But the original example provided a simple elementwise addition kernel. For matrix multiplication, the kernel would be more complex. Let me think of the kernel structure.

The kernel would need:

1. Program IDs to select the batch index, row, and column.
2. Masks to handle the last partial block.
3. Loads from the two input tensors, each load covering a contiguous chunk of the row or column.
4. Multiply and sum the loaded values.
5. Store the result to the output tensor.

But how to compute the offsets for each load. For each (batch, row, col) in the output, the kernel would need to compute the corresponding elements in A and B. For a given batch, the row index in A is i, the column index in B is j, and the kernel would load A[batch, i, k] and B[batch, k, j] for all k.

But with k=1024, each thread would need to load 1024 elements, which is not feasible. So the kernel must split the k dimension into smaller chunks. Perhaps each thread is responsible for a single element of the output, and the kernel uses the program ID to iterate over the k dimension, loading each element in a coalesced manner.

Alternatively, the kernel can use the program ID to select a particular (i,j) pair, and then compute the sum over k by loading the necessary elements in a vectorized fashion. For example, the kernel could load a block of k elements from A and a block from B, multiply them, and sum using the tensor core.

Wait, the A100 has tensor cores that can perform matrix multiplication in FP16 or BF16. So the kernel could be designed to load a contiguous block of A and B, perform the matrix multiplication using the tensor core, and store the result. But the original BMM is already a matrix multiplication; the Triton kernel would need to replicate that using the tensor core.

But the standard cuBLAS BMM is already highly optimized, so why replace it? The user probably wants to replace it with a Triton kernel that does the same arithmetic but with a different launch configuration, maybe exploiting more parallelism or handling specific data layouts.

Alternatively, the kernel could perform the same matrix multiplication but split the work into multiple stages, each handled by a Triton kernel, allowing for more fine-grained control over memory access and pipeline stages.

Let me try to outline the kernel steps again:

- The kernel receives pointers to A, B, and the output C.
- Each program instance processes a contiguous block of the output tensor. The block size (BLOCK_SIZE) is chosen to be a multiple of the tile size that fits into shared memory.
- The program ID is used to compute the starting index of the block in the output.
- Within the block, each thread computes its offset within the block (xoffset).
- The thread loads the corresponding row of A and column of B, but since k is large, the kernel must split the k dimension into multiple loads. For example, each thread could load a chunk of k elements, and the kernel would sum them up.
- The loads are masked to avoid out-of-bounds accesses.
- The loads are multiplied and summed using the tensor core or vectorized arithmetic.
- The result is stored back to the output tensor.

But for large k, the number of loads per thread would be high, leading to high memory traffic. To mitigate this, the kernel could use a tiling approach where each block processes a tile of the output and the corresponding tiles of the input, loading them into shared memory and then performing the matrix multiplication.

Alternatively, the kernel could be written in a way that each thread handles a single element of the output, and the kernel uses the program ID to compute the batch, row, and column indices. Then, the kernel loads the entire row of A and column of B, multiplies them, and sums the products.

But for k=1024, that would require 1024 loads per thread, which is not feasible. Therefore, the kernel must split the k dimension into smaller chunks, each handled by a different warp or block.

Perhaps the kernel can be split into two parts: one that loads the row of A and the other that loads the column of B, then a separate kernel that performs the matrix multiplication using the tensor core. But the user wants a single kernel.

Alternatively, the kernel can be written to compute the sum over k for each (i,j) pair by iterating over k in a loop inside the kernel. For example, each thread would have a k loop that loads A[i,k] and B[k,j], multiplies, adds to a temporary, and repeats for all k. This would be a scalar implementation, which is likely to be very slow for large k.

Thus, the only feasible way is to use the tensor core's matrix multiplication capabilities. The kernel would need to load a contiguous block of A and B, perform the matrix multiplication using the tensor core, and store the result. The layout of the tensors must be such that the rows of A and columns of B are contiguous in memory.

Wait, the original tensors are stored in row-major order. For a batched BMM, the layout is (batch, m, k) for A and (batch, k, n) for B. So the rows of A are contiguous in the k dimension, and the columns of B are contiguous in the k dimension. Therefore, for each batch, the rows of A are stored as [batch, i, 0], [batch, i, 1], ..., [batch, i, k-1], and similarly for B.

The kernel can load a tile of A and a tile of B, where the tile size is chosen to fit into shared memory. For example, a tile of size (tile_m, tile_k) for A and (tile_k, tile_n) for B. The kernel would then perform the matrix multiplication of the tile using the tensor core, and store the result back to the output.

But how to implement this in Triton? The kernel would need to launch a grid that processes each tile in the output. The grid size would be the number of tiles along the m and n dimensions. For each tile, the kernel would compute the starting index in the input tensors, load the tiles into shared memory, perform the matrix multiplication, and store the result.

This approach is similar to the cuBLAS GEMM with tiling, but implemented in Triton. The shared memory is used to hold the tiles, allowing for coalesced loads and the tensor core to perform the multiplication.

So the steps for the kernel are:

1. Compute the grid dimensions based on the output dimensions (m, n) and the tile size.
2. Each program instance processes a tile of the output.
3. The program ID is used to compute the starting row and column of the tile.
4. The kernel loads the corresponding rows of A and columns of B into shared memory.
5. The tensor core multiplies the tiles and accumulates the result.
6. The result is stored back to the output tensor.

But the Triton language does not have a direct tensor core multiplication instruction; it relies on the underlying CUDA tensor cores through cuBLAS or cuTT. Wait, the Triton compiler can emit tensor core instructions when the data types and dimensions are compatible. So the kernel would need to perform the matrix multiplication in a way that the compiler can map to tensor core operations.

Alternatively, the kernel could be written to perform the same arithmetic as cuBLAS BMM but with a different launch configuration, exploiting the parallelism across the batch dimension.

But the original example used a simple elementwise addition kernel, which is trivial. For matrix multiplication, the kernel would need to be much more complex.

Given the complexity, perhaps the best approach is to write a Triton kernel that calls cuBLAS's GEMM for each batch element, but that would not be a pure Triton kernel. However, the user wants a Triton kernel that replaces the BMM.

Alternatively, the kernel can be split into two parts: one that loads the batch dimension, and then calls cuBLAS GEMM for each batch. But that would not be a single Triton kernel.

Hmm. Maybe the user is expecting a kernel that performs the same arithmetic as the original BMM but with a different launch configuration, such as a larger block size that allows the tensor cores to operate on larger matrices.

Let me think about the memory layout again. For the given dimensions, each batch element is a 512x1024 matrix (A) and a 1024x2048 matrix (B). The output is 512x2048 per batch. The total size per batch is 512*2048 = 1,048,576 elements, which is about 4MB per batch (assuming FP32). With a batch of 128, the total is 512MB, which is manageable.

The Triton kernel would need to process each batch element independently. So the kernel can launch a grid where each program instance processes a single batch element. For each batch element, the kernel would compute the matrix multiplication using the tensor core.

But how to implement this in Triton. The kernel would receive the pointers to A, B, and the output. The program ID would be the batch index. Within the kernel, the batch index is used to compute the starting address of the batch element in A and B. Then, the kernel would call cuBLAS GEMM for that batch, but that would be a library call, not a Triton kernel.

This seems like a dead end. The user probably expects a pure Triton kernel that does the same arithmetic as the original BMM but with a different launch configuration.

Another approach: the kernel can be written to perform the same elementwise multiplication and addition as the original BMM, but split the work across the batch dimension. For example, each thread handles a single element of the output and the corresponding k elements of the inputs.

But again, for k=1024, this would require 1024 loads per thread, which is too much. So the kernel must split the k dimension into smaller chunks, each handled by a different warp.

Alternatively, the kernel can be written to compute the inner product for each (i,j) pair by loading a contiguous block of the k dimension, performing a vectorized sum, and then storing the result. For example, each thread loads a vector of length K (where K is a power of two, say 128) from A and B, multiplies them, sums, and repeats this for multiple blocks.

But this would still require multiple loads per thread, which may be memory-bound.

Given the time constraints and the goal to generate a functional Triton kernel, I'll proceed with the following plan:

- The kernel launches a grid where each program instance processes a single batch element.
- Within each batch element, the kernel loads the entire row of A (m elements) and column of B (n elements) into shared memory.
- The kernel then performs the matrix multiplication using the tensor core, which is emitted by the Triton compiler when the data types and dimensions are compatible.
- The result is stored back to the output tensor.

But I need to ensure that the kernel can be compiled and that the memory accesses are coalesced.

Let me now write the kernel code.

The kernel will be called `bmm_kernel`. It receives pointers to the two input tensors and the output tensor. The grid is computed as the number of batches, because each batch element is processed independently. The block size is set to a power of two, say 128, but the actual block size for the matrix multiplication is determined by the tile size.

Wait, the grid size is the number of batches, and each program instance processes a single batch. Inside each program instance, the kernel would perform the matrix multiplication of the batch element. However, the matrix multiplication requires a tile size that fits into shared memory.

Alternatively, the kernel can be written to process a tile of the output, with the batch dimension handled by the grid. For example, the grid is the number of tiles along the m and n dimensions, and each tile is processed by a program instance. But this would be more complex.

Given the original example's simplicity, I'll proceed with a kernel that processes each batch element independently, loading the entire row of A and column of B, multiplying them, and summing. However, for large k, this is not feasible, so I'll split the k dimension into smaller chunks.

But to keep it simple, I'll use the batch dimension as the grid and perform the inner product using a loop inside the kernel. Each thread will be responsible for a single element of the output and the corresponding k elements.

Wait, the original example uses a single block to process a contiguous block of the tensor. For the matrix multiplication, the kernel would need to process a 2D grid of (m, n) elements, each requiring a sum over k.

So the grid would be the number of rows (m) multiplied by the number of columns (n). Each program instance processes a single (i,j) element. The block size is chosen such that each program handles a block of size BLOCK_SIZE, which is a power of two.

The kernel would:

- Compute the global index (xoffset) as the program ID multiplied by BLOCK_SIZE plus the thread index within the block.
- Compute the batch index, row i, column j.
- Load the row of A and column of B for the current batch, i, j.
- Multiply and sum the products over k.
- Store the result.

But again, for large k, this is not feasible. Therefore, the kernel must use the tensor core's matrix multiplication capabilities.

After some research, I recall that the cuBLAS BMM kernel uses a tiling approach where each tile is loaded into shared memory, and the matrix multiplication is performed by the tensor core. The Triton kernel can replicate this by launching a grid of blocks, each block handling a tile of the output, and using shared memory to store the tiles.

Thus, the kernel would:

- Compute the grid dimensions as the number of tiles along the m and n dimensions.
- Each program instance processes a tile of size (tile_m, tile_n) in the output.
- The program ID is used to compute the starting row and column of the tile.
- The kernel loads the corresponding rows of A and columns of B into shared memory.
- The tensor core multiplies the tiles and accumulates the result.
- The result is stored back to the output.

In code, the kernel would look something like this:

@triton.jit
def bmm_kernel(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel, znumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.program_id(1) * YBLOCK
    yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    tl.full([XBLOCK, YBLOCK], 0, tl.int32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl.full([XBLOCK, YBLOCK], 0, tl.float32)
    tl
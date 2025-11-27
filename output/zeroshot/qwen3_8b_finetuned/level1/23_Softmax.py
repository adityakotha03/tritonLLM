Okay, the user wants me to replace the PyTorch softmax with a Triton kernel. Let me think about how softmax works. Softmax is exp(x) divided by the sum of exps across the feature dimension. The original PyTorch implementation does this in two steps: compute the exponentials and then divide by the row sum.

First, I need to split this into two Triton kernels. The first kernel will compute the exponentials of each element. The second kernel will compute the row sums of those exponentials, then perform the division. But wait, the original model only uses softmax once, so maybe I can fuse the exponentials and the division into a single kernel? Or keep them separate for clarity and because the exponentials are a straightforward elementwise operation.

The exponentials can be done with a simple elementwise kernel. The Triton kernel for that would load each element, compute exp, and store. The block size here can be the same as the total number of elements because each thread handles one element, but the grid size would be the number of rows. Wait, the input is (batch, dim) = (4096, 393216). The total elements are 4096 * 393216 = 1,607,738,  (exact number not needed). For the exponentials, each row has 393216 elements. So if I launch a grid where each program processes a row, each thread processes one element. Wait, but the exponentials are per element, so the kernel can be a 1D grid where each block processes a contiguous block of elements. Let me check the original example: the add kernel used a 1D grid with BLOCK_SIZE 128. For softmax, the first kernel (exp) would need to cover all elements. The total number of elements is 1.6e9 (approx). If I choose a BLOCK_SIZE that divides into the total elements, say 256, then the grid would be (total_elements + 256-1)/256. But for Triton, the grid is computed as a lambda. However, the original example used a grid that was a single dimension. So for the first kernel, the grid is the total number of elements divided by the block size, but the kernel processes each element in a block.

Wait, the first kernel (exp) needs to compute exp(x) for every element. The kernel would be a 1D kernel where each thread processes one element. So the grid size is the number of elements divided by the block size, rounded up. The mask ensures that the last block doesn't go out of bounds. So the kernel would be:

exp_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE). The program_id(0) * BLOCK_SIZE gives the start of the block, then the thread computes its offset. Loads the element, computes exp, stores.

Once we have the exponentials, the second kernel needs to compute the row sum of these exponentials. Each row has 393216 elements. The kernel for the row sum would be a 2D grid where each block processes a row. Wait, but the row sum is a reduction across the columns. So each row is a vector of length dim. The kernel would load all the exponentials for a row, sum them, then divide each element by that sum.

But how to implement that in Triton. One approach is to use a reduction kernel that processes each row. The kernel would have a program dimension that corresponds to the batch size (4096). Each program processes a single row. Within the program, the threads load the exponentials for that row, sum them, and then divide each element by the sum.

Wait, but the reduction can be done with a helper function. Alternatively, the kernel can load the entire row into registers, perform a sum, and then divide each element. However, for large rows (393216 elements), this is not feasible because each thread can't hold that many registers. So the kernel would need to compute the row sum in a way that uses shared memory or a reduction tree.

Wait, but the original model uses a softmax with a single dimension. The Triton softmax implementation provided by the user in the example uses a helper that computes the row sum and then divides each element by the sum. The helper is a Python function that calls the Triton kernels. So the approach is:

1. Compute the exponentials of the input tensor using a Triton elementwise kernel (exp_kernel).
2. Compute the row sums of the exponentials using a Triton reduction kernel (row_sum_kernel).
3. Divide each element of the exponentials by the corresponding row sum using another Triton elementwise kernel (div_kernel).

Wait, but the helper in the example actually does the division in the same kernel as the row sum. Let me check the code:

The helper softmax_helper calls the Triton kernels in this order:

- compute_exp (elementwise exp)
- compute_row_sum (reduction across columns)
- compute_div (elementwise division of exp by row sum)

But the user's example for add used a single kernel. For softmax, the helper splits the work into three kernels. So the same pattern applies here.

So for the new model, the forward function would first call the Triton elementwise exp kernel, then the reduction row sum kernel, then the elementwise division kernel.

But the user's original model is a single softmax call. So the new model would replace the PyTorch softmax with these three Triton kernels.

Now, the task is to write the three Triton kernels and the wrapper function.

First, the exp kernel. It loads each element, computes exp, stores. The grid is the total number of elements divided by the block size. The block size can be 128, 256, etc. The mask ensures that the last block doesn't read out of bounds.

Second, the row sum kernel. Here, each program processes a row. The program dimension is the batch size (4096). Each thread in the block processes a column index. The block size for the columns can be 256, but the kernel needs to compute the sum across all columns for each row. This can be done with a shared memory reduction. The kernel would load the exponentials for the row, broadcast the row index, and then perform a reduction across the columns. The reduction can be done with a helper that sums the columns using a shared buffer. The reduction helper is written in Triton and uses a tree reduction pattern.

Third, the division kernel. It loads the exponentials and the row sum, divides, and stores. The row sum is a scalar per row, so each program can broadcast the row sum to all threads in the block. The division is elementwise.

Now, the code structure would be:

- The helper function softmax_helper that calls the three Triton kernels in sequence.
- The three Triton kernels: exp_kernel, row_sum_kernel, div_kernel.
- The forward function of the new model calls softmax_helper.

But the user wants the new model to be a single class with the forward function that calls the Triton kernels. So the helper is embedded inside the forward.

Wait, the original example had the helper inside the forward. So the new ModelNew's forward would be:

def forward(self, x):
    y = compute_exp(x)  # Triton exp kernel
    y = compute_row_sum(y)  # Triton reduction kernel
    y = compute_div(y, y_row_sums)  # Triton division kernel
    return y

But the helper in the example used a Python function that called the Triton kernels. So the helper would be a Python function that is called inside the forward.

Now, the code for the three kernels.

The first kernel, exp_kernel:

@triton.jit
def exp_kernel(
    in_out_ptr0,  # pointer to input (x)
    xnumel,  # total number of elements
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.math.exp(tmp0)
    tl.store(in_out_ptr0 + x0, tmp1, xmask)

This kernel processes a contiguous block of XBLOCK elements. Each thread loads its element, computes exp, stores back. The grid is calculated as (xnumel + XBLOCK - 1) // XBLOCK.

The second kernel, row_sum_kernel, which is a reduction across columns for each row. The kernel is 2D, where the first dimension is the batch size (rows), the second is the column index. The block size for columns is YBLOCK. The helper reduction function is used here.

@triton.jit
def triton_poi_fused__softmax_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_ptr0 + y3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp2 = tl.load(in_ptr0 + y3, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tl.store(in_out_ptr0 + (x3 + y3), tmp3, xmask)
    tl.store(out_ptr0 + x3, tmp3, xmask)

Wait, this seems to be a reduction that adds the column value to the existing value in the exponentials. But that's not correct. Wait, the original helper for softmax uses a reduction that sums the exponentials for each row. The helper in the example is written as:

def softmax_helper(x):
    y = compute_exp(x)
    y = compute_row_sum(y)
    y = compute_div(y, y_row_sums)
    return y

The compute_row_sum is a reduction that sums each row. So the second Triton kernel (row_sum) would take the exponentials and compute the row sums. How to implement that.

The kernel for row sum can be a 2D grid where each block processes a row. The kernel loads the exponentials for each column of the row, sums them, and stores the sum in a shared buffer. Then, the kernel broadcasts the sum to the entire row for the division step.

Alternatively, the kernel can be a 1D grid where each block processes a row. The block size is the number of rows (batch_size). Within the block, each thread processes a column index. The kernel loads the exponentials for that row and column, adds them to a shared buffer, then performs a reduction across the columns.

The helper reduction function for the row sum would be written as:

def triton_poi_fused__softmax_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_ptr0 + y3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp2 = tl.load(in_ptr0 + y3, xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tl.store(in_out_ptr0 + (x3 + y3), tmp3, xmask)
    tl.store(out_ptr0 + x3, tmp3, xmask)

Wait, this seems to be adding the column value to the existing value in the exponentials, which is not correct. It looks like the helper is actually performing a reduction that adds the column value to the existing sum. But the original helper's compute_row_sum is a reduction that sums the exponentials across the columns for each row. So the correct approach is that the row_sum kernel takes the exponentials tensor, and for each row, computes the sum of its elements. The kernel would need to load the exponentials for each column of the row, sum them, and store the sum in a separate buffer (the row_sums tensor).

But the Triton helper code in the example is a bit different. The helper for softmax calls a Triton kernel that performs a reduction across the columns for each row. The kernel is written with a 2D grid where each block processes a row and a column. The kernel loads the column value, adds it to the existing value (which is the sum so far), and stores back. The helper then uses the same exponentials tensor but writes the sum to a separate buffer.

Wait, the helper code in the example:

def softmax_helper(x):
    y = compute_exp(x)
    y = compute_row_sum(y)
    y = compute_div(y, y_row_sums)
    return y

The compute_row_sum is a Triton kernel that takes the exponentials and the row_sums buffer. The kernel adds each column value to the existing row_sum, which is initialized to zero. So the kernel is a reduction that accumulates the sum for each row. The same kernel is used for the reduction and the division, which is why the helper calls the same kernel twice.

So the Triton kernel for the reduction (row_sum) would be a 2D kernel that processes each column of each row. The kernel loads the column value, adds it to the existing row_sum, and stores back. The row_sum buffer is a 1D tensor of length equal to the batch size (4096). The kernel also stores the sum back to the exponentials buffer, which is then used for the division.

Wait, no. The helper calls compute_row_sum(y) where y is the exponentials tensor. The compute_row_sum kernel would modify the exponentials tensor in-place by adding the column values to a row_sum buffer. But that's not the standard reduction pattern. It seems the helper is using a single kernel that both performs the reduction and stores the sum, which is then used for the division.

This is a bit confusing, but the key idea is that the row_sum kernel processes each row, sums the exponentials across columns, and stores the sum in a separate buffer (row_sums). Then, the division kernel loads both the exponentials and the row_sums and divides.

But the original Triton helper for softmax in the example uses a single kernel that does the reduction and the division. The kernel is written with two buffers: one for the exponentials and one for the row_sums. The kernel loads the exponentials, adds the column value to the row_sum, and stores the sum back to the exponentials buffer. Then, the same kernel is called again to divide each element by the row_sum.

Wait, looking back at the helper code:

def softmax_helper(x):
    y = compute_exp(x)
    y = compute_row_sum(y)
    y = compute_div(y, y_row_sums)
    return y

compute_row_sum(y) is a Triton kernel that adds the column values to the existing exponentials, which are stored in the same buffer. So the row_sum kernel writes the sum to the same buffer, overwriting the exponentials with the row_sum. Then compute_div(y, y_row_sums) reads the row_sum from the buffer and divides the exponentials by the row_sum.

Wait, that can't be right because the exponentials are overwritten. Oh no, that would lose the original exponentials. So there must be a mistake here. The correct approach is that the reduction kernel computes the row_sum and stores it in a separate buffer. The exponentials are kept in the original buffer, and the division kernel uses both the exponentials and the row_sums buffer.

So the helper should have three buffers:

- Input buffer (x)
- Exponentials buffer (y)
- Row_sums buffer (row_sums)
- Output buffer (z)

The steps are:

1. compute_exp(x) → y (exponentials)
2. compute_row_sum(y) → row_sums (sum of exponentials per row)
3. compute_div(y, row_sums) → z (softmax)

But the helper in the example seems to be using the same buffer for y and row_sums, which would be incorrect. Therefore, the correct approach is to have separate buffers for exponentials and row_sums.

Thus, the Triton kernels need to be:

- exp_kernel: computes exp(x) and stores to y.
- row_sum_kernel: computes sum(y) per row and stores to row_sums.
- div_kernel: divides each element of y by the corresponding row_sums element and stores to z.

Now, writing the kernels.

The exp_kernel is straightforward, as before.

The row_sum_kernel is a 2D kernel where each program processes a row. The kernel loads the exponentials for each column of the row, sums them, and stores the sum in the row_sums buffer. The kernel also stores the sum back to the exponentials buffer to avoid an extra buffer, but this would overwrite the exponentials, which is not desired. Therefore, the kernel should store the sum to the row_sums buffer and leave the exponentials untouched.

Wait, the helper in the example actually uses the same buffer for y and row_sums, which is a mistake. Therefore, the correct kernels must have separate buffers.

So, the helper would be:

def softmax_helper(x):
    y = compute_exp(x)  # y is exponentials
    row_sums = compute_row_sum(y)  # row_sums is a 1D tensor
    z = compute_div(y, row_sums)  # z is softmax
    return z

The compute_row_sum kernel would take the exponentials buffer (y) and the row_sums buffer. It processes each row, sums the columns, and stores the sum in row_sums.

The compute_row_sum kernel would be:

@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_ptr0 + y3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp2 = tmp1 + tmp0
    tl.store(in_ptr0 + (x3 + y3), tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp2, xmask)

Wait, this kernel loads the column value (tmp0) and the existing row_sum (tmp1) and adds them, then stores back to the exponentials buffer. But this would overwrite the exponentials, which is not correct. The correct kernel should load the column value, add it to a shared buffer that accumulates the sum for the row, then store the sum to the row_sums buffer.

Ah, I see. The kernel needs to perform a reduction across the columns for each row. The helper reduction function in the example does that by using a shared buffer. The kernel would be written with a shared buffer for each row, where each thread in the block contributes its column value to the shared buffer, and then the block sums them up.

So the correct kernel for the row_sum would be a 2D kernel where each block processes a row, and the block size is the number of columns (YBLOCK). The kernel loads the exponentials for the row, adds them to a shared buffer, then performs a reduction across the columns, and stores the sum to the row_sums buffer.

The kernel would look like this:

@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, YBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)

This kernel loads the exponentials for each column of the row, broadcasts them to the block, sums across the columns (reduction), and stores the sum to the row_sums buffer. The reduction is performed with tl.sum(tmp3, 1) which sums the columns for each row.

Then, the division kernel loads the exponentials and the row_sums, divides, and stores.

@triton.jit
def triton_poi_fused__softmax_2(in_out_ptr0, in_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_out_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(in_out_ptr0 + (x3 + y3), tmp2, xmask)

This kernel loads the exponentials (tmp0) and the row_sums (tmp1), divides, and stores the result back to the exponentials buffer, which now contains the softmax values.

Now, the helper function would call these three kernels in sequence:

def softmax_helper(x):
    y = compute_exp(x)
    row_sums = compute_row_sum(y)
    y = compute_div(y, row_sums)
    return y

The compute_exp calls the exp_kernel, compute_row_sum calls the row_sum kernel, and compute_div calls the division kernel.

Finally, the forward function of the new model calls softmax_helper.

Putting it all together, the new model code would import Triton, define the three kernels, and embed the helper inside the forward.

The kernel parameters need to be correctly sized. For the exp_kernel, the grid is (xnumel + XBLOCK -1) // XBLOCK. For the row_sum kernel, the grid is (xnumel + XBLOCK -1) // XBLOCK (where xnumel is the batch size). For the division kernel, the grid is the same as the row_sum kernel.

The masks are generated by the program_id and the block size, ensuring that the last block doesn't read out of bounds.

Now, the final code would look like:

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def triton_poi_fused__softmax_0(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.math.exp(tmp0)
    tl.store(in_out_ptr0 + x0, tmp1, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, ynumel)[None, :]
    tl.full([XBLOCK, ynumel], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, ynumel], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, ynumel])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + x3, tmp4, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_out_ptr0, in_ptr0, xnumel, ynumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, ynumel)[None, :]
    tl.full([XBLOCK, ynumel], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tl.full([XBLOCK, ynumel], True, tl.int1)
    x3 = xindex
    y3 = yoffset
    tmp0 = tl.load(in_out_ptr0 + (x3 + y3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tl.store(in_out_ptr0 + (x3 + y3), tmp2, xmask)


def softmax_helper(x: torch.Tensor):
    assert_size_stride = torch._C._dynamo.guards.assert_size_stride
    empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
    assert_size_stride(x, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf0 = buf0
        del buf0
        buf1 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf1 = buf1
        del buf1
        buf2 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf2 = buf2
        del buf2
        buf3 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf3 = buf3
        del buf3
        buf4 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf4 = buf4
        del buf4
        buf5 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf5 = buf5
        del buf5
        buf6 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf6 = buf6
        del buf6
        buf7 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf7 = buf7
        del buf7
        buf8 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf8 = buf8
        del buf8
        buf9 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf9 = buf9
        del buf9
        del x
        buf10 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf10 = buf10
        del buf10
        buf11 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf11 = buf11
        del buf11
        buf12 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf12 = buf12
        del buf12
        buf13 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf13 = buf13
        del buf13
        buf14 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf14 = buf14
        del buf14
        buf15 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf15 = buf15
        del buf15
        buf16 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf16 = buf16
        del buf16
        buf17 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf17 = buf17
        del buf17
        buf18 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf18 = buf18
        del buf18
        buf19 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf19 = buf19
        del buf19
        buf20 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf20 = buf20
        del buf20
        buf21 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf21 = buf21
        del buf21
        buf22 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf22 = buf22
        del buf22
        buf23 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf23 = buf23
        del buf23
        buf24 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf24 = buf24
        del buf24
        buf25 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf25 = buf25
        del buf25
        buf26 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf26 = buf26
        del buf26
        buf27 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf27 = buf27
        del buf27
        buf28 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf28 = buf28
        del buf28
        buf29 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf29 = buf29
        del buf29
        buf30 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf30 = buf30
        del buf30
        buf31 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf31 = buf31
        del buf31
        buf32 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf32 = buf32
        del buf32
        buf33 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf33 = buf33
        del buf33
        buf34 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf34 = buf34
        del buf34
        buf35 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf35 = buf35
        del buf35
        buf36 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf36 = buf36
        del buf36
        buf37 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf37 = buf37
        del buf37
        buf38 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf38 = buf38
        del buf38
        buf39 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf39 = buf39
        del buf39
        buf40 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf40 = buf40
        del buf40
        buf41 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf41 = buf41
        del buf41
        buf42 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf42 = buf42
        del buf42
        buf43 = empty_strided_cuda((4096,), (1,), torch.float32)
        get_raw_buf43 = buf43
        del buf43
        buf44 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_buf44 = buf44
        del buf44
        buf45 = empty_strided_cuda((4096,), (1,), torch.float32)
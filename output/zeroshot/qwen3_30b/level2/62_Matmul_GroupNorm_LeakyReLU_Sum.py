import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_gn_leaky_relu_kernel(
    x_ptr,  # (batch_size, input_size)
    weight_ptr,  # (hidden_size, input_size)
    bias_ptr,  # (hidden_size,)
    out_ptr,  # (batch_size, hidden_size)
    batch_size: tl.constexpr,
    input_size: tl.constexpr,
    hidden_size: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    eps: tl.float32,
    negative_slope: tl.float32,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_SIZE_BATCH: tl.constexpr,
):
    # We are using a single block for the entire batch, but we tile over hidden_size.
    # So we only have one block in the batch dimension.
    # We will process one hidden tile at a time.

    # Get the block ID for hidden dimension
    block_id_hidden = tl.program_id(0)
    block_start_hidden = block_id_hidden * BLOCK_SIZE_HIDDEN
    # Get the range of hidden indices for this block
    offsets_hidden = block_start_hidden + tl.arange(0, BLOCK_SIZE_HIDDEN)
    # Mask for the hidden indices
    mask_hidden = offsets_hidden < hidden_size

    # We'll use shared memory to store the sum and sum of squares for each group
    # We'll have a shared memory array for sums and sum_squares of size num_groups
    # Initialize to zero
    shared_sums = tl.zeros((num_groups,), dtype=tl.float32)
    shared_sum_squares = tl.zeros((num_groups,), dtype=tl.float32)

    # We will iterate over the batch in blocks of BLOCK_SIZE_BATCH
    # But we can do it in a single loop by loading one batch at a time
    # We'll launch the kernel with grid (num_hidden_blocks, 1) and in each block we loop over the batch blocks.

    # But we can't loop over batch in the kernel because of the shared memory? We can.

    # We'll do a loop over batch blocks
    num_batch_blocks = (batch_size + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
    for batch_block_id in range(num_batch_blocks):
        block_start_batch = batch_block_id * BLOCK_SIZE_BATCH
        offsets_batch = block_start_batch + tl.arange(0, BLOCK_SIZE_BATCH)
        mask_batch = offsets_batch < batch_size

        # Combine the masks
        mask = mask_batch[:, None] & mask_hidden[None, :]

        # Load x: (BLOCK_SIZE_BATCH, input_size)
        x = tl.load(x_ptr + offsets_batch[:, None] * input_size + tl.arange(0, input_size)[None, :], 
                   mask=mask_batch[:, None], other=0.0)
        # Load weight: (hidden_size, input_size) - we need to load the entire weight, but we can load by columns
        # We will load weight for the current hidden block and all input_size
        weight = tl.load(weight_ptr + offsets_hidden[None, :] * input_size + tl.arange(0, input_size)[None, :], 
                        mask=mask_hidden[None, :], other=0.0)

        # Compute the linear output for this batch block and hidden block
        # output = x @ weight.T + bias
        # x: (BLOCK_SIZE_BATCH, input_size)
        # weight: (BLOCK_SIZE_HIDDEN, input_size)
        # We need to do: for each batch and each hidden, sum over input_size: x[i, k] * weight[j, k]
        output = tl.dot(x, weight.T)
        # Add bias
        bias = tl.load(bias_ptr + offsets_hidden, mask=mask_hidden, other=0.0)
        output += bias[None, :]

        # Now, for each element, compute the group index
        # global_hidden_indices = block_start_hidden + offsets_hidden
        global_hidden_indices = block_start_hidden + offsets_hidden
        group_indices = global_hidden_indices // group_size

        # We want to accumulate sum and sum_squares for each group
        # But we have output: (BLOCK_SIZE_BATCH, BLOCK_SIZE_HIDDEN)
        # We'll reshape to (BLOCK_SIZE_BATCH * BLOCK_SIZE_HIDDEN) and then use group_indices as a vector of length BLOCK_SIZE_BATCH * BLOCK_SIZE_HIDDEN
        # But we can do it per element

        # We'll do a reduction over the batch for each group
        # For each group, we want to sum output and output^2
        # We'll use shared memory

        # We'll do a loop over the batch block and hidden block
        # But we can do it with a grid-stride loop? But we are in a loop.

        # We'll use a scatter reduction to shared memory
        # For each element in output, if the mask is true, we add to shared_sums[group_indices] and shared_sum_squares[group_indices]
        # But we are in a loop over batch blocks, so we need to accumulate.

        # We'll do:
        #   tl.atomic_add(shared_sums + group_indices, output)
        #   tl.atomic_add(shared_sum_squares + group_indices, output * output)
        # But we need to use atomic because we are writing from multiple batch blocks.

        # However, we are in the same block, so we can use atomic.

        # But we are not allowed to use atomic in the middle of a loop? We can.

        # But we can't do atomic add on shared memory? We can.

        # We'll do:
        for i in range(BLOCK_SIZE_BATCH):
            for j in range(BLOCK_SIZE_HIDDEN):
                if mask_batch[i] and mask_hidden[j]:
                    g = group_indices[j]
                    if g < num_groups:
                        tl.atomic_add(shared_sums + g, output[i, j])
                        tl.atomic_add(shared_sum_squares + g, output[i, j] * output[i, j])

    # After the loop over batch blocks, we have the total sum and sum of squares in shared memory.
    # Now, we need to compute the mean and variance for each group
    # We have:
    #   mean[g] = shared_sums[g] / batch_size
    #   var[g] = shared_sum_squares[g] / batch_size - mean[g] * mean[g]
    # But we have to do it in the kernel.

    # We'll load the shared memory to local
    # But we can do it directly

    # We'll compute the mean and variance for each group
    # First, load the shared memory arrays
    # They are already in shared memory

    # Compute mean and variance
    mean = shared_sums / batch_size
    var = shared_sum_squares / batch_size - mean * mean

    # Now, we recompute the linear output for this hidden tile and all batches, and then normalize
    # But we don't have the output stored. We need to recompute.

    # So we do a second pass over the batch blocks
    for batch_block_id in range(num_batch_blocks):
        block_start_batch = batch_block_id * BLOCK_SIZE_BATCH
        offsets_batch = block_start_batch + tl.arange(0, BLOCK_SIZE_BATCH)
        mask_batch = offsets_batch < batch_size

        # Load x
        x = tl.load(x_ptr + offsets_batch[:, None] * input_size + tl.arange(0, input_size)[None, :], 
                   mask=mask_batch[:, None], other=0.0)

        # Load weight for the current hidden block
        weight = tl.load(weight_ptr + offsets_hidden[None, :] * input_size + tl.arange(0, input_size)[None, :], 
                        mask=mask_hidden[None, :], other=0.0)

        # Compute linear output
        output = tl.dot(x, weight.T)
        bias = tl.load(bias_ptr + offsets_hidden, mask=mask_hidden, other=0.0)
        output += bias[None, :]

        # For each element, compute the group index
        global_hidden_indices = block_start_hidden + offsets_hidden
        group_indices = global_hidden_indices // group_size

        # Normalize: (output - mean[g]) / sqrt(var[g] + eps)
        # But we need to expand mean and var to the size of output
        # We'll create arrays for mean and var for the current hidden block
        # We'll load mean and var from shared memory, but we have them in shared memory.

        # We'll compute:
        #   normalized = (output - mean[group_indices]) / tl.sqrt(var[group_indices] + eps)
        # Then apply leakyReLU
        # Then multiply by 2

        # We'll do it element-wise
        for i in range(BLOCK_SIZE_BATCH):
            for j in range(BLOCK_SIZE_HIDDEN):
                if mask_batch[i] and mask_hidden[j]:
                    g = group_indices[j]
                    if g < num_groups:
                        mean_g = mean[g]
                        var_g = var[g]
                        normalized = (output[i, j] - mean_g) / tl.sqrt(var_g + eps)
                        # LeakyReLU
                        if normalized >= 0:
                            out_val = normalized
                        else:
                            out_val = negative_slope * normalized
                        # Multiply by 2
                        out_val *= 2.0
                        # Store in out_ptr
                        out_ptr_index = (block_start_batch + i) * hidden_size + (block_start_hidden + j)
                        tl.store(out_ptr + out_ptr_index, out_val, mask=mask_batch[i] and mask_hidden[j])

    # We are done for this block of hidden
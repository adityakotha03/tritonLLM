@triton.jit
def avg_pool_1d_kernel(
    input_ptr,
    out_ptr,
    batch_size,
    in_channels,
    input_length,
    kernel_size,
    padding,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    block_idx = tl.program_id(2)

    # compute base pointer for this batch/channel
    batch_stride = in_channels * input_length
    channel_stride = input_length

    base_offset = batch_idx * batch_stride + channel_idx * channel_stride

    block_start = block_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # compute output length
    output_length = input_length + 2 * padding - kernel_size + 1
    mask = offsets < output_length

    # For each output position in block, accumulate sum
    sums = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Loop over kernel
    for i in range(kernel_size):
        # compute padded index
        in_offsets = offsets + i - padding
        # check bounds: in_offsets in [0, input_length)
        in_mask = (in_offsets >= 0) & (in_offsets < input_length)
        # load input values
        vals = tl.load(input_ptr + base_offset + in_offsets, mask=in_mask, other=0.0)
        sums += vals

    out = sums / kernel_size
    # store output
    tl.store(out_ptr + batch_idx * batch_stride + channel_idx * channel_stride + block_start, out, mask=mask)
@triton.jit
def bias_add_kernel(
    output_ptr,  # Pointer to output tensor
    bias_ptr,    # Pointer to bias tensor
    n_channels,  # Number of channels (64)
    n_elements_per_channel,  # 128*128 = 16384
    BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < (n_channels * n_elements_per_channel)
    channel_index = offset // n_elements_per_channel
    bias_index = channel_index
    bias = tl.load(bias_ptr + bias_index, mask=channel_index < n_channels)
    output = tl.load(output_ptr + offset, mask=mask, other=0.0)
    output = output + bias
    tl.store(output_ptr + offset, output, mask=mask)
@triton.jit
def avg_pool_kernel(
    x_ptr,  # (batch, out_features)
    out_ptr,  # (batch, pooled_size)
    batch_size,
    out_features,
    pool_kernel_size: tl.constexpr,
    pooled_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element? Or multiple?
    # Let's have one program per output element.
    # program_id(0) is batch index
    # program_id(1) is pooled index
    pid_batch = tl.program_id(0)
    pid_pooled = tl.program_id(1)

    # Global index in the output
    output_idx = pid_batch * pooled_size + pid_pooled
    # Base index in the input for this pooled window
    start_idx = pid_pooled * pool_kernel_size

    # Load the window of pool_kernel_size elements
    offsets = start_idx + tl.arange(0, pool_kernel_size)
    mask = offsets < out_features
    x = tl.load(x_ptr + pid_batch * out_features + offsets, mask=mask, other=0.0)

    # Compute average
    sum_val = tl.sum(x, axis=0)
    count = tl.sum(mask, axis=0)
    avg = sum_val / count

    # Store
    tl.store(out_ptr + output_idx, avg)
@triton.jit
def reverse_cumsum_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    # Additional arguments to control kernel behavior
    dim: tl.constexpr,
    stride: tl.constexpr,
):
    # The program_id axis 0 corresponds to the row (or sequence) dimension
    row_id = tl.program_id(0)
    
    # Determine the starting offset for this block in the input
    # Each thread handles BLOCK_SIZE elements in the sequence dimension
    block_start = row_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle edge cases when the total length isn't divisible by BLOCK_SIZE
    mask = offsets < n_elements

    # Load data into registers
    x = tl.load(x_ptr + offsets * stride, mask=mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform the reverse cumulative sum: from end to start
    # Use a loop that runs backwards through the block
    for i in range(BLOCK_SIZE - 1, -1, -1):
        # Update accumulator with current element
        acc += x[i]
        # Store the result in reverse order (from the end)
        out_idx = offsets[i]
        tl.store(out_ptr + out_idx * stride, acc, mask=offsets[i] < n_elements)
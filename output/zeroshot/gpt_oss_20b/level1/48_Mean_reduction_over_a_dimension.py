@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    stride_n,
    stride_d, # stride along dim to reduce
    N,  # number of rows
    D,  # size of reduced dimension
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= N:
        return
    # base pointer for row
    base_ptr = input_ptr + row_id * stride_n
    sum = tl.zeros([REDUCE_BLOCK], dtype=tl.float32) # if D large, iterate
    for offset in range(0, D, REDUCE_BLOCK):
        offsets = base_ptr + offset * stride_d + tl.arange(0, REDUCE_BLOCK)
        mask = offsets < base_ptr + D * stride_d
        vals = tl.load(offsets, mask=mask, other=0.0)
        sum += vals
    total = tl.sum(sum)  # warp-level? but sum over block
    mean = total / D
    tl.store(output_ptr + row_id, mean)
@triton.jit
def reduce_sum(
    x_ptr,
    y_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < n
    x = tl.load(x_ptr + offset + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = tl.sum(x, dim=0)
    tl.store(y_ptr + pid, y)
@triton.jit
def softmax(
    x_ptr,
    out_ptr,
    stride_x,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.reduce(x, op=tl.maximum, axis=0)
    x = tl.exp(x - max_val)
    sum_exp = tl.reduce(x, op=tl.sum, axis=0)
    out = x / sum_exp
    tl.store(out_ptr + offsets, out, mask=mask)
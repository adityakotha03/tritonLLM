@triton.jit
def argmax_last_dim_kernel(x_ptr, out_idx_ptr, out_val_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    # Each program processes one row
    row = tl.program_id(0)
    # offset in x: row * K
    row_start = row * K
    # We'll keep track of max value and index
    max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
    max_idx = tl.arange(0, BLOCK_SIZE)
    # But we need to iterate over K
    # We'll load in chunks
    offset = 0
    while offset < K:
        offsets = row_start + offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_start + K
        vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        idxs = tl.arange(offset, offset + BLOCK_SIZE)
        # Compare
        cond = vals > max_val
        max_val = tl.where(cond, vals, max_val)
        max_idx = tl.where(cond, idxs, max_idx)
        offset += BLOCK_SIZE
    # Now we need to reduce across threads (within block). But each program is one row, so we just need to reduce across threads? Wait, we used BLOCK_SIZE as chunk size, but we can just do per-element within the row. After loop, max_val and max_idx contain values for each thread; we need to find global max across the row. We can use block-wide reduction using tl.reduce? There's tl.reduce? Actually Triton provides tl.reduce? There is tl.max, tl.argmax? There's tl.max and tl.argmin? Let's check. I think Triton has tl.argmax for reductions across a block? Not sure. But we can implement reduction by using tl.reduce to compute max across 128 threads. We can use tl.max or tl.reduce with custom lambda? Wait, we can use tl.max() on a vector? There's tl.max? Not sure. I think there is tl.max or tl.argmin? But we can do manual reduction by performing warp-level reduction within the block. However each program is a block; we can treat each thread in block as element in row. But we might not need to reduce across threads if we compute per-thread max over its chunk and then do final reduction. But each thread will produce a candidate; we can perform a block-wide reduction using `tl.reduce` functions. Triton provides `tl.reduce`, I recall reading. Let's look: there's `tl.reduce` with lambda and init. But easier: we can use `tl.max` maybe. However we need index as well. Might need to implement custom reduction: we can have thread 0 (maybe only one thread per block) compute final result by scanning across all `max_val` values.

But we can also just allocate 128 threads, but each thread processes one element; we then use a warp-level reduction with `tl.warp_reduce_max`? Actually there is `tl.warp_reduce_max`, I think. But not sure.

Alternatively, simpler: Use `tl.max` across a vector, but we also need index. We can pack value and index into two arrays: after the loop, we can store them into shared memory or local arrays, then do a reduction across 128 values using `tl.reduce`. Implementation: After loop, we need to find global max across 128 candidate values. We can do:

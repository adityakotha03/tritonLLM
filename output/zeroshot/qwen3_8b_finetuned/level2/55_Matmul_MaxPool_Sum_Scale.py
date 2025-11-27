@triton.jit
def max_pool_1d_kernel(in_ptr0, out_ptr0, nbatch, nhead, nfeatures, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    tl.full([XBLOCK, nhead], True, tl.int1)
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None] * (nhead * nfeatures) + tl.arange(0, nhead)[None, :] * nfeatures
    xmask = xindex < (nbatch * nhead * nfeatures)
    x3 = xindex % nfeatures
    x2 = xindex // nfeatures
    x0 = xindex // (nhead * nfeatures)
    tmp0 = tl.load(in_ptr0 + xindex, xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, nhead])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, nhead, nfeatures])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 2)
    tmp8 = tmp7 / tl.broadcast_to(tl.full([XBLOCK, nhead], 2, tl.int32), [XBLOCK, nhead, nfeatures])
    tmp9 = tmp4 - tmp8
    tmp10 = tl.sum(tmp9, 2)
    tmp11 = tl.sum(tmp10, 1)
    tmp12 = tmp11 / tl.broadcast_to(tl.full([XBLOCK, nhead], 2, tl.int32), [XBLOCK, nhead, nfeatures])
    tmp13 = tmp10 - tmp12
    tmp14 = tl.sum(tmp13, 1)
    tl.store(out_ptr0 + x0, tmp14, xmask)
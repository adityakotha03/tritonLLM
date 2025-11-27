@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex
    r0 = tl.arange(0, rnumel)
    r1 = r0
    tl.full([XBLOCK, rnumel], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0 + r0, xmask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, rnumel])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tl.load(out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp8 = tmp6 * tmp7
    tmp9 = tl.full([XBLOCK, 1], 0, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.where(tmp8 < tmp10, tmp10, tmp8)
    tl.store(out_ptr0 + x0, tmp11, xmask)
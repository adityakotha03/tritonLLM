@triton.jit
def triton_poi_fused_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex % 256
    x2 = xindex // 256
    x1 = xindex // 65536
    x0 = xindex // 16777216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x4, xmask)
    tmp1 = 2.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x4, tmp2, xmask)
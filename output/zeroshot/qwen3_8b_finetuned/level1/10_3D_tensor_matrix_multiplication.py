@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = x3 // 768
    x1 = x3 % 768
    x4 = x3
    x0 = x4 % 2048
    x5 = x4 // 2048
    x6 = x0
    x7 = x2
    tmp0 = tl.load(in_ptr0 + (x7 * 2048 + x6), x6 < 2048)
    tmp1 = tl.load(in_ptr1 + (x6 + 2048 * x1), x6 < 2048)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(out_ptr0 + xindex, None)
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + xindex, tmp6, None)
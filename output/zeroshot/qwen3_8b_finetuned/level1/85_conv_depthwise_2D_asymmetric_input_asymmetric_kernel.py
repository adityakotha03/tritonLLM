@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel, znumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.arange(0, ynumel)[None, :]
    yoffset = yoffset % ynumel
    tl.full([XBLOCK, ynumel], True, tl.int1)
    tl.full([XBLOCK, ynumel], True, tl.int1)
    yindex = yoffset
    xindex = xoffset
    tmp0 = tl.load(in_ptr0 + (xindex + yindex), None)
    tmp1 = tl.load(in_ptr1 + yindex, None)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + yindex, tmp2, None)
@triton.jit
def triton_poi_fused_matmul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, ynumel, 
    znumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xoffset + tl.arange(0, XBLOCK)[:, None]
    yoffset = tl.full([XBLOCK, YBLOCK], tl.program_id(1) * YBLOCK)
    yoffset + tl.arange(0, YBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], xnumel)
    tl.full([XBLOCK, YBLOCK], ynumel)
    tl.full([XBLOCK, YBLOCK], znumel)
    yoffset = yoffset % ynumel
    yoffset = yoffset
    tl.full([XBLOCK, YBLOCK], xnumel)
    tl.full([XBLOCK, YBLOCK], ynumel)
    tl.full([XBLOCK, YBLOCK], znumel)
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    yindex = yindex % ynumel
    ymask = yindex < ynumel
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xindex = xindex % xnumel
    xmask = xindex < xnumel
    ymask = ymask & xmask
    rindex = yindex
    tl.full([XBLOCK, YBLOCK], xnumel)
    tl.full([XBLOCK, YBLOCK], ynumel)
    tl.full([XBLOCK, YBLOCK], znumel)
    r1 = tl.load(in_ptr0 + (rindex + tl.arange(0, XBLOCK)[None, :]), xmask, 
        other=0.0)
    r2 = tl.load(in_ptr1 + (rindex + tl.arange(0, XBLOCK)[None, :]), xmask, 
        other=0.0)
    r3 = r1 * r2
    tl.store(in_out_ptr0 + rindex, r3, xmask)
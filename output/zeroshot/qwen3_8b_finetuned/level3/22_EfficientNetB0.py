@triton.jit
def triton_poi_fused_relu6_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = 6.0
    tmp3 = tmp1 > tmp2
    tmp4 = tmp1 <= tmp2
    tmp5 = tmp1 < 0.0
    tmp6 = tmp3 | tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = 0.0
    tmp9 = tmp1 * tmp7
    tmp10 = tmp9 + tmp8
    tmp11 = tmp1 - tmp10
    tmp12 = tmp1 <= tmp2
    tmp13 = tmp11 <= tmp2
    tmp14 = tmp12 & tmp13
    tmp15 = tmp10 + tmp14
    tl.store(in_out_ptr0 + x0, tmp15, xmask)
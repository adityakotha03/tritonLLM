@triton.jit
def gelu_kernel(in_ptr0, out_ptr0, n_elements, XBLOCK: tl.constexpr):
    xnumel = n_elements
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.044715
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp6 = 2.0 / 3.141592653589793
    tmp7 = math.sqrt(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = math.tanh(tmp8)
    tmp10 = 1.0 + tmp9
    tmp11 = tmp0 * tmp10
    tmp12 = 0.5 * tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)
import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_abs_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 161061216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = xindex % 393216
    x1 = xindex // 393216
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp1 = tl.abs(tmp0)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_rawbuf0 = buf0
        triton_poi_fused_abs_div_0[grid(161061216)](arg0_1, get_rawbuf0, 161061216, XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a Softsign activation, implemented with a custom Triton kernel.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_softplus_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 156294656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl.where(tmp2, tmp0, tl.where(tmp2, tmp0, 1.0))
    tmp4 = tl.where(tmp2, tmp0, 1.0)
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = -1.0
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp2, tmp6, tmp8)
    tmp10 = tl.where(tmp2, tmp4, tmp9)
    tmp11 = tl.where(tmp2, tmp3, tmp10)
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 393216), (393216, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4096, 393216), (393216, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_softplus_0[grid(156294656)](arg0_1, buf0, 156294656,
            XBLOCK=512, num_warps=8, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]

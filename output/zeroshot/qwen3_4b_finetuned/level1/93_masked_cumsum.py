import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_cumsum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = tl.full([1], 1, tl.int8)
    tmp6 = tl.full([1], 0, tl.int8)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp10 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp12 = tl.sum(tmp10, 0)[:, None]
    tl.store(out_ptr0 + x0, tmp12, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cumsum_0[grid(32768)](arg0_1, arg1_1, buf0, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]

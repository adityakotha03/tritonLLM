import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_cat_cumsum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (-1 + x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (-1 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 + -1
    tmp9 = tmp6 + tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp2, tmp10, tmp9)
    tmp12 = tmp3 + tmp11
    tl.store(out_ptr0 + x0, tmp12, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), 32768)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_cumsum_0[grid(32768)](arg0_1, buf0, 32768,
            XBLOCK=256, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
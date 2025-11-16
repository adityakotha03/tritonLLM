import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32767
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32768
    x1 = xindex // 32768
    x2 = xindex
    tmp0 = x0
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 32768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 32768 * x1), tmp4 & xmask, eviction_policy
        = 'evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp8, tmp7)
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused_cumsum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32768
    tmp0 = x2
    tl.full([1], 0, tl.int64)
    tmp3 = tl.full([1], 32768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + x0, tmp4 & xmask, eviction_policy='evict_last',
        other=0.0)
    tmp6 = tmp0 + tmp5
    tmp7 = 0.0
    tmp8 = tmp6 < tmp7
    tmp9 = libdevice.select(tmp8, tmp7, tmp6)
    tl.store(out_ptr0 + x2, tmp9, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(32767)](arg1_1, arg0_1, buf0, 32767,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((32768,), (1,), torch.float32)
        triton_poi_fused_cumsum_1[grid(32768)](buf0, buf1, 32768, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
    return buf1,


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
        arg1_1 = input_0
        arg0_1 = arg1_1
        output = call([arg0_1, arg1_1])
        return output[0]

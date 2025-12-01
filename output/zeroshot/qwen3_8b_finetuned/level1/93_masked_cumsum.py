import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_cumsum_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tmp0 + tmp2
    tmp4 = tmp1 + tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32768,), (1,))
    assert_size_stride(arg1_1, (32768,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768,), (1,), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(32768)](arg0_1, arg1_1, buf0, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((32768,), (1,), torch.float32)
        triton_poi_fused_cumsum_1[grid(32768)](buf0, arg1_1, buf1, 32768,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
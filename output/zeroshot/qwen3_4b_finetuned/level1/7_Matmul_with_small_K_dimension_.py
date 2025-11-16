import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 16384 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 32 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (x0 % 16384 + 16384 * tl.floor(x0 / 16384)), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0 // 16384 * 32 + 32 * (x0 % 16384)), xmask,
        eviction_policy='evict_last')
    tmp4 = tmp0 * tmp1
    tmp5 = tmp2 * tmp3
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16384, 32), (32, 1))
    assert_size_stride(arg1_1, (32, 65536), (65536, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 65536), (65536, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(1048576)](arg0_1, arg1_1, buf0, 1048576,
            XBLOCK=512, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]

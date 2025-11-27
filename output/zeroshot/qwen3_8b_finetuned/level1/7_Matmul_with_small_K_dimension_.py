import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, XBLOCK: tl.constexpr, YBLOCK: tl.constexpr):
    xnumel = 32768
    ynumel = 32768
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    x0 = xindex
    y1 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 16384 * y3), xmask & ymask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 64 * y1), xmask & ymask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.load(out_ptr0 + (x0 + 16384 * y1), xmask & ymask, eviction_policy
        = 'evict_last')
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0 + 16384 * y1), tmp4, xmask & ymask)


def call(args):
    (primals_1, primals_2) = args
    args.clear()
    assert_size_stride(primals_1, (32768, 64), (64, 1))
    assert_size_stride(primals_2, (64, 32768), (1, 64))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 32768), (32768, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_add_0[grid(32768, 32768)](primals_1, primals_2,
            buf0, 32768, 32768, 0, 0, XBLOCK=128, YBLOCK=64, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, input_0, input_1):
        arg0_1, arg1_1 = input_0, input_1
        output = call([arg0_1, arg1_1])
        return output[0]
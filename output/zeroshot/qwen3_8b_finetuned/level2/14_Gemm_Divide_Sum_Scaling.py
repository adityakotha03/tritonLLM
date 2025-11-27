import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, rnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    ynumel = 1024
    yoffset = 1
    yindex = tl.arange(0, XBLOCK)[:]
    ymask = yindex < ynumel
    xoffset = 0
    xindex = tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex
    y1 = yindex
    y2 = yindex
    tmp0 = tl.load(in_ptr0 + (y3 + x2 * ynumel), xmask & ymask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + x2 * ynumel), xmask & ymask, eviction_policy
        ='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.load(in_ptr0 + (y1 + x2 * ynumel), xmask & ymask, eviction_policy
        ='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tl.load(in_ptr0 + (y2 + x2 * ynumel), xmask & ymask, eviction_policy
        ='evict_last')
    tmp6 = tmp4 + tmp5
    tl.store(out_ptr0 + (y3 + x2 * ynumel), tmp6, xmask & ymask)


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 * 8192), xmask)
    tmp1 = 2.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.sum(tmp3, 0)[:, None]
    tmp5 = tmp4 * 1.5
    tl.store(out_ptr0 + x2, tmp5, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(8192, 1024, 8192)](primals_2,
            primals_1, buf0, 8192, 1024, 0, 1, 8192, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((1024, 1), (1, 1), torch.float32)
        triton_poi_fused_convolution_1[grid(1024)](buf0, buf1, 1024, XBLOCK=
            128, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_2, buf1


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_1 = self.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]
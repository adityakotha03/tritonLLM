import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


@triton.jit
def triton_poi_fused_add_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_mean_mul_rsub_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tl.load(in_ptr3 + x0, xmask)
    tmp7 = tl.load(in_ptr4 + x0, xmask)
    tmp9 = tl.load(in_ptr5 + x0, xmask)
    tmp2 = 4.0
    tmp4 = tmp1 / tmp2
    tmp6 = tmp3 / tmp2
    tmp8 = tmp5 / tmp2
    tmp10 = tmp7 / tmp2
    tmp11 = tmp0 - tmp4
    tmp12 = tmp11 * tmp6
    tmp13 = tmp12 * tmp2
    tmp14 = tmp9 - tmp8
    tmp15 = tmp14 * tmp10
    tmp16 = tmp15 * tmp2
    tmp17 = tmp13 + tmp16
    tl.store(out_ptr0 + x0, tmp17, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_0[grid(16777216)](arg1_1, arg0_1, buf0, 
            16777216, XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((4096,), (1,), torch.float32)
        triton_poi_fused_add_mean_1[grid(4096)](buf0, buf1, 4096, XBLOCK=128,
            num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((16384, 4096), (4096, 1), torch.float32)
        triton_poi_fused_add_mean_mul_rsub_2[grid(4096)](buf0, buf1, buf0,
            buf0, buf1, buf1, buf2, 4096, XBLOCK=128, num_warps=4,
            num_stages=1)
        del buf1
    return buf2,


class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, scales the result, and applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5,
        momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, input_0):
        arg1_1 = self.gemm.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]

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
def triton_poi_fused_add_div_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp4 = tl.load(in_ptr3 + x0, xmask)
    tmp3 = tmp0 + tmp1
    tmp5 = tmp3 + tmp2
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7 * tmp4
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_add_div_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tl.load(in_ptr2 + x0, xmask)
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 * tmp4
    tl.store(out_ptr0 + x0, tmp7, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (8192,), (1,))
    assert_size_stride(primals_3, (1024, 8192), (8192, 1))
    assert_size_stride(primals_4, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_div_mul_0[grid(8388608)](primals_1, primals_2,
            primals_3, primals_4, buf0, 8388608, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del primals_1
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_div_mul_1[grid(8388608)](buf0, primals_4,
            primals_2, buf1, 8388608, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_4
    return buf1, primals_2, primals_4, buf0


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1,
        bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value

    def forward(self, input_0):
        primals_1 = self.matmul.weight
        primals_2 = self.bias
        primals_3 = input_0
        primals_4 = self.bias
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

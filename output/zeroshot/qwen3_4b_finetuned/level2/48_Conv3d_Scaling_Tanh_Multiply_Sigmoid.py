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
def triton_poi_fused_add_mul_sigmoid_tanh_0(in_ptr0, in_ptr1, in_ptr2,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 381248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = 0.5
    tmp5 = tmp2 * tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp3 * tmp6
    tmp8 = torch.tanh(tmp7)
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (243, 81, 27, 9, 3), 0)
    assert_size_stride(primals_2, (128, 3, 16, 64, 64), (2048, 684, 42, 64, 1
        ), 0)
    assert_size_stride(primals_3, (16,), (1,), 0)
    assert_size_stride(primals_4, (16,), (1,), 0)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 14, 62, 62), (73728, 4646, 326,
            5, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_tanh_0[grid(381248)](primals_1,
            primals_2, primals_3, buf0, 381248, XBLOCK=512, num_warps=8,
            num_stages=1)
        del primals_1
        del primals_3
    return buf0, primals_2, primals_4


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor,
        bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.scaling_factor
        primals_4 = self.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

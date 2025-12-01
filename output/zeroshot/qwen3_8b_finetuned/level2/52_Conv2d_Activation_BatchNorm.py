import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_softplus_tanh_0(in_out_ptr0, in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp4 = libdevice.expm1(tmp2)
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp0 * tmp7
    tl.store(in_out_ptr0 + x2, tmp0, xmask)
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 128, 128, 128), (2097152, 16384, 128, 
            1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((64, 128, 128, 128), (2097152, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_mul_softplus_tanh_0[grid(134217728)](buf1,
            primals_3, buf2, 134217728, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_3
    return buf2, primals_1, primals_2, buf1


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies activation, and then applies Batch Normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = self.bn.weight
        primals_4 = self.bn.bias
        primals_1 = primals_1
        primals_2 = primals_2
        primals_3 = primals_3
        primals_4 = primals_4
        input_0 = input_0
        output = call([primals_1, primals_2, primals_3, input_0])
        return output[0]
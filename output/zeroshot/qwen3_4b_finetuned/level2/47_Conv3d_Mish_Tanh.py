import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1887456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 516096 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mish_tanh_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1887456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp0 > tmp3
    tmp5 = 1.7015113416954286
    tmp6 = tmp0 * tmp5
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp7 * tmp1
    tmp9 = tl.where(tmp4, tmp2, tmp8)
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl_math.log(tmp10)
    tmp12 = tmp9 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = 0.01
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 + tmp1
    tmp17 = tmp9 * tmp16
    tl.store(out_ptr0 + x0, tmp17, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 32, 64, 64), (131072, 4096, 128,
        2, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 30, 62, 62), (754368, 11787, 393, 
            6, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1887456)](buf1, primals_2, 
            1887456, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16, 64, 30, 62, 62), (754368, 11787, 393,
            6, 1), torch.float32)
        triton_poi_fused_mish_tanh_1[grid(1887456)](buf1, buf2, 1887456,
            XBLOCK=512, num_warps=8, num_stages=1)
    return buf2, primals_1, primals_3, buf1


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

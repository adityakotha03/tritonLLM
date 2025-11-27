import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 65536 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mish_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = libdevice.log1p(tmp4)
    tmp6 = 2.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp4 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4194304, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 128, 254, 254), (8398048, 65536, 258,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(131072)](buf1, primals_2, 
            131072, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((64, 128, 254, 254), (8398048, 65536, 258,
            1), torch.float32)
        triton_poi_fused_mish_1[grid(131072)](buf1, buf2, 131072, XBLOCK=
            512, num_warps=8, num_stages=1)
        buf3 = extern_kernels.convolution(buf2, primals_1, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (64, 128, 252, 252), (8398048, 65536, 258, 
            1))
        buf4 = buf3
        del buf3
        triton_poi_fused_convolution_2[grid(131072)](buf4, primals_2, 
            131072, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf5 = empty_strided_cuda((64, 128, 252, 252), (8398048, 65536, 258,
            1), torch.float32)
        triton_poi_fused_mish_3[grid(131072)](buf4, buf5, 131072, XBLOCK=
            512, num_warps=8, num_stages=1)
    return buf5, primals_1, primals_3, buf1, buf2, buf4


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

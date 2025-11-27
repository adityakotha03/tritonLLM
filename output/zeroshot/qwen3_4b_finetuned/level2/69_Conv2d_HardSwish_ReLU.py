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
def triton_poi_fused_hardswish_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 * tmp3
    tmp5 = 3.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 12, tl.int32)
    tmp8 = tmp6 > tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_hardswish_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 * tmp3
    tmp5 = 3.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 12, tl.int32)
    tmp8 = tmp6 > tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_hardswish_relu_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 * tmp3
    tmp5 = 3.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 12, tl.int32)
    tmp8 = tmp6 > tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (64, 8, 3, 3), (72, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(arg0_1, arg0_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128,
            1))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_hardswish_relu_0[grid(134217728)](buf0, buf1, 
            134217728, XBLOCK=1024, num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.bool)
        triton_poi_fused_hardswish_relu_1[grid(134217728)](buf0, buf2,
            134217728, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf3 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.bool)
        triton_poi_fused_hardswish_relu_2[grid(134217728)](buf0, buf3,
            134217728, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
    return buf1, buf2, buf3, arg0_1


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        arg0_1 = self.conv.weight
        output = call([arg0_1, input_0])
        return output[0]

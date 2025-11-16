import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 65536 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)


@triton.jit
def triton_poi_fused_mish_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 262144 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 262144 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 262144 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (192 + x0 + 262144 * x1), xmask)
    tmp2 = libdevice.tanh(tmp1)
    tmp4 = libdevice.tanh(tmp3)
    tmp5 = tmp2 * tmp4
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


@triton.jit
def triton_poi_fused_mish_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 262144 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 262144 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 262144 * x1), xmask)
    tmp6 = tl.load(in_ptr0 + (192 + x0 + 262144 * x1), xmask)
    tmp2 = libdevice.tanh(tmp1)
    tmp4 = libdevice.tanh(tmp3)
    tmp5 = tmp2 * tmp4
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = tmp5 * tmp7
    tmp9 = 1.0
    tmp10 = tmp8 + tmp9
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + x2, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4194304, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1,
            64, 256, 256), (4194304, 65536, 256, 1), 0), primals_1, stride=(1,
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 128, 256, 256), (8388608, 65536, 256, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(134217728)](buf1, primals_2, 
            134217728, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1, 128, 256, 256), (8388608, 65536, 256, 
            1), torch.float32)
        triton_poi_fused_mish_1[grid(1048576)](buf1, buf2, 1048576, XBLOCK=
            1024, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((1, 128, 256, 256), (8388608, 65536, 256, 
            1), torch.float32)
        triton_poi_fused_mish_2[grid(1048576)](buf2, buf3, 1048576, XBLOCK=
            512, num_warps=8, num_stages=1)
        del buf2
    return buf3, reinterpret_tensor(primals_3, (1, 64, 256, 256), (4194304, 
        65536, 256, 1), 0), primals_1, buf1, buf3


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

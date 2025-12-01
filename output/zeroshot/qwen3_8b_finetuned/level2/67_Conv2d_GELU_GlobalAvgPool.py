import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + x0, tmp8, xmask)


@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.where(xmask, tmp2, 0)
    tmp4 = tl.sum(tmp3, 0)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (128, 8, 256, 256), (524288, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 254, 254), (4194304, 65536, 256, 
            1))
        buf1 = empty_strided_cuda((128, 64, 254, 254), (4194304, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_0[grid(1048576)](buf0, buf1, 1048576,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        buf2 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        triton_poi_fused_mean_1[grid(8192)](buf1, buf2, 8192, XBLOCK=256,
            num_warps=4, num_stages=1)
        del buf1
    return buf2, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]
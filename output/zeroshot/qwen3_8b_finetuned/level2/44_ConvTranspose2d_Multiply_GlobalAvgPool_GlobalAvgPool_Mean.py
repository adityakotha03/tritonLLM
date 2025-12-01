import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_mul_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr1 + 0)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp7 = tmp6 * tmp4
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp9 = tl.where(xmask, tmp8, 0)
    tmp10 = tl.sum(tmp9, 0)[:, None]
    tmp11 = 16.0
    tmp12 = tmp10 / tmp11
    tl.store(out_ptr0 + x2, tmp12, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 16, 1, 1), (16, 1, 16, 1), torch.
            float32)
        extern_kernels.convolution(primals_3, primals_1, stride=(2, 2),
            padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None, stride=(2, 2),
            padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        del primals_1
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_mul_mean_0[grid(128)](buf1, primals_2, buf1, 128,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return buf1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
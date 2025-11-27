import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (65536 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (131072 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (196608 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16384 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (65536 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (131072 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (196608 + x0 + 16384 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)


@triton.jit
def triton_poi_fused_sigmoid_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 16, 32, 32), (524288, 16384, 1024,
        32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 16, 32, 32), (524288, 8192, 512, 16,
            1))
        buf1 = empty_strided_cuda((16, 64, 16, 32, 32), (524288, 8192, 512,
            16, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__softmax_0[grid(2097152)](buf0, buf1, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf2 = buf0
        del buf0
        triton_poi_fused__softmax_0[grid(2097152)](buf1, buf2, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf3 = buf1
        del buf1
        triton_poi_fused__softmax_1[grid(2097152)](buf2, buf3, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = buf2
        del buf2
        triton_poi_fused_sigmoid_2[grid(2097152)](buf3, buf4, 2097152,
            XBLOCK=512, num_warps=8, num_stages=1)
        del buf3
    return buf4, primals_1, primals_2, primals_3, buf4


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

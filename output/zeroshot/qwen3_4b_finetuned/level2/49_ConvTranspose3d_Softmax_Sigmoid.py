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
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 15360 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 15360 % 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (15360 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (30720 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (46080 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (61440 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 - tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = tmp8 < tmp9
    tl.store(out_ptr0 + x3, tmp10, xmask)


@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 15360 % 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (15360 + x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (30720 + x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (46080 + x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (61440 + x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 - tmp7
    tmp9 = tl.sigmoid(tmp8)
    tl.store(out_ptr0 + x3, tmp9, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 16, 32, 32), (524288, 16384, 32768,
        1024, 1))
    assert_size_stride(arg1_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 16, 32, 32), (524288, 8192, 512,
            16, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(491520)](buf0, arg1_1, 491520,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg1_1
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((16, 64, 16, 32, 32), (524288, 8192, 512,
            16, 1), torch.bool)
        triton_poi_fused__softmax_1[grid(491520)](buf1, buf2, 491520,
            XBLOCK=1024, num_warps=4, num_stages=1)
        buf3 = buf1
        del buf1
        triton_poi_fused__softmax_2[grid(491520)](buf2, buf3, 491520,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf2
    return buf3, arg0_1


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_0):
        arg1_1 = self.conv_transpose.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]

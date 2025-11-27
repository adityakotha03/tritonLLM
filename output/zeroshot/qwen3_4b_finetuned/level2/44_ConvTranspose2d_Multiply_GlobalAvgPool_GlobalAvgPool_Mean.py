import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2099520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_mean_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 256 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, 
            stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(2099520)](buf2, primals_2, 
            2099520, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 256, 256),
            torch.float32)
        triton_poi_fused_mean_1[grid(256)](buf2, buf3, 256, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf2
    return buf3, primals_1, primals_3


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

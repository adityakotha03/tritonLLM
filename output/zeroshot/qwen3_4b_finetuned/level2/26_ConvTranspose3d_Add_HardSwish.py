import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 256 % 256
    x0 = xindex % 256
    x4 = xindex // 1024
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 1024 * x4), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr2 + x3, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + 1024 * x4), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 3.0
    tmp10 = tmp6 * tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tl_math.rsqrt(tmp14)
    tmp16 = tmp6 * tmp15
    tmp17 = tl.where(tmp15 < tmp9, tmp15, tmp9)
    tmp18 = tl.where(tmp16 < tmp8, tmp16, tmp8)
    tl.store(out_ptr0 + x5, tmp17, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 32, 16, 16, 16), (8192, 256, 16, 1, 
        0))
    assert_size_stride(primals_3, (128, 64, 33, 33, 33), (72576, 1136, 33, 
        1, 0))
    assert_size_stride(primals_4, (128, 64), (64, 1))
    assert_size_stride(primals_5, (128, 64, 33, 33, 33), (72576, 1136, 33, 
        1, 0))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_2, (
            128, 32, 16, 16, 16), (256, 8, 1, 1, 1), 0, 0, 1, 0, 1), primals_1,
            stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1),
            transposed=True, output_padding=(1, 1, 1), groups=1,
            bias=None)
        assert_size_stride(buf0, (128, 64, 33, 33, 33), (72576, 1136, 33, 
            1, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64, 33, 33, 33), (72576, 1136, 33,
            1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(12582912)](buf1, primals_3,
            primals_4, primals_5, buf2, 12582912, XBLOCK=2048, num_warps=16,
            num_stages=1)
    return buf2, primals_1, primals_2, primals_3, primals_4, primals_5


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0, input_1):
        primals_1 = self.conv_transpose.weight
        primals_2 = input_0
        primals_4 = self.bias
        primals_3 = self.conv_transpose.bias
        primals_5 = input_1
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_mul_0(in_out_ptr0,
    in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1876800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = xindex // 15360 % 16
    x0 = xindex % 15360
    x3 = xindex // 15360
    tmp0 = tl.load(in_out_ptr0 + x4, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x3, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 3.0
    tmp5 = tmp2 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = -1.0
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = tmp13 - tmp3
    tmp15 = tmp13 > tmp10
    tmp16 = tmp13 < tmp12
    tmp17 = tmp15 & tmp16
    tl.store(in_out_ptr0 + x4, tmp2, xmask)
    tl.store(out_ptr0 + x4, tmp13, xmask)
    tl.store(out_ptr1 + x4, tmp17, xmask)


@triton.jit
def triton_per_fused__logsumexp_1(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 128
    rnumel = 15360
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 15360 * x0), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float('-inf'))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl_math.log(tmp10)
    tl.store(out_ptr1 + (r1 + 15360 * x0), tmp11, rmask & xmask)


@triton.jit
def triton_poi_fused__logsumexp_add_mul_sub_2(in_out_ptr0, in_ptr0, in_ptr1,
    in_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_out_ptr0 + x0, xmask)
    tmp7 = tl.load(in_ptr2 + x0, xmask)
    tmp3 = tmp0 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 - tmp7
    tmp9 = tl_math.exp(tmp5)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp10 * tmp10
    tmp12 = tmp11 * tmp10
    tmp13 = tmp12 + tmp8
    tmp14 = tl_math.log(tmp13)
    tmp15 = tmp14 + tmp10
    tmp16 = tmp15 - tmp14
    tl.store(in_out_ptr0 + x0, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (49152, 16384, 512,
        16, 1))
    assert_size_stride(primals_4, (1, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(reinterpret_tensor(
            primals_3, (1, 3, 16, 32, 32), (0, 15360, 960, 30, 1), 0), 
            primals_1, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1,
            1), transposed=True, output_padding=(0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (128, 16, 15, 16, 16), (61440, 3840, 256, 
            16, 1))
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 16, 15, 16, 16), (61440, 3840, 256,
            16, 1), torch.float32)
        buf3 = empty_strided_cuda((128, 16, 15, 16, 16), (61440, 3840, 256,
            16, 1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_hardtanh_hardtanh_backward_mul_0[grid
            (1876800)](buf1, primals_2, primals_4, buf2, buf3, 1876800,
            XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf4 = empty_strided_cuda((128, 15360), (15360, 1), torch.float32)
        triton_per_fused__logsumexp_1[grid(128)](buf2, buf4, 128, 15360,
            XBLOCK=32, num_warps=4, num_stages=1)
        buf5 = reinterpret_tensor(buf2, (128, 1, 15, 16, 16), (3840, 3840, 
            256, 16, 1), 0)
        del buf2
        triton_poi_fused__logsumexp_add_mul_sub_2[grid(128)](buf5, buf4,
            primals_4, buf1, 128, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
        del buf4
    return buf5, primals_1, primals_4, reinterpret_tensor(primals_3, (1, 3,
        16, 32, 32), (0, 15360, 960, 30, 1), 0), buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

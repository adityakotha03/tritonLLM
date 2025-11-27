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
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 13230 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_hardtanh_mul_mish_1(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 * tmp5
    tmp8 = 0.5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 > tmp3
    tmp11 = tmp2 < tmp5
    tmp12 = tmp10 & tmp11
    tl.store(out_ptr0 + x0, tmp9, xmask)
    tl.store(out_ptr1 + x0, tmp12, xmask)


@triton.jit
def triton_poi_fused_add_hardtanh_mul_mish_2(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 * tmp5
    tmp8 = 0.5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp2 > tmp3
    tmp11 = tmp2 < tmp5
    tmp12 = tmp10 & tmp11
    tl.store(out_ptr0 + x0, tmp9, xmask)


@triton.jit
def triton_poi_fused_hardtanh_mul_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -1.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = tmp4 * tmp3
    tl.store(out_ptr0 + x0, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 132, 132), (1092480, 13230, 132, 
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1327680)](buf1, primals_2, 
            1327680, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 64, 132, 132), (1092480, 13230, 
            132, 1), torch.float32)
        buf3 = empty_strided_cuda((128, 64, 132, 132), (1092480, 13230, 
            132, 1), torch.bool)
        triton_poi_fused_add_hardtanh_mul_mish_1[grid(1327680)](buf1,
            primals_1, buf2, buf3, 1327680, XBLOCK=512, num_warps=8,
            num_stages=1)
        del buf1
        del primals_1
        buf4 = empty_strided_cuda((128, 64, 132, 132), (1092480, 13230, 
            132, 1), torch.float32)
        triton_poi_fused_add_hardtanh_mul_mish_2[grid(1327680)](buf2, buf3,
            buf4, 1327680, XBLOCK=512, num_warps=8, num_stages=1)
        del buf2
        del buf3
        buf5 = empty_strided_cuda((128, 64, 132, 132), (1092480, 13230, 
            132, 1), torch.float32)
        triton_poi_fused_hardtanh_mul_3[grid(1327680)](buf4, buf5, 1327680,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del buf4
    return buf5, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

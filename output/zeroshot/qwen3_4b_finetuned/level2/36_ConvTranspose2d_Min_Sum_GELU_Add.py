import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1296 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_minimum_sum_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100
    x1 = xindex // 100
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1296 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + 1296 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + 1296 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (384 + x0 + 1296 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (512 + x0 + 1296 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (640 + x0 + 1296 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (768 + x0 + 1296 * x1), xmask)
    tmp13 = tl.load(in_ptr0 + (896 + x0 + 1296 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (1024 + x0 + 1296 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (1152 + x0 + 1296 * x1), xmask)
    tmp19 = tl.load(in_ptr0 + (1280 + x0 + 1296 * x1), xmask)
    tmp21 = tl.load(in_ptr0 + (1408 + x0 + 1296 * x1), xmask)
    tmp23 = tl.load(in_ptr0 + (1536 + x0 + 1296 * x1), xmask)
    tmp25 = tl.load(in_ptr0 + (1664 + x0 + 1296 * x1), xmask)
    tmp27 = tl.load(in_ptr0 + (1792 + x0 + 1296 * x1), xmask)
    tmp29 = tl.load(in_ptr0 + (1920 + x0 + 1296 * x1), xmask)
    tmp31 = tl.load(in_ptr0 + (2048 + x0 + 1296 * x1), xmask)
    tmp33 = tl.load(in_ptr0 + (2176 + x0 + 1296 * x1), xmask)
    tmp35 = tl.load(in_ptr0 + (2304 + x0 + 1296 * x1), xmask)
    tmp2 = triton_helpers.minimum(tmp1, tmp0)
    tmp4 = triton_helpers.minimum(tmp3, tmp2)
    tmp6 = triton_helpers.minimum(tmp5, tmp4)
    tmp8 = triton_helpers.minimum(tmp7, tmp6)
    tmp10 = triton_helpers.minimum(tmp9, tmp8)
    tmp12 = triton_helpers.minimum(tmp11, tmp10)
    tmp14 = triton_helpers.minimum(tmp13, tmp12)
    tmp16 = triton_helpers.minimum(tmp15, tmp14)
    tmp18 = triton_helpers.minimum(tmp17, tmp16)
    tmp20 = triton_helpers.minimum(tmp19, tmp18)
    tmp22 = triton_helpers.minimum(tmp21, tmp20)
    tmp24 = triton_helpers.minimum(tmp23, tmp22)
    tmp26 = triton_helpers.minimum(tmp25, tmp24)
    tmp28 = triton_helpers.minimum(tmp27, tmp26)
    tmp30 = triton_helpers.minimum(tmp29, tmp28)
    tmp32 = triton_helpers.minimum(tmp31, tmp30)
    tmp34 = triton_helpers.minimum(tmp33, tmp32)
    tmp36 = triton_helpers.minimum(tmp35, tmp34)
    tmp37 = tmp36 + tmp36
    tl.store(out_ptr0 + x2, tmp37, xmask)


@triton.jit
def triton_poi_fused_add_gelu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + x0, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_4, (1, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (16, 128, 100, 100), (1280000, 10000, 100, 
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(230400)](buf1, primals_2, 
            230400, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((16, 1, 100, 100), (10000, 10000, 100, 1),
            torch.float32)
        triton_poi_fused_minimum_sum_1[grid(1600)](buf1, buf2, 1600, XBLOCK
            =128, num_warps=4, num_stages=1)
        buf3 = empty_strided_cuda((16, 1, 100, 100), (10000, 10000, 100, 1),
            torch.float32)
        triton_poi_fused_add_gelu_2[grid(1600)](buf2, primals_4, buf3, 1600,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
        del primals_4
    return buf3, primals_1, primals_3, buf1


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

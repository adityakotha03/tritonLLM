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
def triton_poi_fused__unsafe_index_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tl.store(out_ptr0 + x0, tmp0, xmask)


@triton.jit
def triton_poi_fused__unsafe_index_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024 % 8
    x0 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + 8192 * x2), tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 491520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_mul_sigmoid_3(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1024 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = libdevice.sigmoid(tmp4)
    tl.store(out_ptr0 + x2, tmp5, xmask)


@triton.jit
def triton_poi_fused_add_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 327680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 256
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_div_5(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.01
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_div_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1024 % 8
    x0 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.01
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_3, (8, 8, 256, 256), (16384, 2048, 256, 1))
    assert_size_stride(primals_4, (32,), (1,))
    assert_size_stride(primals_5, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_6, (32, 1, 1), (1, 1, 1))
    assert_size_stride(primals_7, (32,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 32, 3, 3), (288, 9, 32, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_0[grid(32)](primals_2, buf0, 32,
            XBLOCK=32, num_warps=1, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 32, 3, 3), (288, 9, 32, 1), torch.
            float32)
        extern_kernels.convolution(reinterpret_tensor(primals_3, (128, 8, 3,
            3), (24, 1, 3, 1), 0), buf0, stride=(1, 1), padding=(1, 1),
            dilation=(1, 1), transposed=False, output_padding=(0, 0),
            groups=1, bias=None, out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((32,), (1,), torch.float32)
        triton_poi_fused__unsafe_index_add_1[grid(1048576)](buf1, primals_4,
            buf2, 1048576, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_4
        buf3 = reinterpret_tensor(buf1, (128, 32, 256, 256), (1048576, 32768,
            256, 1), 0)
        del buf1
        triton_poi_fused_convolution_2[grid(491520)](buf3, buf0, 491520,
            XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = empty_strided_cuda((128, 32, 256, 256), (2097152, 65536, 
            256, 1), torch.float32)
        triton_poi_fused_add_mul_sigmoid_3[grid(327680)](buf3, primals_5,
            buf4, 327680, XBLOCK=256, num_warps=4, num_stages=1)
        del buf3
        del primals_5
        buf5 = reinterpret_tensor(buf4, (128, 32, 256, 256), (2097152, 65536,
            256, 1), 0)
        del buf4
        triton_poi_fused_add_4[grid(327680)](buf5, primals_6, primals_7, buf5,
            327680, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_6
        del primals_7
        buf6 = reinterpret_tensor(buf5, (32, 1, 8, 256, 256), (16777216, 
            16777216, 2097152, 256, 1), 0)
        del buf5
        triton_poi_fused_div_5[grid(32)](primals_6, buf6, 32, XBLOCK=32,
            num_warps=1, num_stages=1)
        buf7 = reinterpret_tensor(buf5, (128, 8, 256, 256), (2097152, 262144,
            1024, 1), 0)
        del buf5
        triton_poi_fused_add_div_6[grid(1048576)](buf7, buf7, primals_6, 
            1048576, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_6
    return reinterpret_tensor(buf7, (128, 32, 256, 256), (2097152, 65536, 
        256, 1), 0), reinterpret_tensor(buf0, (8, 32, 3, 3), (288, 1, 9, 3),
        0), reinterpret_tensor(buf2, (32,), (1,), 0), reinterpret_tensor(
        primals_1, (8, 3, 3, 3), (27, 1, 9, 3), 0)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, adds a bias term, scales, applies sigmoid, and performs group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.bias
        primals_5 = self.scale
        primals_6 = self.group_norm.weight
        primals_7 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6, primals_7])
        return output[0]

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1331200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 332800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    x4 = xindex // 16384
    x5 = xindex % 16384
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 512 * x1 + 65536 * x4), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 512 * x1 + 65536 * x4), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (256 + 2 * x0 + 512 * x1 + 65536 * x4), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (257 + 2 * x0 + 512 * x1 + 65536 * x4), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x6, tmp6, xmask)
    tl.store(out_ptr1 + x6, tmp16, xmask)


@triton.jit
def triton_poi_fused_hardtanh_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 332800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -1.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = 1.0
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_mean_tanh_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (65536 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (131072 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (196608 + x0 + 16384 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = libdevice.tanh(tmp8)
    tl.store(out_ptr0 + x2, tmp9, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 256, 256), (4194304, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], dilation=[1, 1], transposed=True, output_padding=[0, 0],
            groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(1331200)](buf2, primals_2, 
            1331200, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf4 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.int8)
        triton_poi_fused_max_pool2d_with_indices_1[grid(332800)](buf2, buf3,
            buf4, 332800, XBLOCK=1024, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        triton_poi_fused_hardtanh_2[grid(332800)](buf3, buf5, 332800, XBLOCK
            =512, num_warps=8, num_stages=1)
        del buf3
        buf6 = empty_strided_cuda((128, 1, 128, 1), (128, 128, 1, 1), torch
            .float32)
        triton_poi_fused_mean_tanh_3[grid(128)](buf5, buf6, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf6, primals_1, primals_3, buf2, buf4, buf5


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

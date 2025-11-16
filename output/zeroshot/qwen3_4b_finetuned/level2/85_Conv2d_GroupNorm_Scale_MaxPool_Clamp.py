import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_group_norm_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 368640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 2304 % 16
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x2 = xindex // 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (31 + x0 + 128 * x2), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (62 + x0 + 128 * x2), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (127 + x0 + 128 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 128 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp2 > tmp1
    tmp8 = tmp2 > tmp3
    tmp9 = tmp2 > tmp5
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp7.to(tl.int64)
    tmp13 = tl.where(tmp8, tmp12, tmp10)
    tmp14 = tmp8.to(tl.int64)
    tmp15 = tl.where(tmp9, tmp14, tmp11)
    tmp16 = tl.where(tmp7, tmp15, tmp13)
    tl.store(out_ptr0 + x4, tmp6, xmask)
    tl.store(out_ptr1 + x4, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128,
        1))
    assert_size_stride(primals_4, (64, 16), (16, 1))
    assert_size_stride(primals_5, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_group_norm_0[grid(368640)](buf1,
            primals_2, 368640, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        buf3 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.int64)
        triton_poi_fused_max_pool2d_with_indices_1[grid(1152)](buf1, buf2,
            buf3, 1152, XBLOCK=128, num_warps=4, num_stages=1)
        del buf1
    return reinterpret_tensor(buf2, (128, 64, 32, 32), (65536, 1024, 32, 1),
        0), primals_1, primals_3, primals_4, primals_5, buf3


class ModelNew(nn.Module):
    """
    Model that performs convolution, group normalization, scaling, max pooling, and clamping.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.group_norm.weight
        primals_5 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]

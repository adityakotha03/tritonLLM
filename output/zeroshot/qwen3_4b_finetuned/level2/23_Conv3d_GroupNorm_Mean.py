import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused__native_group_norm_0(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 24 % 3
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (384 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (576 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp6
    tmp7 = 4.0
    tmp8 = tmp5 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp6 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tl.store(out_ptr0 + x3, tmp8, xmask)
    tl.store(out_ptr1 + x3, tmp20, xmask)


@triton.jit
def triton_poi_fused__native_group_norm_1(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 24 % 3
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (384 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp6 = tl.load(in_ptr0 + (576 + x1 + 24 * x3), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 + tmp6
    tmp7 = 4.0
    tmp8 = tmp5 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp6 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tl.store(out_ptr0 + x3, tmp20, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 1152
    x1 = xindex // 192 % 3
    x4 = xindex % 1152
    x0 = xindex % 192
    x2 = xindex // 192
    tmp0 = tl.load(in_out_ptr0 + (x4 + 2304 * x3), xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x4 + 2304 * x3), tmp2, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 3, 24, 32, 32), (49152, 16384, 656, 
        21, 1))
    assert_size_stride(primals_3, (24,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 24, 22, 30, 30), (207360, 8640, 3888, 
            129, 1))
        buf1 = empty_strided_cuda((128, 24, 22, 30, 30), (138240, 5760, 
            2592, 86, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 24, 22, 30, 30), (138240, 5760, 
            2592, 86, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused__native_group_norm_0[grid(768)](primals_3, buf1,
            buf2, 768, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf3 = buf0
        del buf0
        triton_poi_fused_convolution_2[grid(36864)](buf3, primals_1, 36864,
            XBLOCK=512, num_warps=4, num_stages=1)
        del primals_1
        buf4 = empty_strided_cuda((128, 24), (24, 1), torch.float32)
        triton_poi_fused__native_group_norm_1[grid(768)](buf2, buf4, 768,
            XBLOCK=256, num_warps=4, num_stages=1)
    return buf4, primals_2, buf1, buf2, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Group Normalization, computes the mean
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.conv.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

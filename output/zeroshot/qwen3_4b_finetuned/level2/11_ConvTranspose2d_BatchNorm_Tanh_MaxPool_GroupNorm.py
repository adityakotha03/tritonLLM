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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 530848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 8192 % 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp1 - tmp0
    tmp10 = tmp9 * tmp9
    tmp11 = tmp3 - tmp2
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp5 - tmp4
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 / tmp7
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused_native_batch_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16 % 16
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 64 * x1 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 64 * x1 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2 * x0 + 64 * x1 + 1024 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2 * x0 + 64 * x1 + 1024 * x2), xmask,
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
    tl.store(out_ptr0 + x3, tmp6, xmask)
    tl.store(out_ptr1 + x3, tmp16, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_4(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4096 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1024 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (3072 + 4096 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp1 - tmp0
    tmp10 = tmp9 * tmp9
    tmp11 = tmp3 - tmp2
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp5 - tmp4
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16 / tmp7
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr0 + x0, tmp8, xmask)
    tl.store(out_ptr1 + x0, tmp20, xmask)


@triton.jit
def triton_poi_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr5 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 * tmp10
    tl.store(out_ptr0 + x2, tmp9, xmask)


@triton.jit
def triton_poi_fused_tanh_6(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7) = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (512, 64, 32, 32), (65536, 1024, 32, 1))
    assert_size_stride(primals_4, (128, 128), (128, 1))
    assert_size_stride(primals_5, (128,), (1,))
    assert_size_stride(primals_6, (128,), (1,))
    assert_size_stride(primals_7, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], dilation=[1, 1], transposed=True, output_padding=[0, 0],
            groups=8, bias=None)
        buf1 = buf0
        del buf0
        buf2 = buf1
        del buf1
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(530848)](buf2, primals_2, 
            530848, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf3 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.
            float32)
        buf4 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128),
            torch.float32)
        triton_poi_fused_native_batch_norm_1[grid(512)](buf2, buf3, buf4, 
            512, XBLOCK=256, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((512, 128, 32, 32), (131072, 1, 4096, 128
            ), torch.float32)
        triton_poi_fused_native_batch_norm_2[grid(2048)](buf2, buf3, buf4,
            primals_4, primals_5, buf5, 2048, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_5
        buf6 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128),
            torch.float32)
        buf7 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128),
            torch.int8)
        triton_poi_fused_max_pool2d_with_indices_3[grid(131072)](buf5, buf6,
            buf7, 131072, XBLOCK=512, num_warps=8, num_stages=1)
        buf8 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 1, 1), torch.
            float32)
        buf9 = empty_strided_cuda((512, 128, 1, 1), (128, 1, 128, 128),
            torch.float32)
        triton_poi_fused_native_group_norm_4[grid(512)](buf6, buf8, buf9, 
            512, XBLOCK=256, num_warps=4, num_stages=1)
        buf10 = empty_strided_cuda((512, 128, 16, 16), (32768, 1, 2048, 128
            ), torch.float32)
        triton_poi_fused_native_group_norm_5[grid(1024)](buf6, buf8, buf9,
            primals_6, primals_7, primals_4, buf10, 1024, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf8
        del buf9
        del primals_7
        buf11 = buf5
        del buf5
        triton_poi_fused_tanh_6[grid(131072)](buf10, buf11, 131072, XBLOCK
            =512, num_warps=4, num_stages=1)
        del buf10
    return (buf11, primals_1, primals_3, primals_4, buf2, buf3, buf4, buf6,
        buf7, primals_6)


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, batch normalization, tanh activation, max pooling, and group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_4 = self.batch_norm.weight
        primals_5 = self.batch_norm.bias
        primals_6 = self.group_norm.weight
        primals_7 = self.group_norm.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7])
        return output[0]

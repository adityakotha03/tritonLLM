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
def triton_per_fused__native_batch_norm_legit_convolution_relu_0(in_out_ptr0,
    in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 160
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 40
    tmp0 = tl.load(in_out_ptr0 + (r2 + 256 * x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 256.0
    tmp21 = tmp18 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tl.store(in_out_ptr0 + (r2 + 256 * x3), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + x3, tmp24, xmask)
    tl.store(out_ptr2 + (r2 + 256 * x3), tmp25, xmask)
    tl.store(out_ptr3 + x3, tmp12, xmask)
    tl.store(out_ptr4 + x3, tmp23, xmask)
    tl.store(out_ptr1 + x3, tmp18, xmask)
    tl.store(out_ptr0 + x3, tmp9, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_1(in_ptr0,
    in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = xindex // 40
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 480 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tmp6 = tl.full([XBLOCK, 1], 160, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp13 = tl.sum(tmp11, 0)[:, None]
    tl.store(out_ptr0 + x2, tmp8, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused__native_batch_norm_legit_convolution_2(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 40
    x1 = xindex // 40
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 480 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp5 = tl.load(in_ptr3 + x2, xmask)
    tmp7 = tl.load(in_ptr4 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 160.0
    tmp8 = tmp5 / tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp4 * tmp11
    tmp13 = tmp12 * tmp7
    tl.store(out_ptr0 + x2, tmp11, xmask)
    tl.store(out_ptr1 + x2, tmp13, xmask)


@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_3(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 400
    x4 = xindex % 400
    x5 = xindex
    x6 = xindex // 100 % 40
    x1 = xindex // 400 % 40
    x2 = xindex // 1600
    x7 = xindex % 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + 400 * x3), xmask)
    tmp2 = tl.load(in_ptr2 + x6, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 + 400 * x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + x7, xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5 - tmp7
    tmp8 = 160.0
    tmp9 = tmp6 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tmp13 * tmp9
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + x5, tmp12, xmask)
    tl.store(out_ptr1 + x5, tmp16, xmask)
    tl.store(out_ptr2 + x5, tmp18, xmask)


@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_4(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1,
    out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 102400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 400
    x4 = xindex % 400
    x5 = xindex
    x6 = xindex // 100 % 40
    x1 = xindex // 400 % 40
    x2 = xindex // 1600
    x7 = xindex % 1600
    tmp0 = tl.load(in_ptr0 + x3, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + 400 * x3), xmask)
    tmp2 = tl.load(in_ptr2 + x6, xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 + 400 * x2), xmask, eviction_policy=
        'evict_last')
    tmp7 = tl.load(in_ptr4 + x1, xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + x7, xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp5 - tmp7
    tmp8 = 160.0
    tmp9 = tmp6 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tmp13 * tmp9
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tl.store(out_ptr0 + x5, tmp12, xmask)
    tl.store(out_ptr1 + x5, tmp16, xmask)
    tl.store(out_ptr2 + x5, tmp18, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15) = args
    args.clear()
    assert_size_stride(primals_1, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_2, (40,), (1,))
    assert_size_stride(primals_3, (10, 240, 224, 224), (11796480, 480, 224,
        1))
    assert_size_stride(primals_4, (40, 40, 3, 3), (360, 9, 3, 1))
    assert_size_stride(primals_5, (40,), (1,))
    assert_size_stride(primals_6, (40,), (1,))
    assert_size_stride(primals_7, (480, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_8, (480,), (1,))
    assert_size_stride(primals_9, (480, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_10, (480,), (1,))
    assert_size_stride(primals_11, (480,), (1,))
    assert_size_stride(primals_12, (480, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_13, (480,), (1,))
    assert_size_stride(primals_14, (10, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_15, (10,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (10, 40, 224, 224), (1975680, 49392, 224, 1))
        buf1 = reinterpret_tensor(buf0, (10, 40, 224, 224), (1975680, 49392,
            224, 1), 0)
        del buf0
        buf2 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf3 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf4 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf5 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf7 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf8 = reinterpret_tensor(buf7, (1, 160, 1, 1), (160, 1, 1, 1), 0)
        del buf7
        buf9 = empty_strided_cuda((1, 160, 1, 1), (160, 1, 1, 1), torch.float32
            )
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_relu_0[grid(160)
            ](buf1, buf8, primals_2, buf2, buf3, buf4, buf5, buf6, buf9, 
            160, 256, XBLOCK=8, num_warps=4, num_stages=1)
        del primals_2
        buf10 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf10, (10, 40, 224, 224), (1975680, 49392, 224, 1))
        buf11 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf13 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_convolution_1[grid(160)](
            buf10, primals_5, buf11, buf12, 160, XBLOCK=128, num_warps=4,
            num_stages=1)
        del primals_5
        buf16 = buf10
        del buf10
        buf17 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        buf18 = empty_strided_cuda((1, 160), (160, 1), torch.float32)
        triton_poi_fused__native_batch_norm_legit_convolution_2[grid(160)](
            buf16, primals_6, buf11, buf12, buf13, buf17, buf18, 160,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf11
        del buf12
        del primals_6
        buf19 = extern_kernels.convolution(buf1, primals_7, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf19, (10, 480, 224, 224), (23596800, 480, 224, 
            1))
        buf20 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.float32)
        buf21 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.float32)
        buf22 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.bool)
        triton_poi_fused_add_convolution_relu_threshold_backward_3[grid(102400)
            ](buf17, buf16, primals_8, buf15, buf14, primals_10, buf20,
            buf21, buf22, 102400, XBLOCK=512, num_warps=8, num_stages=1)
        del buf14
        del buf15
        del buf16
        del buf17
        del primals_10
        buf23 = extern_kernels.convolution(buf20, primals_9, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf23, (10, 480, 224, 224), (23596800, 480, 224, 
            1))
        buf24 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.float32)
        buf25 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.float32)
        buf26 = empty_strided_cuda((10, 40, 224, 224), (1975680, 49392, 224,
            1), torch.bool)
        triton_poi_fused_add_convolution_relu_threshold_backward_4[grid(102400)
            ](buf18, buf23, primals_13, buf13, primals_11, primals_14,
            buf24, buf25, buf26, 102400, XBLOCK=512, num_warps=8, num_stages=1)
        del buf13
        del buf18
        del primals_11
        del primals_13
        del primals_14
    return (buf25, primals_1, primals_3, primals_4, primals_7, primals_8,
        primals_9, primals_12, reinterpret_tensor(buf1, (10, 40, 224, 224),
        (1975680, 49392, 224, 1), 0), buf19, buf20, buf23, buf22, buf26,
        reinterpret_tensor(buf2, (1, 160), (160, 1), 0), reinterpret_tensor(
        buf3, (1, 160), (160, 1), 0), reinterpret_tensor(buf4, (1, 160), (
        160, 1), 0), reinterpret_tensor(buf5, (1, 160), (160, 1), 0),
        reinterpret_tensor(buf6, (1, 160), (160, 1), 0), reinterpret_tensor(
        buf9, (1, 160, 1, 1), (160, 1, 1, 1), 0))


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param groups: Number of groups for group convolution.
        """
        super(ModelNew, self).__init__()
        
        # Ensure the output channels are divisible by groups
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
        # Shortcut connection if input and output channels are the same
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.bn1.weight
        primals_5 = self.bn1.bias
        primals_4 = self.conv2.weight
        primals_6 = self.bn2.weight
        primals_8 = self.bn2.bias
        primals_7 = self.conv3.weight
        primals_10 = self.bn3.weight
        primals_13 = self.bn3.bias
        primals_9 = self.shortcut.conv2d.weight
        primals_11 = self.shortcut.conv2d.bias
        primals_12 = self.shortcut.conv2d.weight
        primals_14 = self.shortcut.conv2d.bias
        primals_15 = self.shortcut.conv2d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15])
        return output[0]

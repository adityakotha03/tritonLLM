import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + x2, tmp4, xmask)
    tl.store(out_ptr0 + x2, tmp6, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 138240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK
    : tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 138240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_add_convolution_relu_threshold_backward_5(in_ptr0,
    in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask)
    tmp4 = tl.load(in_ptr3 + x1, xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp11 = 0.0
    tmp12 = tmp10 <= tmp11
    tl.store(out_ptr0 + x2, tmp10, xmask)
    tl.store(out_ptr1 + x2, tmp12, xmask)


@triton.jit
def triton_poi_fused_convolution_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 138240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        """
        Channel shuffle operation.

        :param groups: Number of groups for shuffling.
        """
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        """
        Forward pass for channel shuffle.

        :param x: Input tensor, shape (batch_size, channels, height, width)
        :return: Output tensor, shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose
        x = x.transpose(1, 2).contiguous()
        
        # Flatten
        x = x.view(batch_size, -1, height, width)
        
        return x


@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 138240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1105920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1440 % 16
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23, primals_24) = args
    args.clear()
    assert_size_stride(primals_1, (16, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (10, 16, 224, 224), (394240, 240, 1, 1))
    assert_size_stride(primals_4, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_5, (16,), (1,))
    assert_size_stride(primals_6, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_7, (16, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (16,), (1,))
    assert_size_stride(primals_9, (480, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_10, (480,), (1,))
    assert_size_stride(primals_11, (10, 480, 224, 224), (107520, 240, 1, 1))
    assert_size_stride(primals_12, (10,), (1,))
    assert_size_stride(primals_13, (10, 10, 224, 224), (491520, 240, 1, 1))
    assert_size_stride(primals_14, (10,), (1,))
    assert_size_stride(primals_15, (10, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_16, (10,), (1,))
    assert_size_stride(primals_17, (10, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_18, (10,), (1,))
    assert_size_stride(primals_19, (10, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_20, (10,), (1,))
    assert_size_stride(primals_21, (10, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_22, (10,), (1,))
    assert_size_stride(primals_23, (10, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_24, (10,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_1[0](buf0, primals_1, 138240, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_relu_2[0](buf1, primals_2, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_3[0](buf2, primals_4, 138240, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_relu_2[0](buf3, primals_5, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf4 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_4[0](buf4, primals_6, 1105920, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_6
        buf5 = empty_strided_cuda((10, 16, 224, 224), (394240, 1, 1576, 6),
            torch.float32)
        triton_poi_fused_convolution_relu_2[0](buf5, primals_7, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_7
        buf6 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_add_convolution_relu_threshold_backward_5[0](
            buf6, primals_8, buf1, buf2, buf3, buf4, buf5, 1105920, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del primals_8
        buf7 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.bool)
        triton_poi_fused_add_convolution_relu_threshold_backward_5[0](
            buf6, primals_9, buf1, buf2, buf3, buf4, buf5, 1105920, XBLOCK=
            1024, num_warps=4, num_stages=1, XBLOCK=1024)
        del primals_9
        buf8 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_6[0](buf8, primals_10, 138240, XBLOCK=
            1024, num_warps=4, num_stages=1)
        del primals_10
        buf9 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_7[0](buf9, primals_11, 1105920, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_11
        buf10 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_8[0](buf10, primals_12, 138240, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_12
        buf11 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_9[0](buf11, primals_13, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_13
        buf12 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.bool)
        triton_poi_fused_add_convolution_relu_threshold_backward_5[0](
            buf12, primals_14, buf9, buf10, buf11, primals_15, primals_16,
            1105920, XBLOCK=1024, num_warps=4, num_stages=1, XBLOCK=1024)
        del primals_14
        buf13 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_6[0](buf13, primals_17, 138240, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_17
        buf14 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_7[0](buf14, primals_18, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_18
        buf15 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_8[0](buf15, primals_19, 138240, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_19
        buf16 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_9[0](buf16, primals_20, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_20
        buf17 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.bool)
        triton_poi_fused_add_convolution_relu_threshold_backward_5[0](
            buf17, primals_21, buf14, buf15, buf16, primals_22, primals_23,
            1105920, XBLOCK=1024, num_warps=4, num_stages=1, XBLOCK=1024)
        del primals_21
        buf18 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_6[0](buf18, primals_24, 138240, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_24
        buf19 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_7[0](buf19, primals_14, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_14
        buf20 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_8[0](buf20, primals_16, 138240, XBLOCK
            =1024, num_warps=4, num_stages=1)
        del primals_16
        buf21 = empty_strided_cuda((10, 480, 224, 224), (107520, 1, 476, 2),
            torch.float32)
        triton_poi_fused_convolution_9[0](buf21, primals_18, 1105920,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_18
    return (buf17, buf19, buf20, primals_3, primals_15, primals_22, primals_13,
        primals_11, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8,
        buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18,
        buf19, buf20, buf21)


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
        primals_2 = self.conv1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.conv2.bias
        primals_6 = self.conv2.weight
        primals_7 = self.conv2.bias
        primals_9 = self.conv3.weight
        primals_10 = self.conv3.bias
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.conv2.bias
        primals_6 = self.conv2.weight
        primals_7 = self.conv2.bias
        primals_9 = self.conv3.weight
        primals_10 = self.conv3.bias
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_10 = self.conv3.bias
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_10 = self.conv3.bias
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_10 = self.conv3.bias
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2.num_batches_tracked
        primals_11 = self.bn3.weight
        primals_12 = self.bn3.bias
        primals_13 = self.bn3.running_mean
        primals_14 = self.bn3.running_var
        primals_15 = self.bn3.num_batches_tracked
        primals_16 = self.bn2.weight
        primals_17 = self.bn2.bias
        primals_18 = self.bn2.running_mean
        primals_19 = self.bn2.running_var
        primals_20 = self.bn2
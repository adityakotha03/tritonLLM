import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 576 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 576
    x1 = xindex // 576
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 2304 * x1), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 2304 * x1), None,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1152 + 2 * x0 + 2304 * x1), None,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1153 + 2 * x0 + 2304 * x1), None,
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
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x2, tmp16, None)


@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 144 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 144
    x1 = xindex // 144
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 576 * x1), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 576 * x1), None, eviction_policy
        ='evict_last')
    tmp3 = tl.load(in_ptr0 + (288 + 2 * x0 + 576 * x1), None,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (289 + 2 * x0 + 576 * x1), None,
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
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x2, tmp16, None)


@triton.jit
def triton_poi_fused_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 144 % 384
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 144 % 384
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = xindex // 144 % 256
    tmp0 = tl.load(in_out_ptr0 + x3, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x3, tmp4, None)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_7(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 576
    x1 = xindex // 576
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 1152 * x1), None, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 1152 * x1), None,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + 2 * x0 + 1152 * x1), None,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (577 + 2 * x0 + 1152 * x1), None,
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
    tl.store(out_ptr0 + x2, tmp6, None)
    tl.store(out_ptr1 + x2, tmp16, None)


@triton.jit
def triton_poi_fused_relu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, None)


@triton.jit
def triton_poi_fused_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, None)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21) = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (96,), (1,))
    assert_size_stride(primals_3, (1024, 3, 224, 224), (150528, 50176, 224,
        1))
    assert_size_stride(primals_4, (256, 96, 5, 5), (2400, 25, 5, 1))
    assert_size_stride(primals_5, (256,), (1,))
    assert_size_stride(primals_6, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_7, (384,), (1,))
    assert_size_stride(primals_8, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_9, (384,), (1,))
    assert_size_stride(primals_10, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_11, (256,), (1,))
    assert_size_stride(primals_12, (4096, 2304), (2304, 1))
    assert_size_stride(primals_13, (4096,), (1,))
    assert_size_stride(primals_14, (4096, 4096), (4096, 1))
    assert_size_stride(primals_15, (4096,), (1,))
    assert_size_stride(primals_16, (1000, 4096), (4096, 1))
    assert_size_stride(primals_17, (1000,), (1,))
    assert_size_stride(primals_18, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_19, (256,), (1,))
    assert_size_stride(primals_20, (384, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_21, (384,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(4, 
            4), padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1024, 96, 57, 57), (314688, 3276, 57, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(32212032)](buf1,
            primals_2, 32212032, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((1024, 96, 28, 28), (75264, 784, 28, 1),
            torch.float32)
        buf3 = empty_strided_cuda((1024, 96, 28, 28), (75264, 784, 28, 1),
            torch.int8)
        triton_poi_fused_max_pool2d_with_indices_1[grid(7864320)](buf1,
            buf2, buf3, 7864320, XBLOCK=512, num_warps=8, num_stages=1)
        buf4 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1),
            padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (1024, 256, 28, 28), (200704, 784, 28, 1))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_2[grid(20579584)](buf5,
            primals_5, 20579584, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_5
        buf6 = empty_strided_cuda((1024, 256, 14, 14), (50176, 196, 14, 1),
            torch.float32)
        buf7 = empty_strided_cuda((1024, 256, 14, 14), (50176, 196, 14, 1),
            torch.int8)
        triton_poi_fused_max_pool2d_with_indices_3[grid(516064)](buf5,
            buf6, buf7, 516064, XBLOCK=512, num_warps=8, num_stages=1)
        buf8 = extern_kernels.convolution(buf6, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1024, 384, 14, 14), (75264, 196, 14, 1))
        buf9 = buf8
        del buf8
        triton_poi_fused_convolution_relu_4[grid(7705664)](buf9,
            primals_7, 7705664, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_7
        buf10 = extern_kernels.convolution(buf9, primals_8, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1024, 384, 14, 14), (75264, 196, 14, 1))
        buf11 = buf10
        del buf10
        triton_poi_fused_convolution_relu_5[grid(7705664)](buf11,
            primals_9, 7705664, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_9
        buf12 = extern_kernels.convolution(buf11, primals_10, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1024, 256, 14, 14), (50176, 196, 14, 1))
        buf13 = buf12
        del buf12
        triton_poi_fused_convolution_relu_6[grid(516064)](buf13,
            primals_11, 516064, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_11
        buf14 = empty_strided_cuda((1024, 256, 7, 7), (12544, 49, 7, 1),
            torch.float32)
        buf15 = empty_strided_cuda((1024, 256, 7, 7), (12544, 49, 7, 1),
            torch.int8)
        triton_poi_fused_max_pool2d_with_indices_7[grid(131072)](buf13,
            buf14, buf15, 131072, XBLOCK=1024, num_warps=4, num_stages=1)
        buf16 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(buf14, (1024, 2304), (2304, 1),
            0), reinterpret_tensor(primals_12, (2304, 4096), (1, 2304), 0),
            out=buf16)
        buf17 = buf16
        del buf16
        triton_poi_fused_relu_8[grid(4194304)](buf17, primals_13, 4194304,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_13
        buf18 = empty_strided_cuda((1024, 4096), (4096, 1), torch.float32)
        extern_kernels.mm(buf17, reinterpret_tensor(primals_14, (4096, 4096
            ), (1, 4096), 0), out=buf18)
        buf19 = buf18
        del buf18
        triton_poi_fused_relu_9[grid(4194304)](buf19, primals_15, 4194304,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_15
        buf20 = empty_strided_cuda((1024, 1000), (1000, 1), torch.float32)
        extern_kernels.addmm(primals_17, buf19, reinterpret_tensor(
            primals_16, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf20
            )
        del primals_17
        buf21 = extern_kernels.convolution(buf13, primals_18, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (1024, 256, 14, 14), (50176, 196, 14, 1))
        buf22 = buf21
        del buf21
        triton_poi_fused_convolution_relu_4[grid(516064)](buf22,
            primals_19, 516064, XBLOCK=512, num_warps=8, num_stages=1)
        del primals_19
        buf23 = extern_kernels.convolution(buf22, primals_20, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (1024, 384, 14, 14), (75264, 196, 14, 1))
        buf24 = buf23
        del buf23
        triton_poi_fused_convolution_relu_5[grid(7705664)](buf24,
            primals_21, 7705664, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_21
    return (buf20, primals_1, primals_3, primals_4, primals_6, primals_8,
        primals_10, primals_18, primals_20, buf1, buf2, buf3, buf5, buf6,
        buf7, buf9, buf11, buf13, buf14, buf15, buf17, buf19, buf22, buf24,
        primals_16, primals_14, primals_12)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.conv2.bias
        primals_6 = self.conv3.weight
        primals_7 = self.conv3.bias
        primals_8 = self.conv4.weight
        primals_9 = self.conv4.bias
        primals_10 = self.conv5.weight
        primals_11 = self.conv5.bias
        primals_12 = self.fc1.weight
        primals_13 = self.fc1.bias
        primals_14 = self.fc2.weight
        primals_15 = self.fc2.bias
        primals_16 = self.fc3.weight
        primals_17 = self.fc3.bias
        primals_18 = self.conv1.weight
        primals_19 = self.conv1.bias
        primals_20 = self.conv2.weight
        primals_21 = self.conv2.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21])
        return output[0]

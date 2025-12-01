import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17, primals_18
        ) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (2, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_6, (64,), (1,))
    assert_size_stride(primals_7, (64,), (1,))
    assert_size_stride(primals_8, (128,), (1,))
    assert_size_stride(primals_9, (128,), (1,))
    assert_size_stride(primals_10, (128,), (1,))
    assert_size_stride(primals_11, (128,), (1,))
    assert_size_stride(primals_12, (128,), (1,))
    assert_size_stride(primals_13, (128,), (1,))
    assert_size_stride(primals_14, (256,), (1,))
    assert_size_stride(primals_15, (256,), (1,))
    assert_size_stride(primals_16, (256,), (1,))
    assert_size_stride(primals_17, (256,), (1,))
    assert_size_stride(primals_18, (512,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_5, primals_1, stride=(2, 
            2), padding=(3, 3), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (2, 64, 112, 112), (78400, 1225, 112, 1))
        buf1 = buf0
        del buf0
        buf2 = extern_kernels.convolution(primals_5, primals_1, stride=(2, 
            2), padding=(3, 3), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del primals_1
        buf3 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        get_raw_buf = torch._C._dynamo.guards._get_raw_buf
        buf4 = get_raw_buf(buf3, 0, 1, 0)
        buf5 = get_raw_buf(buf3, 0, 0, 0)
        buf6 = get_raw_buf(buf3, 0, 1, 1)
        buf7 = get_raw_buf(buf3, 0, 0, 1)
        extern_kernels.addmm(primals_3, buf2, reinterpret_tensor(primals_2,
            (64, 64), (1, 0), 0), alpha=1, beta=1, out=buf4)
        del primals_2
        buf8 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf9 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf10 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf11 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf12 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_0[grid(49152)](buf1, primals_3, buf8, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        triton_poi_fused_add_0[grid(49152)](buf1, primals_4, buf9, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf13 = extern_kernels.convolution(buf8, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf8
        buf14 = extern_kernels.convolution(buf9, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf9
        buf15 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf16 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_1[grid(18432)](buf13, primals_7, buf15, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf13
        del primals_7
        triton_poi_fused_add_1[grid(18432)](buf14, primals_8, buf16, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf14
        del primals_8
        buf17 = extern_kernels.convolution(buf15, primals_9, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf15
        buf18 = extern_kernels.convolution(buf16, primals_9, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf16
        buf19 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf20 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_0[grid(49152)](buf17, primals_10, buf19, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf17
        del primals_10
        triton_poi_fused_add_0[grid(49152)](buf18, primals_11, buf20, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf18
        del primals_11
        buf21 = extern_kernels.convolution(buf19, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf19
        buf22 = extern_kernels.convolution(buf20, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf20
        buf23 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf24 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_1[grid(18432)](buf21, primals_13, buf23, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf21
        del primals_13
        triton_poi_fused_add_1[grid(18432)](buf22, primals_14, buf24, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf22
        del primals_14
        buf25 = extern_kernels.convolution(buf23, primals_15, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf23
        buf26 = extern_kernels.convolution(buf24, primals_15, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf24
        buf27 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf28 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_0[grid(49152)](buf25, primals_16, buf27, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf25
        del primals_16
        triton_poi_fused_add_0[grid(49152)](buf26, primals_17, buf28, 49152,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf26
        del primals_17
        buf29 = extern_kernels.convolution(buf27, primals_18, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf27
        buf30 = extern_kernels.convolution(buf28, primals_18, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        del buf28
        buf31 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        buf32 = empty_strided_cuda((2, 64, 112, 112), (78400, 1225, 112, 1),
            torch.float32)
        triton_poi_fused_add_1[grid(18432)](buf29, primals_18, buf31, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf29
        del primals_18
        triton_poi_fused_add_1[grid(18432)](buf30, primals_18, buf32, 18432,
            XBLOCK=128, num_warps=4, num_stages=1)
        del buf30
    return (buf32, primals_12, primals_15, primals_18, buf31, buf1, buf2,
        buf3, buf4, buf5, buf6, buf7, buf11, buf12, buf22, buf24, buf26,
        buf28)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: Number of output classes
        """
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, input_0):
        primals_1 = self.conv1.weight
        primals_2 = self.bn1.weight
        primals_3 = self.bn1.bias
        primals_4 = self.bn1.running_mean
        primals_5 = self.bn1.running_var
        primals_6 = self.layer1[0].conv1.weight
        primals_7 = self.layer1[0].bn1.weight
        primals_8 = self.layer1[0].bn1.bias
        primals_9 = self.layer1[0].bn1.running_mean
        primals_10 = self.layer1[0].bn1.running_var
        primals_11 = self.layer1[0].conv2.weight
        primals_12 = self.layer1[0].bn2.weight
        primals_13 = self.layer1[0].bn2.bias
        primals_14 = self.layer1[0].bn2.running_mean
        primals_15 = self.layer1[0].bn2.running_var
        primals_16 = self.layer1[1].conv1.weight
        primals_17 = self.layer1[1].bn1.weight
        primals_18 = self.layer1[1].bn1.bias
        primals_19 = self.layer1[1].bn1.running_mean
        primals_20 = self.layer1[1].bn1.running_var
        primals_21 = self.layer1[1].conv2.weight
        primals_22 = self.layer1[1].bn2.weight
        primals_23 = self.layer1[1].bn2.bias
        primals_24 = self.layer1[1].bn2.running_mean
        primals_25 = self.layer1[1].bn2.running_var
        primals_26 = self.layer2[0].conv1.weight
        primals_27 = self.layer2[0].bn1.weight
        primals_28 = self.layer2[0].bn1.bias
        primals_29 = self.layer2[0].bn1.running_mean
        primals_30 = self.layer2[0].bn1.running_var
        primals_31 = self.layer2[0].conv2.weight
        primals_32 = self.layer2[0].bn2.weight
        primals_33 = self.layer2[0].bn2.bias
        primals_34 = self.layer2[0].bn2.running_mean
        primals_35 = self.layer2[0].bn2.running_var
        primals_36 = self.layer2[1].conv1.weight
        primals_37 = self.layer2[1].bn1.weight
        primals_38 = self.layer2[1].bn1.bias
        primals_39 = self.layer2[1].bn1.running_mean
        primals_40 = self.layer2[1].bn1.running_var
        primals_41 = self.layer2[1].conv2.weight
        primals_42 = self.layer2[1].bn2.weight
        primals_43 = self.layer2[1].bn2.bias
        primals_44 = self.layer2[1].bn2.running_mean
        primals_45 = self.layer2[1].bn2.running_var
        primals_46 = self.layer3[0].conv1.weight
        primals_47 = self.layer3[0].bn1.weight
        primals_48 = self.layer3[0].bn1.bias
        primals_49 = self.layer3[0].bn1.running_mean
        primals_50 = self.layer3[0].bn1.running_var
        primals_51 = self.layer3[0].conv2.weight
        primals_52 = self.layer3[0].bn2.weight
        primals_53 = self.layer3[0].bn2.bias
        primals_54 = self.layer3[0].bn2.running_mean
        primals_55 = self.layer3[0].bn2.running_var
        primals_56 = self.layer3[1].conv1.weight
        primals_57 = self.layer3[1].bn1.weight
        primals_58 = self.layer3[1].bn1.bias
        primals_59 = self.layer3[1].bn1.running_mean
        primals_60 = self.layer3[1].bn1.running_var
        primals_61 = self.layer3[1].conv2.weight
        primals_62 = self.layer3[1].bn2.weight
        primals_63 = self.layer3[1].bn2.bias
        primals_64 = self.layer3[1].bn2.running_mean
        primals_65 = self.layer3[1].bn2.running_var
        primals_66 = self.layer4[0].conv1.weight
        primals_67 = self.layer4[0].bn1.weight
        primals_68 = self.layer4[0].bn1.bias
        primals_69 = self.layer4[0].bn1.running_mean
        primals_70 = self.layer4[0].bn1.running_var
        primals_71 = self.layer4[0].conv2.weight
        primals_72 = self.layer4[0].bn2.weight
        primals_73 = self.layer4[0].bn2.bias
        primals_74 = self.layer4[0].bn2.running_mean
        primals_75 = self.layer4[0].bn2.running_var
        primals_76 = self.layer4[1].conv1.weight
        primals_77 = self.layer4[1].bn1.weight
        primals_78 = self.layer4[1].bn1.bias
        primals_79 = self.layer4[1].bn1.running_mean
        primals_80 = self.layer4[1].bn1.running_var
        primals_81 = self.layer4[1].conv2.weight
        primals_82 = self.layer4[1].bn2.weight
        primals_83 = self.layer4[1].bn2.bias
        primals_84 = self.layer4[1].bn2.running_mean
        primals_85 = self.layer4[1].bn2.running_var
        primals_86 = self.fc.weight
        primals_87 = self.fc.bias
        output = call([primals_1, primals_2, primals_3, primals_4,
            input_0, primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23, primals_24,
            primals_25, primals_26, primals_27, primals_28, primals_29,
            primals_30, primals_31, primals_32, primals_33, primals_34,
            primals_35, primals_36, primals_37, primals_38, primals_39,
            primals_40, primals_41, primals_42, primals_43, primals_44,
            primals_45, primals_46, primals_47, primals_48, primals_49,
            primals_50, primals_51, primals_52, primals_53, primals_54,
            primals_55, primals_56, primals_57, primals_58, primals_59,
            primals_60, primals_61, primals_62, primals_63, primals_64,
            primals_65, primals_66, primals_67, primals_68, primals_69,
            primals_70, primals_71, primals_72, primals_73, primals_74,
            primals_75, primals_76, primals_77, primals_78, primals_79,
            primals_80, primals_81, primals_82, primals_83, primals_84,
            primals_85, primals_86, primals_87])
        return output[0]
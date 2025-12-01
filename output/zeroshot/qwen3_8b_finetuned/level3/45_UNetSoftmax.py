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
def triton_poi_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (8, 8, 64, 512), (262144, 32768, 512, 1))
    assert_size_stride(primals_2, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64,), (1,))
    assert_size_stride(primals_6, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (128,), (1,))
    assert_size_stride(primals_8, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_9, (256,), (1,))
    assert_size_stride(primals_10, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_11, (512,), (1,))
    assert_size_stride(primals_12, (1024, 512, 3, 3), (4608, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf2 = extern_kernels.max_pool2d(buf1, kernel_size=(2, 2),
            stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=True)
        assert_size_stride(buf2, (8, 64, 32, 32), (65536, 1024, 32, 1))
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf4 = extern_kernels.convolution(buf3, primals_5, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf5 = extern_kernels.max_pool2d(buf4, kernel_size=(2, 2),
            stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=True)
        assert_size_stride(buf5, (8, 128, 16, 16), (32768, 256, 16, 1))
        buf6 = extern_kernels.convolution(buf5, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf7 = extern_kernels.convolution(buf6, primals_7, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf8 = extern_kernels.max_pool2d(buf7, kernel_size=(2, 2),
            stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=True)
        assert_size_stride(buf8, (8, 256, 8, 8), (16384, 64, 8, 1))
        buf9 = extern_kernels.convolution(buf8, primals_8, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf10 = extern_kernels.convolution(buf9, primals_9, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf11 = extern_kernels.convolution(buf10, primals_10, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf12 = torch.ops.aten.cat.default([buf11, buf6], dim=1)
        assert_size_stride(buf12, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf13 = extern_kernels.convolution(buf12, primals_11, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf14 = extern_kernels.convolution(buf13, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf15 = extern_kernels.convolution(buf14, primals_10, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf16 = torch.ops.aten.cat.default([buf15, buf7], dim=1)
        assert_size_stride(buf16, (8, 1024, 32, 32), (1048576, 1024, 32, 1))
        buf17 = extern_kernels.convolution(buf16, primals_11, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 1024, 32, 32), (1048576, 1024, 32, 1))
        buf18 = extern_kernels.convolution(buf17, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 1024, 32, 32), (1048576, 1024, 32, 1))
        buf19 = extern_kernels.convolution(buf18, primals_10, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 512, 64, 64), (2097152, 4096, 64, 1))
        buf20 = torch.ops.aten.cat.default([buf19, buf4], dim=1)
        assert_size_stride(buf20, (8, 1024, 64, 64), (4194304, 4096, 64, 1))
        buf21 = extern_kernels.convolution(buf20, primals_11, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 1024, 64, 64), (4194304, 4096, 64, 1))
        buf22 = extern_kernels.convolution(buf21, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 1024, 64, 64), (4194304, 4096, 64, 1))
        buf23 = extern_kernels.convolution(buf22, primals_10, stride=(2, 2),
            padding=(0, 0), dilation=(1, 1), transposed=True,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 512, 128, 128), (8388608, 8192, 128, 1))
        buf24 = torch.ops.aten.cat.default([buf23, buf1], dim=1)
        assert_size_stride(buf24, (8, 1024, 128, 128), (134217728, 131072, 
            128, 1))
        buf25 = extern_kernels.convolution(buf24, primals_11, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 1024, 128, 128), (134217728, 131072, 
            128, 1))
        buf26 = extern_kernels.convolution(buf25, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 1024, 128, 128), (134217728, 131072, 
            128, 1))
        buf27 = empty_strided_cuda((8, 4, 128, 128), (65536, 16384, 128, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(131072)](buf26, primals_11, primals_12,
            buf27, 131072, XBLOCK=128, num_warps=4, num_stages=1)
        del buf26
        del primals_12
    return (buf27, primals_1, primals_2, primals_3, primals_4, primals_5,
        primals_6, primals_7, primals_8, primals_9, primals_10, primals_11,
        buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10,
        buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20,
        buf21, buf22, buf23, buf24, buf25)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.double_conv(x)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of base features (will be doubled in each layer)
        """
        super(ModelNew).__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, input_0):
        primals_2 = self.encoder1.double_conv[0].weight
        primals_3 = self.encoder1.double_conv[0].bias
        primals_4 = self.encoder1.double_conv[2].weight
        primals_5 = self.encoder1.double_conv[2].bias
        primals_6 = self.encoder2.double_conv[0].weight
        primals_7 = self.encoder2.double_conv[0].bias
        primals_8 = self.encoder2.double_conv[2].weight
        primals_9 = self.encoder2.double_conv[2].bias
        primals_10 = self.encoder3.double_conv[0].weight
        primals_11 = self.encoder3.double_conv[0].bias
        primals_12 = self.encoder3.double_conv[2].weight
        primals_13 = self.encoder3.double_conv[2].bias
        primals_14 = self.encoder4.double_conv[0].weight
        primals_15 = self.encoder4.double_conv[0].bias
        primals_16 = self.encoder4.double_conv[2].weight
        primals_17 = self.encoder4.double_conv[2].bias
        primals_18 = self.bottleneck.double_conv[0].weight
        primals_19 = self.bottleneck.double_conv[0].bias
        primals_20 = self.bottleneck.double_conv[2].weight
        primals_21 = self.bottleneck.double_conv[2].bias
        primals_22 = self.upconv4.weight
        primals_23 = self.upconv4.bias
        primals_24 = self.decoder4.double_conv[0].weight
        primals_25 = self.decoder4.double_conv[0].bias
        primals_26 = self.decoder4.double_conv[2].weight
        primals_27 = self.decoder4.double_conv[2].bias
        primals_28 = self.upconv3.weight
        primals_29 = self.upconv3.bias
        primals_30 = self.decoder3.double_conv[0].weight
        primals_31 = self.decoder3.double_conv[0].bias
        primals_32 = self.decoder3.double_conv[2].weight
        primals_33 = self.decoder3.double_conv[2].bias
        primals_34 = self.upconv2.weight
        primals_35 = self.upconv2.bias
        primals_36 = self.decoder2.double_conv[0].weight
        primals_37 = self.decoder2.double_conv[0].bias
        primals_38 = self.decoder2.double_conv[2].weight
        primals_39 = self.decoder2.double_conv[2].bias
        primals_40 = self.upconv1.weight
        primals_41 = self.upconv1.bias
        primals_42 = self.decoder1.double_conv[0].weight
        primals_43 = self.decoder1.double_conv[0].bias
        primals_44 = self.decoder1.double_conv[2].weight
        primals_45 = self.decoder1.double_conv[2].bias
        primals_46 = self.final_conv.weight
        primals_47 = self.final_conv.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23, primals_24,
            primals_25, primals_26, primals_27, primals_28, primals_29,
            primals_30, primals_31, primals_32, primals_33, primals_34,
            primals_35, primals_36, primals_37, primals_38, primals_39,
            primals_40, primals_41, primals_42, primals_43, primals_44,
            primals_45, primals_46, primals_47])
        return output[0]
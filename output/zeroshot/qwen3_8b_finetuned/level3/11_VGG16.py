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
    xnumel = 5408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16 % 64
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + x2, tmp4, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = xindex // 14
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (14 * x0 + 14 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (7 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (8 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int64)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_1_2(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = xindex // 14
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (14 * x0 + 14 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (7 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (8 + 14 * x0 + 14 * x1), xmask,
        eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int64)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr1 + x2, tmp16, xmask)


@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16, primals_17,
        primals_18, primals_19, primals_20, primals_21, primals_22,
        primals_23) = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (1, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(primals_4, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (128,), (1,))
    assert_size_stride(primals_6, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (256,), (1,))
    assert_size_stride(primals_8, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_9, (256,), (1,))
    assert_size_stride(primals_10, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_11, (256,), (1,))
    assert_size_stride(primals_12, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_13, (256,), (1,))
    assert_size_stride(primals_14, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_15, (512,), (1,))
    assert_size_stride(primals_16, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_17, (512,), (1,))
    assert_size_stride(primals_18, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_19, (512,), (1,))
    assert_size_stride(primals_20, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_21, (512,), (1,))
    assert_size_stride(primals_22, (512, 512), (512, 1))
    assert_size_stride(primals_23, (512,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 64, 224, 224), (3097600, 48400, 224, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(5408)](buf1, primals_2, 
            5408, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (1, 128, 224, 224), (6590400, 51200, 224, 
            1))
        buf3 = buf2
        del buf2
        triton_poi_fused_convolution_relu_0[grid(117968)](buf3, primals_5,
            117968, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_5
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (1, 256, 224, 224), (13176000, 51200, 224,
            1))
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_relu_0[grid(147456)](buf5, primals_7,
            147456, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_7
        buf6 = extern_kernels.convolution(buf5, primals_8, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (1, 256, 224, 224), (13176000, 51200, 224,
            1))
        buf7 = buf6
        del buf6
        triton_poi_fused_convolution_relu_0[grid(147456)](buf7, primals_9,
            147456, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_9
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (1, 256, 224, 224), (13176000, 51200, 224,
            1))
        buf9 = buf8
        del buf8
        triton_poi_fused_convolution_relu_0[grid(147456)](buf9, primals_11,
            147456, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_11
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1, 256, 224, 224), (13176000, 51200, 224,
            1))
        buf11 = buf10
        del buf10
        triton_poi_fused_convolution_relu_0[grid(147456)](buf11, primals_13,
            147456, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_13
        buf12 = extern_kernels.convolution(buf11, primals_14, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 512, 224, 224), (26346240, 51200, 224,
            1))
        buf13 = buf12
        del buf12
        triton_poi_fused_convolution_relu_0[grid(294912)](buf13, primals_15,
            294912, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_15
        buf14 = extern_kernels.convolution(buf13, primals_16, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (1, 512, 224, 224), (26346240, 51200, 224,
            1))
        buf15 = buf14
        del buf14
        triton_poi_fused_convolution_relu_0[grid(294912)](buf15, primals_17,
            294912, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_17
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (1, 512, 224, 224), (26346240, 51200, 224,
            1))
        buf17 = buf16
        del buf16
        triton_poi_fused_convolution_relu_0[grid(294912)](buf17, primals_19,
            294912, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_19
        buf18 = extern_kernels.convolution(buf17, primals_20, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 512, 224, 224), (26346240, 51200, 224,
            1))
        buf19 = buf18
        del buf18
        triton_poi_fused_convolution_relu_0[grid(294912)](buf19, primals_21,
            294912, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_21
        buf20 = empty_strided_cuda((1, 512, 112, 112), (6590400, 51200, 4608,
            4), torch.float32)
        buf21 = empty_strided_cuda((1, 512, 112, 112), (6590400, 51200, 4608,
            4), torch.int64)
        triton_poi_fused_max_pool2d_with_indices_1[grid(1024)](buf19, buf20,
            buf21, 1024, XBLOCK=1024, num_warps=8, num_stages=1)
        buf22 = empty_strided_cuda((1, 512, 112, 112), (6590400, 51200, 4608,
            4), torch.float32)
        buf23 = empty_strided_cuda((1, 512, 112, 112), (6590400, 51200, 4608,
            4), torch.int64)
        triton_poi_fused_max_pool2d_with_indices_1_2[grid(256)](buf20,
            buf22, buf23, 256, XBLOCK=256, num_warps=4, num_stages=1)
        buf24 = extern_kernels.convolution(buf22, primals_22, stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (1, 512, 112, 112), (6590400, 51200, 4608,
            4))
        buf25 = buf24
        del buf24
        triton_poi_fused_add_2[grid(50331648)](buf25, primals_23, 50331648,
            XBLOCK=128, num_warps=8, num_stages=1)
        del primals_23
    return (buf25, primals_1, primals_3, primals_4, primals_6, primals_8,
        primals_10, primals_12, primals_14, primals_16, primals_18,
        primals_20, primals_22, buf1, buf3, buf5, buf7, buf9, buf11, buf13,
        buf15, buf17, buf19, buf20, buf21, buf22, buf23)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # VGG16 architecture: 5 blocks of convolutional layers followed by max pooling
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, input_0):
        primals_1 = self.features[0].weight
        primals_2 = self.features[0].bias
        primals_4 = self.features[2].weight
        primals_5 = self.features[2].bias
        primals_6 = self.features[4].weight
        primals_7 = self.features[4].bias
        primals_8 = self.features[6].weight
        primals_9 = self.features[6].bias
        primals_10 = self.features[8].weight
        primals_11 = self.features[8].bias
        primals_12 = self.features[10].weight
        primals_13 = self.features[10].bias
        primals_14 = self.features[12].weight
        primals_15 = self.features[12].bias
        primals_16 = self.features[14].weight
        primals_17 = self.features[14].bias
        primals_18 = self.features[16].weight
        primals_19 = self.features[16].bias
        primals_20 = self.features[18].weight
        primals_21 = self.features[18].bias
        primals_22 = self.classifier[0].weight
        primals_23 = self.classifier[0].bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18, primals_19,
            primals_20, primals_21, primals_22, primals_23])
        return output[0]
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2073600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 50176 % 192
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 50176 % 208
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2073600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 50176 % 48
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_max_pool2d_with_indices_3(in_ptr0, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 2073600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 224
    x1 = xindex // 224 % 224
    x2 = xindex // 50176
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 896 * x1 + 207360 * x2), xmask,
        eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2 * x0 + 896 * x1 + 207360 * x2), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (448 + 2 * x0 + 896 * x1 + 207360 * x2), xmask,
        eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (449 + 2 * x0 + 896 * x1 + 207360 * x2), xmask,
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
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 2073600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 50176 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4, primals_5, primals_6,
        primals_7, primals_8, primals_9, primals_10, primals_11, primals_12,
        primals_13, primals_14, primals_15, primals_16) = args
    args.clear()
    assert_size_stride(primals_1, (192, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_2, (192,), (1,))
    assert_size_stride(primals_3, (10, 480, 224, 224), (230400, 480, 224, 1))
    assert_size_stride(primals_4, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_5, (96,), (1,))
    assert_size_stride(primals_6, (208, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_7, (208,), (1,))
    assert_size_stride(primals_8, (16, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_9, (16,), (1,))
    assert_size_stride(primals_10, (48, 16, 5, 5), (400, 25, 5, 1))
    assert_size_stride(primals_11, (48,), (1,))
    assert_size_stride(primals_12, (64, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_13, (64,), (1,))
    assert_size_stride(primals_14, (64, 10, 3, 3), (90, 9, 3, 1))
    assert_size_stride(primals_15, (64,), (1,))
    assert_size_stride(primals_16, (10, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 192, 224, 224), (949184, 1, 4208, 
            19), torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(2073600)](buf1, primals_1, 
            2073600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((10, 208, 224, 224), (1024000, 1, 4576,
            19), torch.float32)
        buf3 = buf2
        del buf2
        triton_poi_fused_convolution_1[grid(18874368)](buf3, primals_6, 
            18874368, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_6
        buf4 = empty_strided_cuda((10, 48, 224, 224), (230400, 1, 936, 4),
            torch.float32)
        buf5 = buf4
        del buf4
        triton_poi_fused_convolution_2[grid(2073600)](buf5, primals_10, 
            2073600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_10
        buf6 = empty_strided_cuda((10, 480, 224, 224), (230400, 1, 936, 4),
            torch.float32)
        buf7 = buf6
        del buf6
        triton_poi_fused_max_pool2d_with_indices_3[grid(2073600)](primals_3,
            buf7, buf8, 2073600, XBLOCK=1024, num_warps=4, num_stages=1)
        buf9 = buf8
        del buf8
        triton_poi_fused_convolution_4[grid(2073600)](buf9, primals_12, 
            2073600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_12
        buf10 = empty_strided_cuda((10, 64, 224, 224), (322560, 1, 1435, 
            6), torch.float32)
        buf11 = buf10
        del buf10
        triton_poi_fused_convolution_4[grid(3225600)](buf11, primals_14, 
            3225600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_14
        buf12 = empty_strided_cuda((10, 64, 224, 224), (322560, 1, 1435, 
            6), torch.float32)
        buf13 = buf12
        del buf12
        triton_poi_fused_convolution_4[grid(3225600)](buf13, primals_16, 
            3225600, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_16
    return (buf11, buf13, primals_3, primals_2, primals_4, primals_5,
        primals_7, primals_8, primals_9, primals_11, buf1, buf3, buf5, buf7,
        buf9, buf11, buf13)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5,
        out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, input_0):
        primals_1 = self.branch1x1.weight
        primals_2 = self.branch1x1.bias
        primals_4 = self.branch3x3.conv1.weight
        primals_5 = self.branch3x3.conv1.bias
        primals_6 = self.branch3x3.conv2.weight
        primals_7 = self.branch3x3.conv2.bias
        primals_8 = self.branch5x5.conv1.weight
        primals_9 = self.branch5x5.conv1.bias
        primals_10 = self.branch5x5.conv2.weight
        primals_11 = self.branch5x5.conv2.bias
        primals_12 = self.branch_pool.conv.weight
        primals_13 = self.branch_pool.conv.bias
        primals_14 = self.branch_pool.max_pool2d_with_indices.weight
        primals_15 = self.branch_pool.max_pool2d_with_indices.bias
        primals_16 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16])
        return output[0]

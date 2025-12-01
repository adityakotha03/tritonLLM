import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 8601600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 224
    x1 = xindex // 224 % 224
    x2 = xindex // (224 * 224) % 480
    x3 = xindex // (224 * 224 * 480)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 224 * x1 + 45056 * x2 + 9830400 * x3),
        xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 + 224 * x1 + 12544 * x2 + 26214400 * x3),
        xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0 + 224 * x1 + 114688 * x2 + 24105600 * x3),
        xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x0 + 224 * x1 + 45056 * x2 + 9830400 * x3),
        xmask, eviction_policy='evict_last')
    tmp4 = tmp0 + tmp1
    tmp5 = tmp4 + tmp2
    tmp6 = tmp5 + tmp3
    tl.store(out_ptr0 + x4, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16 = args
    args.clear()
    assert_size_stride(primals_1, (192, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(primals_2, (192, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(primals_3, (1, 192, 1, 1), (192, 1, 192, 192))
    assert_size_stride(primals_4, (10, 480, 224, 224), (11059200, 226272, 105, 1))
    assert_size_stride(primals_5, (96, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(primals_6, (96, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(primals_7, (1, 96, 1, 1), (96, 1, 96, 96))
    assert_size_stride(primals_8, (208, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_9, (208, 208, 1, 1), (208, 1, 208, 208))
    assert_size_stride(primals_10, (1, 208, 1, 1), (208, 1, 208, 208))
    assert_size_stride(primals_11, (16, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(primals_12, (16, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(primals_13, (1, 16, 1, 1), (16, 1, 16, 16))
    assert_size_stride(primals_14, (48, 16, 5, 5), (400, 25, 5, 1))
    assert_size_stride(primals_15, (48, 48, 1, 1), (48, 1, 48, 48))
    assert_size_stride(primals_16, (1, 48, 1, 1), (48, 1, 48, 48))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_4, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (10, 192, 224, 224), (9830400, 51200, 224, 
            1))
        buf1 = extern_kernels.convolution(primals_4, primals_5, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (10, 96, 224, 224), (4505600, 46875, 224, 
            1))
        buf2 = extern_kernels.convolution(buf1, primals_8, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (10, 208, 224, 224), (10505408, 50485, 224,
            1))
        buf3 = extern_kernels.convolution(primals_4, primals_11, stride=(1, 
            1), padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (10, 16, 224, 224), (784000, 49000, 224, 1))
        buf4 = extern_kernels.convolution(buf3, primals_14, stride=(1, 1),
            padding=(2, 2), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (10, 48, 224, 224), (2190048, 45937.5, 224,
            1))
        buf5 = extern_kernels.max_pool2d_with_indices(buf4, (3, 3), (1, 1),
            (1, 1), padding=(1, 1), ceil_mode=False)
        assert_size_stride(buf5, (10, 48, 224, 224), (2190048, 45937.5, 224,
            1))
        buf6 = extern_kernels.convolution(buf5[0], primals_16, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (10, 64, 224, 224), (3097600, 48125, 224, 1))
        buf7 = empty_strided_cuda((10, 912, 224, 224), (9830400, 10752, 43,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_cat_0[grid(8601600)](buf0, buf2, buf6, buf4, buf7,
            8601600, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del buf2
        del buf4
        del buf6
    return (buf7, primals_1, primals_2, primals_3, primals_4, primals_5,
        primals_6, primals_7, primals_8, primals_9, primals_10, primals_11,
        primals_12, primals_13, primals_14, primals_15, primals_16, buf5[1])


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
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
        primals_5 = self.branch3x3[0].weight
        primals_6 = self.branch3x3[0].bias
        primals_7 = self.branch3x3[1].weight
        primals_8 = self.branch3x3[1].bias
        primals_11 = self.branch5x5[0].weight
        primals_12 = self.branch5x5[0].bias
        primals_13 = self.branch5x5[1].weight
        primals_14 = self.branch5x5[1].bias
        primals_16 = self.branch_pool[1].weight
        primals_15 = self.branch_pool[1].bias
        primals_3 = self.branch_pool[0].weight
        primals_4 = input_0
        primals_9 = self.branch3x3[1].bias
        primals_10 = self.branch3x3[0].bias
        primals_13 = self.branch5x5[1].weight
        primals_12 = self.branch5x5[0].bias
        primals_14 = self.branch5x5[1].bias
        primals_15 = self.branch_pool[1].bias
        primals_1 = self.branch1x1.weight
        primals_2 = self.branch1x1.bias
        primals_4 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16])
        return output[0]
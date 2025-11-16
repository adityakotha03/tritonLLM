import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16777216 % 6
    x0 = xindex % 16777216
    x2 = xindex // 6
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x0 + 16777216 * x2), tmp4, xmask)


@triton.jit
def triton_poi_fused_convolution_relu_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1073741824 % 64
    x0 = xindex % 1073741824
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x0 + 1073741824 * x1), tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (6, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (6,), (1,))
    assert_size_stride(primals_3, (64, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 6, 3, 3), (54, 9, 3, 1))
    assert_size_stride(primals_2, (6,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 6, 256, 256), (393216, 65536, 16, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(134217728)](primals_1,
            primals_2, buf0, 134217728, XBLOCK=512, num_warps=4, num_stages=1)
        del primals_2
        buf1 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 16, 1
            ), torch.float32)
        triton_poi_fused_convolution_relu_1[grid(536870912)](buf0,
            primals_3, primals_4, buf1, 536870912, XBLOCK=512, num_warps=8,
            num_stages=1)
        del primals_4
        buf2 = empty_strided_cuda((128, 64, 256, 256), (4194304, 65536, 16, 1
            ), torch.float32)
        triton_poi_fused_convolution_relu_1[grid(536870912)](buf0,
            primals_5, primals_2, buf2, 536870912, XBLOCK=512, num_warps=8,
            num_stages=1)
        del primals_5
        del primals_2
    return buf2, primals_1, primals_3, primals_5, buf0, buf1


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(ModelNew, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, input_0):
        primals_1 = self.squeeze.weight
        primals_2 = self.squeeze.bias
        primals_3 = self.expand1x1.weight
        primals_4 = self.expand1x1.bias
        primals_5 = self.expand3x3.weight
        primals_6 = self.expand3x3.bias
        primals_7 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]

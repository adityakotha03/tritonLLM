import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 256 % 65536
    x0 = xindex % 256
    x2 = xindex // 65536
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 256 * (x1 % 65536) + 65536 * (x2 % 1048576
        )), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + x3, xmask)
    tmp3 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + x3, tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (6, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (128, 3, 256, 256), (196608, 65536, 256,
        1))
    assert_size_stride(primals_3, (64, 6, 1, 1), (6, 1, 1, 1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 6, 3, 3), (54, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_2, primals_1, 
            stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 6, 256, 256), (393216, 65536, 256, 1))
        buf1 = empty_strided_cuda((128, 6, 256, 256), (393216, 65536, 256, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(1048576)](buf0, primals_1,
            buf1, 1048576, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_1
        buf2 = torch.ops.aten.convolution.default(buf1, primals_3, stride=(1,
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf3 = torch.ops.aten.convolution.default(buf1, primals_5, stride=(1,
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (128, 64, 256, 256), (4194304, 65536, 256, 
            1))
        buf4 = torch.ops.aten.cat.default([buf2, buf3], 1)
    return buf4, primals_2, primals_3, primals_5, buf1, buf2, buf3


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
        primals_3 = self.expand1x1.weight
        primals_5 = self.expand3x3.weight
        primals_2 = input_0
        primals_4 = self.expand1x1.bias
        primals_6 = self.expand3x3.bias
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]
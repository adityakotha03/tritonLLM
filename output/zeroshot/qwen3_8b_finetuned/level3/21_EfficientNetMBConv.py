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
def triton_poi_fused_add_convolution_relu6_0(in_ptr0, in_ptr1, in_ptr2,
    in_ptr3, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 566112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4 + tmp6
    tmp7 = 6.0
    tmp8 = triton_helpers.maximum(tmp7, tmp5)
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (192, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(primals_2, (192,), (1,))
    assert_size_stride(primals_3, (192,), (1,))
    assert_size_stride(primals_4, (192,), (1,))
    assert_size_stride(primals_5, (10, 112, 224, 224), (563872, 1, 224, 112))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 192, 224, 224), (86016, 1, 376, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_convolution_relu6_0[grid(566112)](primals_5,
            primals_1, primals_2, primals_3, buf0, 566112, XBLOCK=128,
            num_warps=4, num_stages=1)
        del primals_1
        del primals_2
        del primals_3
    return buf0, primals_5


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, input_0):
        primals_1 = self.project_conv[0].weight
        primals_2 = self.project_conv[1].weight
        primals_3 = self.project_conv[1].bias
        primals_5 = input_0
        output = call([primals_1, primals_2, primals_3, primals_5])
        return output[0]
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
def triton_poi_fused_add_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 53084160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp3 = tl.load(in_ptr2 + x0, xmask)
    tmp5 = tl.load(in_ptr3 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tl.store(out_ptr0 + x0, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (60, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_2, (60,), (1,))
    assert_size_stride(primals_3, (10, 224, 224, 60), (301920, 1344, 6, 1))
    assert_size_stride(primals_4, (60, 60, 3, 3), (540, 9, 3, 1))
    assert_size_stride(primals_5, (60,), (1,))
    assert_size_stride(primals_6, (480, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_7, (480,), (1,))
    assert_size_stride(primals_8, (10, 480, 224, 224), (11059200, 480, 224,
        1))
    assert_size_stride(primals_9, (480,), (1,))
    assert_size_stride(primals_10, (480,), (1,))
    assert_size_stride(primals_11, (480,), (1,))
    assert_size_stride(primals_12, (480,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=3, bias=None)
        assert_size_stride(buf0, (10, 60, 224, 224), (301920, 5040, 21, 1))
        buf1 = extern_kernels.batch_norm(primals_2, buf0, eps=1e-05,
            momentum=0.1, training=True, output_mean_var=True)
        buf2, buf3, buf4 = buf1
        del buf1
        buf5 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=60, bias=None)
        assert_size_stride(buf5, (10, 60, 224, 224), (301920, 5040, 21, 1))
        buf6 = extern_kernels.batch_norm(buf5, primals_5, eps=1e-05,
            momentum=0.1, training=True, output_mean_var=True)
        buf7, buf8, buf9 = buf6
        del buf6
        buf10 = buf7
        del buf7
        buf11 = empty_strided_cuda((10, 60, 224, 224), (301920, 5040, 21, 1),
            torch.float32)
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf10, (10, 60),
            (60, 1), 0), reinterpret_tensor(primals_6, (60, 480), (1, 60), 
            0), alpha=1, beta=1, out=buf11)
        del primals_10
        buf12 = extern_kernels.batch_norm(buf11, primals_12, eps=1e-05,
            momentum=0.1, training=True, output_mean_var=True)
        buf13, buf14, buf15 = buf12
        del buf12
        buf16 = empty_strided_cuda((10, 480, 224, 224), (11059200, 480, 21,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_relu_0[grid(53084160)](buf13, primals_8,
            primals_11, primals_12, buf16, 53084160, XBLOCK=512, num_warps=
            4, num_stages=1)
        del buf13
        del primals_12
        del buf14
        del primals_11
        del buf15
    return buf16, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, buf0, buf2, buf3, buf4, buf5, buf8, buf9, buf10, buf11, reinterpret_tensor(buf10, (10, 60), (60, 1), 0)


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
    
batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [input_channels, out_channels, groups]

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
        
        # Shuffle operation
        self.shuffle = ChannelShuffle(groups)
        
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
        primals_2 = self.bn1.weight
        primals_3 = self.bn1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.bn2.weight
        primals_6 = self.bn2.bias
        primals_7 = self.conv3.weight
        primals_8 = self.bn3.weight
        primals_9 = self.bn3.bias
        primals_10 = self.shortcut[0].weight
        primals_11 = self.shortcut[1].weight
        primals_12 = self.shortcut[1].bias
        primals_13 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5,
            primals_6, primals_7, primals_8, primals_9, primals_10,
            primals_11, primals_12, primals_13])
        return output[0]
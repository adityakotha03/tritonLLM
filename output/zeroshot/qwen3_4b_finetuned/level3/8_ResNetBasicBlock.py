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
def triton_per_fused__native_batch_norm_legit_convolution_relu_0(in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    x2 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 * tmp5
    tmp8 = 0.0
    tmp9 = tmp7 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = 0.9999999999999999
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 / tmp5
    tmp14 = tmp8 - tmp13
    tmp15 = tmp7 + tmp14
    tmp16 = 0.0625
    tmp17 = tmp13 * tmp16
    tmp18 = tmp15 - tmp17
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp21 = tmp20 * tmp5
    tl.store(out_ptr0 + (x0 + 64 * r1), tmp2, None
        )
    tl.store(out_ptr1 + (r1 + 64 * x0), tmp21, None)


@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_1(in_out_ptr0,
    in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 224
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x3, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (10, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (10,), (1,))
    assert_size_stride(primals_3, (10, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_4, (10, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (10, 64, 1, 1), (64, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((10, 64, 3, 3), (576, 1, 192, 64), torch
            .float32)
        extern_kernels.convolution(primals_1, primals_2, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None, out=buf0)
        del primals_2
        buf1 = extern_kernels.batch_norm(primals_3, buf0, None, None,
            None, 0, 1e-05, True)
        buf16 = empty_strided_cuda((10, 64, 224, 224), (32256, 1, 144, 1),
            torch.float32)
        buf17 = empty_strided_cuda((10, 64, 224, 224), (32256, 1, 144, 1),
            torch.float32)
        get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_relu_0[grid(4096
            )](buf1, buf0, primals_3, buf16, buf17, 4096, 64, XBLOCK=128,
            num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((10, 64, 3, 3), (576, 1, 192, 64), torch
            .float32)
        extern_kernels.convolution(buf16, primals_4, stride=(1, 1),
            padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None, out=buf2)
        buf3 = extern_kernels.batch_norm(primals_5, buf2, None, None, None, 
            0, 1e-05, True)
        del primals_5
        buf4 = buf3
        del buf3
        triton_per_fused__native_batch_norm_legit_convolution_relu_1[grid(4096
            )](buf4, primals_4, primals_3, 4096, XBLOCK=128, num_warps=4,
            num_stages=1)
    return buf4, primals_1, primals_4, primals_3, buf16, buf0, buf2, buf17


class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        :param downsample: Downsample layer for the shortcut connection
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion,
                kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, input_0):
        primals_3 = self.conv1.weight
        primals_2 = self.conv1.bias
        primals_4 = self.conv2.weight
        primals_5 = self.downsample[0].weight
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5])
        return output[0]

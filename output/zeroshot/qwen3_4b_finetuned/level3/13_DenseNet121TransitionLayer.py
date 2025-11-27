import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp1 <= tmp0
    tmp3 = 0.0
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_poi_fused_avg_pool2d_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 128 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + x2, tmp8, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 32, 256, 256), (2097152, 6656, 256,
        1))
    assert_size_stride(primals_4, (64,), (1,))
    assert_size_stride(primals_5, (64, 64, 1, 1), (64, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16384)](primals_3, buf0, 16384,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_3
        buf1 = buf0
        del buf0
        triton_poi_fused_relu_1[grid(16384)](buf1, 16384, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf2 = empty_strided_cuda((128, 64, 256, 256), (16384, 1, 64, 1),
            torch.float32)
        triton_poi_fused_convolution_0[grid(16384)](buf1, buf2, 16384,
            XBLOCK=128, num_warps=4, num_stages=1)
        buf3 = buf2
        del buf2
        triton_poi_fused_relu_1[grid(16384)](buf3, 16384, XBLOCK=256,
            num_warps=4, num_stages=1)
        buf4 = empty_strided_cuda((128, 64, 128, 128), (1048576, 1, 8192, 1
            ), torch.float32)
        triton_poi_fused_avg_pool2d_2[grid(1024)](buf3, buf4, 1024, XBLOCK=
            256, num_warps=4, num_stages=1)
    return buf4, primals_1, primals_2, primals_4, primals_5, buf1, buf3


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1,
                bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, input_0):
        primals_1 = self.transition[0].weight
        primals_2 = self.transition[0].running_mean
        primals_3 = self.transition[0].running_var
        primals_4 = self.transition[0].num_batches_tracked
        primals_5 = self.transition[2].weight
        primals_6 = self.transition[1].weight
        primals_7 = self.transition[1].bias
        primals_8 = self.transition[3].weight
        primals_9 = self.transition[3].bias
        primals_10 = self.transition[3].stride
        primals_11 = self.transition[3].padding
        primals_12 = self.transition[3].dilation
        primals_13 = self.transition[3].transposed
        primals_14 = self.transition[3].output_padding
        primals_15 = self.transition[3].ceil_mode
        primals_16 = self.transition[3].groups
        primals_17 = self.transition[3].bias
        primals_18 = self.transition[3].padding_mode
        input_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4,
            primals_5, input_1, primals_6, primals_7, primals_8, primals_9,
            primals_10, primals_11, primals_12, primals_13, primals_14,
            primals_15, primals_16, primals_17, primals_18])
        return output[0]

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2 % 64
    x2 = xindex // 128
    x0 = xindex % 2
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2 * x1 + 64 * x2 + 128 * x0), xmask)
    tmp1 = tl.load(in_ptr0 + (2 + 2 * x1 + 64 * x2 + 128 * x0), xmask)
    tmp3 = tl.load(in_ptr0 + (1 + 2 * x1 + 64 * (1 + x2) + 128 * x0), xmask)
    tmp5 = tl.load(in_ptr0 + (2 + 2 * x1 + 64 * (1 + x2) + 128 * x0), xmask)
    tmp7 = tl.load(in_ptr0 + (63 - x1 + 1 + 2 * (63 - x1) + 64 * x2 + 128 * x0
        ), xmask)
    tmp9 = tl.load(in_ptr0 + (64 - x1 + 2 + 2 * (63 - x1) + 64 * x2 + 128 * x0
        ), xmask)
    tmp11 = tl.load(in_ptr0 + (63 - x1 + 1 + 2 * (63 - x1) + 64 * (1 + x2) +
        128 * x0), xmask)
    tmp13 = tl.load(in_ptr0 + (64 - x1 + 2 + 2 * (63 - x1) + 64 * (1 + x2) +
        128 * x0), xmask)
    tmp15 = tl.load(in_ptr0 + (63 - x1 + 1 + 2 * (63 - x1) + 64 * x2 + 128 *
        (1 + x0)), xmask)
    tmp17 = tl.load(in_ptr0 + (64 - x1 + 2 + 2 * (63 - x1) + 64 * x2 + 128 *
        (1 + x0)), xmask)
    tmp20 = tl.load(in_ptr0 + (63 - x1 + 1 + 2 * (63 - x1) + 64 * (1 + x2) +
        128 * (1 + x0)), xmask)
    tmp22 = tl.load(in_ptr0 + (64 - x1 + 2 + 2 * (63 - x1) + 64 * (1 + x2) +
        128 * (1 + x0)), xmask)
    tmp2 = libdevice.tanh(tmp0)
    tmp4 = libdevice.tanh(tmp1)
    tmp6 = libdevice.tanh(tmp3)
    tmp8 = libdevice.tanh(tmp5)
    tmp10 = libdevice.tanh(tmp7)
    tmp12 = libdevice.tanh(tmp9)
    tmp14 = libdevice.tanh(tmp11)
    tmp16 = libdevice.tanh(tmp13)
    tmp18 = libdevice.tanh(tmp15)
    tmp21 = libdevice.tanh(tmp17)
    tmp23 = libdevice.tanh(tmp20)
    tmp24 = libdevice.tanh(tmp22)
    tmp25 = tmp1 + tmp2
    tmp26 = tmp4 + tmp5
    tmp27 = tmp25 + tmp6
    tmp28 = tmp27 + tmp8
    tmp29 = tmp28 / 4.0
    tmp30 = tmp10 + tmp12
    tmp31 = tmp14 + tmp16
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 / 4.0
    tmp34 = tmp18 + tmp21
    tmp35 = tmp23 + tmp24
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36 / 4.0
    tmp38 = tmp29 + tmp33
    tmp39 = tmp38 / 2.0
    tmp40 = tmp39 + tmp37
    tmp41 = tmp40 / 2.0
    tl.store(out_ptr0 + x3, tmp41, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 128, 128), (524288, 8192, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 3, 3), (1152, 9, 3, 1), torch.
            float32)
        get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0[grid(256)](arg0_1, buf0, 256, XBLOCK
            =128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value,
        subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]

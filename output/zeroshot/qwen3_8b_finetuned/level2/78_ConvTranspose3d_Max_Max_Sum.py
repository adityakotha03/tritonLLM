import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool3d_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32768 * x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16384 + x0 + 32768 * x1), xmask)
    tmp3 = tl.load(in_ptr0 + (32768 + x0 + 32768 * x1), xmask)
    tmp5 = tl.load(in_ptr0 + (49152 + x0 + 32768 * x1), xmask)
    tmp7 = tl.load(in_ptr0 + (65536 + x0 + 32768 * x1), xmask)
    tmp9 = tl.load(in_ptr0 + (81920 + x0 + 32768 * x1), xmask)
    tmp11 = tl.load(in_ptr0 + (98304 + x0 + 32768 * x1), xmask)
    tmp13 = tl.load(in_ptr0 + (114688 + x0 + 32768 * x1), xmask)
    tmp15 = tl.load(in_ptr0 + (131072 + x0 + 32768 * x1), xmask)
    tmp17 = tl.load(in_ptr0 + (147456 + x0 + 32768 * x1), xmask)
    tmp19 = tl.load(in_ptr0 + (163840 + x0 + 32768 * x1), xmask)
    tmp21 = tl.load(in_ptr0 + (179232 + x0 + 32768 * x1), xmask)
    tmp23 = tl.load(in_ptr0 + (195616 + x0 + 32768 * x1), xmask)
    tmp25 = tl.load(in_ptr0 + (211936 + x0 + 32768 * x1), xmask)
    tmp27 = tl.load(in_ptr0 + (228320 + x0 + 32768 * x1), xmask)
    tmp29 = tl.load(in_ptr0 + (244704 + x0 + 32768 * x1), xmask)
    tmp31 = tl.load(in_ptr0 + (261088 + x0 + 32768 * x1), xmask)
    tmp33 = tl.load(in_ptr0 + (277472 + x0 + 32768 * x1), xmask)
    tmp35 = tl.load(in_ptr0 + (293856 + x0 + 32768 * x1), xmask)
    tmp37 = tl.load(in_ptr0 + (310240 + x0 + 32768 * x1), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp20 = triton_helpers.maximum(tmp19, tmp18)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp24 = triton_helpers.maximum(tmp23, tmp22)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp32 = triton_helpers.maximum(tmp31, tmp30)
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp38 = triton_helpers.maximum(tmp37, tmp36)
    tmp40 = tmp38 + tmp37
    tl.store(out_ptr0 + x2, tmp40, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 32, 32), (32768, 1024, 32, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(arg0_1, weight=None, bias=None,
            stride=(2, 2, 2), padding=(2, 2, 2), dilation=(1, 1, 1),
            transposed=True, output_padding=(0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (16, 64, 32, 32, 32), (2097152, 32768, 1024,
            32, 1))
        buf1 = empty_strided_cuda((16, 1, 32, 32, 32), (16384, 1, 512, 16, 1
            ), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool3d_sum_0[grid(1024)](buf0, buf1, 1024,
            XBLOCK=1024, num_warps=4, num_stages=1)
    return buf1, arg0_1, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
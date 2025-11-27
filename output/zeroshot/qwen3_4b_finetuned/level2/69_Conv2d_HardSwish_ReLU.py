import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_hardtanh_hardswish_relu_0(in_out_ptr0,
    in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1600 % 64
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 3.0
    tmp8 = tmp6 > tmp7
    tmp9 = tl.full([1], 1, tl.int8)
    tmp10 = tl.full([1], 0, tl.int8)
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tmp6 - tmp7
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp16 = tl.where(tmp8, tmp15, tmp15)
    tmp17 = tl.where(tmp8, tmp11, tmp10)
    tmp18 = triton_helpers.maximum(tmp16, tmp3)
    tmp19 = libdevice.signbit(tmp18) == 0
    tl.store(in_out_ptr0 + x3, tmp2, xmask)
    tl.store(out_ptr0 + x3, tmp17, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 128, 128), (131072, 16384, 128, 
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [1,
            1], dilation=[1, 1], transposed=False, output_padding=[0, 0],
            groups=1, bias=None)
        buf1 = buf0
        del buf0
        buf2 = empty_strided_cuda((128, 64, 126, 126), (10077696, 157689,
            126, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_hardtanh_hardswish_relu_0[grid(73728)](
            buf1, primals_2, buf2, 73728, XBLOCK=512, num_warps=4, num_stages=1
            )
        del primals_2
    return buf2, primals_1, primals_3, buf1


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

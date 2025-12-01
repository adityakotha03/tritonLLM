import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_mish_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = xindex // 65536
    tmp3 = tmp2 % 128
    tmp4 = tmp0 + tmp1
    tmp5 = tl_math.exp(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tl_math.log(tmp7)
    tmp9 = tl_math.tanh(tmp8)
    tmp10 = tmp4 * tmp9
    tl.store(out_ptr0 + x2, tmp10, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4194304, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_3, primals_1,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=
            False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 128, 256, 256), (8388608, 65536, 256, 
            1))
        buf1 = empty_strided_cuda((64, 128, 256, 256), (8388608, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_mish_0[grid(536870912)](buf0, primals_2, buf1,
            536870912, XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((64, 128, 256, 256), (8388608, 65536, 256,
            1), torch.float32)
        triton_poi_fused_add_mish_0[grid(536870912)](buf1, primals_2, buf2,
            536870912, XBLOCK=256, num_warps=4, num_stages=1)
        del buf1
        del primals_2
    return buf2, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
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
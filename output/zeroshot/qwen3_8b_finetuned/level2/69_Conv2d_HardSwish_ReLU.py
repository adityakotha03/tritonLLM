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


@triton.jit
def triton_poi_fused_hardtanh_mul_relu_0(in_ptr0, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = triton_helpers.minimum(tmp2, tmp3)
    tmp5 = 3.0
    tmp6 = tmp4 + tmp5
    tmp7 = 6.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp4 * tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tl.store(out_ptr0 + x0, tmp12, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 128, 128), (1048576, 16384, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(arg0_1, weight, bias=None,
            stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False
            , output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128, 1
            ))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128, 
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_hardtanh_mul_relu_0[grid(134217728)](buf0, buf1, 
            134217728, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return buf1, arg0_1, weight


class ModelNew(nn.Module):
    """
    Model that performs a convolution, applies HardSwish, and then ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        arg0_1 = input_0
        weight = self.conv.weight
        output = call([arg0_1, weight])
        return output[0]
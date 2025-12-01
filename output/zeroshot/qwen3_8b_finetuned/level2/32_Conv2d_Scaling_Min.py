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
def triton_poi_fused_add_convolution_min_mul_0(in_out_ptr0, in_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = xindex // 128
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = tl.load(in_out_ptr0 + (x0 + 128 * x2), xmask, eviction_policy=
        'evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tl.load(in_out_ptr0 + (64 + x0 + 128 * x2), xmask, eviction_policy
        ='evict_last')
    tmp7 = tmp3 * tmp6
    tmp8 = tl.load(in_out_ptr0 + (128 + x0 + 128 * x2), xmask,
        eviction_policy='evict_last')
    tmp9 = tmp3 * tmp8
    tmp10 = triton_helpers.minimum(tmp5, tmp7)
    tmp11 = triton_helpers.minimum(tmp10, tmp9)
    tmp12 = tl.load(in_out_ptr0 + (192 + x0 + 128 * x2), xmask,
        eviction_policy='evict_last')
    tmp13 = tmp3 * tmp12
    tmp14 = triton_helpers.minimum(tmp11, tmp13)
    tl.store(in_out_ptr0 + x3, tmp14, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (64, 64, 256, 256), (4194304, 65536, 256,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 128, 256, 256), (8388608, 65536, 256,
            1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_convolution_min_mul_0[grid(4194304)](buf1,
            primals_2, 4194304, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
    return buf1, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
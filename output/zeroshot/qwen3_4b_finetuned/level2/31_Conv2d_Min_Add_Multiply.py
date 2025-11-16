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
def triton_poi_fused_add_min_mul_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex // 4096 % 128
    x4 = xindex // 524288
    tmp0 = tl.load(in_ptr0 + (3 + x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (3 + x3), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (x0 + 64 * x2 + 4096 * x4), xmask,
        eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr3 + x2, xmask, eviction_policy='evict_last')
    tmp2 = 0.5
    tmp3 = tmp1 <= tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tmp0 * (~tmp3)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp1 * tmp2
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = 2.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 + tmp17
    tmp14 = tmp13 * tmp11
    tmp15 = tmp14 + tmp24
    tl.store(out_ptr0 + x3, tmp15, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_2, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (128,), (1,))
    assert_size_stride(primals_4, (128, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 
            1), padding=(0, 0), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 128, 128), (2097152, 16384, 128,
            1))
        buf1 = empty_strided_cuda((128, 128, 128, 128), (2097152, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_min_mul_0[grid(128800)](buf0, buf1, primals_3,
            primals_4, buf1, 128800, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
        del primals_3
        del primals_4
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, 
        bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_2 = self.conv.weight
        primals_3 = self.conv.bias
        primals_1 = input_0
        primals_4 = self.bias
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

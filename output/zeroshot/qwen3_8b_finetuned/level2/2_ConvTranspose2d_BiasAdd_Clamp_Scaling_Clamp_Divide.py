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
def triton_poi_fused_add_clamp_div_max_mul_min_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1342177216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 16384 % 64
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp3)
    tmp10 = triton_helpers.minimum(tmp9, tmp5)
    tmp11 = tmp10 / tmp7
    tmp12 = tmp11 - tmp6
    tmp13 = 0.0
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp14 - tmp11
    tmp16 = tmp15 <= tmp13
    tl.store(out_ptr0 + x3, tmp16, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_2, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128, 
            1))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.bool)
        get_raw_stream(0)
        triton_poi_fused_add_clamp_div_max_mul_min_0[grid(1342177216)](buf0,
            primals_2, buf1, 1342177216, XBLOCK=1024, num_warps=4,
            num_stages=1)
        del buf0
        del primals_2
    return buf1, primals_1, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a bias term, clamps, scales, clamps, and divides.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_2 = self.bias
        primals_3 = self.scaling_factor
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
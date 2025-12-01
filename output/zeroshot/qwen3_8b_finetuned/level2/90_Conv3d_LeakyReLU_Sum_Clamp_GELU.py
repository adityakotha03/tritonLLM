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
def triton_poi_fused_leaky_relu_clamp_gelu_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 33296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = 1.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = -1.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = 0.7071067811865476
    tmp13 = tmp9 * tmp12
    tmp14 = tl_math.tanh(tmp13)
    tmp15 = 1.0
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = 0.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = 0.0
    tmp22 = tmp20 + tmp21
    tmp23 = tmp17 + tmp22
    tl.store(out_ptr0 + x0, tmp23, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 8, 16, 64, 64), (8192, 1024, 64, 1,
        1))
    assert_size_stride(primals_3, (64, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 17, 65, 65), (707872, 110592, 4160,
            64, 1))
        buf1 = empty_strided_cuda((128, 64, 17, 65, 65), (707872, 110592, 
            4160, 64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_clamp_gelu_0[grid(33296)](buf0, buf1, 
            33296, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.sum_tensor
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
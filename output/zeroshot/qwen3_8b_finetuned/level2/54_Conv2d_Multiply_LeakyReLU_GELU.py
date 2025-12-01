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
def triton_poi_fused_leaky_relu_mul_native_layer_norm_native_layer_norm_3(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 0.01
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tmp9 = 0.5
    tmp10 = 1.0
    tmp11 = tmp8 * tmp9
    tmp12 = tmp8 * tmp10
    tmp13 = 1.4142135623731027
    tmp14 = tmp12 / tmp13
    tmp15 = tl_math.erf(tmp14)
    tmp16 = tmp15 + tmp10
    tmp17 = tmp11 * tmp16
    tmp18 = tl.where(tmp5, tmp17, tmp11)
    tl.store(out_ptr0 + x0, tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_3, (64, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (64, 64, 256, 256), (4194304, 65536, 256, 1))
        buf1 = empty_strided_cuda((64, 64, 256, 256), (4194304, 65536, 256,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_mul_native_layer_norm_native_layer_norm_3[
            grid(268435456)](buf0, primals_3, buf1, 268435456, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf1, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        primals_3 = self.multiplier
        output = call([primals_1, primals_2, primals_3])
        return output[0]
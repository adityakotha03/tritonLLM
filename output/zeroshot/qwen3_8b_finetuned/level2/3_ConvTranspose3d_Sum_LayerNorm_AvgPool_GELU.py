import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_gelu_layer_norm_0(
    in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 16384
    tmp0 = tl.load(in_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.6000000238410949
    tmp4 = tmp2 * tmp3
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp0
    tmp10 = 0.5
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp10
    tmp13 = libdevice.erf(tmp12)
    tmp14 = tmp13 * tmp10
    tmp15 = tmp10 + tmp14
    tmp16 = tmp15 * tmp11
    tmp17 = 2.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18 * tmp10
    tmp20 = tmp19 + tmp18
    tmp21 = tmp20 * tmp10
    tmp22 = tmp11 * tmp21
    tl.store(out_ptr0 + x2, tmp22, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (32, 64, 16, 32, 32), (16384, 256, 16, 0.5,
            0.015625))
        buf1 = empty_strided_cuda((32, 64, 16, 32, 32), (16384, 256, 16, 0.5,
            0.015625), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_avg_pool3d_convolution_gelu_layer_norm_0[grid(2048
            )](buf0, primals_3, buf1, 2048, XBLOCK=128, num_warps=4, num_stages
            =1)
        del buf0
        del primals_3
    return buf1, primals_1, primals_2


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_3 = self.sum_weight
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
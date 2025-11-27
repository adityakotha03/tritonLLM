import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_clamp_gelu_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = 0.0
    tmp5 = tmp3 > tmp4
    tmp6 = 0.2
    tmp7 = tmp3 * tmp6
    tmp8 = tl.where(tmp5, tmp3, tmp7)
    tmp9 = -1.0
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tmp11 = 1.0
    tmp12 = tl.where(tmp5, tmp11, tmp10)
    tmp13 = tmp8 * tmp12
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 * tmp15
    tmp17 = tmp13 - tmp16
    tmp18 = tmp17 * tmp12
    tmp19 = tl.where(tmp5, tmp13, tmp18)
    tl.store(out_ptr0 + x0, tmp19, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 8, 3, 3, 3), (576, 72, 9, 3, 1))
    assert_size_stride(arg1_1, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 15, 63, 63), (576000, 9000, 63,
            1, 1), torch.float32)
        triton_poi_fused_add_clamp_gelu_leaky_relu_0[ext_import('triton', 'grid')]
        (1048576,), (128, 64, 15, 63, 63), 0)
        buf1 = F.convolution(arg1_1, arg0_1, stride=(1, 1, 1), padding=(1,
            1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0,
            0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (128, 64, 15, 63, 63), (576000, 9000, 63, 1,
            1))
        del arg0_1
        del arg1_1
    return buf1, buf0


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, input_0):
        arg1_1 = self.sum_tensor
        arg0_1 = self.conv.weight
        output = call([arg0_1, arg1_1])
        return output[0]

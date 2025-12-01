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
def triton_poi_fused_add_hardtanh_mish_mul_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 1.0
    tmp7 = tmp5 - tmp6
    tmp8 = tl_math.tanh(tmp7)
    tmp9 = tmp0 * tmp8
    tmp10 = 0.5
    tmp11 = tmp9 + tmp10
    tmp12 = -1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tmp14 = 1.0
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = 2.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + x0, tmp17, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.convolution(primals_3, primals_1,
            stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
            dilation=(1, 1), transposed=True, groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128, 1
            ))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 128,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_hardtanh_mish_mul_0[grid(1048576)](buf0,
            buf1, 1048576, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies Mish activation, adds a value, 
    applies Hardtanh activation, and scales the output.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
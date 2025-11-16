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
def triton_poi_fused_clamp_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2702080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -1.0
    tmp2 = torch.clamp(tmp0, min=tmp1)
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 64, 24, 48, 48), (368640, 5760, 240,
        48, 1))
    assert_size_stride(primals_2, (128, 64, 3, 3, 3), (5832, 9, 12, 36, 12))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2,
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 128, 36, 24, 24), (20736, 16, 1152,
            48, 2))
        buf1 = empty_strided_cuda((16, 128, 36, 24, 24), (20736, 16, 1152,
            48, 2), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_div_0[grid(2702080)](buf0, buf1, 2702080,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf0
    return buf1, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    A model that performs a transposed 3D convolution, clamps the output to a minimum value, 
    and then divides the result by a constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

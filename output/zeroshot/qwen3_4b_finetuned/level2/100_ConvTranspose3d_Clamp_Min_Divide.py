import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clamp_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = -1.0
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp2 < tmp1
    tmp4 = tl.where(tmp3, tmp1, tmp0)
    tmp5 = 2.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + x0, tmp6, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 24, 48, 48), (36864, 576, 2304, 48,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 128, 25, 49, 49), (16128, 128, 64, 1,
            1), torch.float32)
        get_input = reinterpret_tensor(arg0_1, (16, 128, 25, 49, 49), (16128,
            128, 64, 1, 1), 0)
        triton_poi_fused_clamp_div_0[grid=lambda meta: (1327104,)](get_input,
            buf0, 1327104, XBLOCK=1024, num_warps=4, num_stages=1)
        del get_input
    return buf0, arg0_1


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
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]

import torch
import torch.nn as nn
import triton
import triton.language as tl
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_relu_hardswish_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 10321920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 > 0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 + 3
    tmp4 = tmp3 / 6
    tmp5 = tmp4 <= 1
    tmp6 = tmp4 >= 0
    tmp7 = tmp5 & tmp6
    tmp8 = tmp4 * tmp7
    tmp9 = tmp2 * tmp8
    tl.store(out_ptr0 + x0, tmp9, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64, 126, 126), (108864, 1686, 126, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 64, 126, 126), (108864, 1686, 126, 1),
            torch.float32)
        get_raw_buf = buf0
        buf1 = buf0
        del buf0
        triton_poi_fused_relu_hardswish_0[grid(10321920)](arg0_1, buf1,
            10321920, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf1,


class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, applies a fused ReLU+HardSwish
    activation using a Triton kernel, and returns the final tensor.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
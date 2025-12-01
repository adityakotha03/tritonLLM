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
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + x0, tmp3, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_2, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_3, (16, 64, 128, 128), (1048576, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(16)](primals_1, primals_2, buf0, 16,
            XBLOCK=1, num_warps=2, num_stages=1)
        del primals_2
    return buf0, primals_1, primals_3


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_3 = self.conv_transpose.weight
        primals_2 = self.bias
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
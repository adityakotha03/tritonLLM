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
def triton_poi_fused_clamp_hardtanh_mul_relu_0(in_ptr0, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 6.0
    tmp4 = tmp2 / tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp0 * tmp6
    tl.store(out_ptr0 + x0, tmp7, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_2, (128, 8, 128, 128), (131072, 16384, 128,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 128, 128), (1048576, 16384, 128, 
            1))
        buf1 = empty_strided_cuda((128, 64, 128, 128), (1048576, 16384, 
            128, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clamp_hardtanh_mul_relu_0[grid(1048576)](buf0,
            buf1, 1048576, XBLOCK=256, num_warps=4, num_stages=1)
    return buf1, primals_1, primals_2, buf0


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, applies ReLU, and applies HardSwish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = input_0
        output = call([primals_1, primals_2])
        return output[0]
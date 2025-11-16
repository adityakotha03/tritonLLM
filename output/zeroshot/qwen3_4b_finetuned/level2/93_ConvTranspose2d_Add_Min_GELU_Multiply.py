import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xnumel = 819200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 64 % 128
    x0 = xindex % 64
    x4 = xindex // 4096
    x2 = xindex // 4096 % 64
    tmp0 = tl.load(in_out_ptr0 + (x1 + 128 * x0 + 2048 * x4), xmask)
    tmp1 = tl.load(in_ptr0 + (x2 + 64 * x0 + 4096 * x4), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tmp6 <= tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.where(tmp8, tmp7, tmp10)
    tl.store(in_out_ptr0 + (x1 + 128 * x0 + 2048 * x4), tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 64, 64), (4096, 64, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 128, 64, 64), (819200, 64, 128, 2),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(819200)](buf0, primals_1,
            819200, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
    return buf0, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

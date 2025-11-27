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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused__unsafe_index_convolution_mul_sub_tanh_0(in_ptr0,
    in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 331776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096 % 64
    x0 = xindex % 4096
    x4 = xindex // 4096
    x2 = xindex // 16384 % 256
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 + 1024 * x2), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(out_ptr0 + x5, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (32, 64, 256, 256), (4096, 64, 256, 1))
    assert_size_stride(primals_4, (64, 1, 1), (1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = torch.ops.aten.convolution.default(primals_3, primals_1, [2,
            2], [1, 1], [1, 1], False, False, 1).output
        assert_size_stride(buf0, (32, 64, 256, 256), (4096, 64, 256, 1))
        buf1 = empty_strided_cuda((32, 64, 256, 256), (4096, 64, 256, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused__unsafe_index_convolution_mul_sub_tanh_0[grid(331776)
            ](primals_4, buf0, primals_2, buf1, 331776, XBLOCK=512,
            num_warps=8, num_stages=1)
        del buf0
        del primals_2
    return buf1, primals_1, primals_3, primals_4


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, subtracts a bias term, and applies tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_4 = self.bias
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

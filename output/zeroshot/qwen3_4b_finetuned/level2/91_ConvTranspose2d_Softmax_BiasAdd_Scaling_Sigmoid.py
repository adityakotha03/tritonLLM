import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4096 % 128
    x0 = xindex % 4096
    x2 = xindex // 524288
    x4 = xindex
    tmp0 = tl.load(in_out_ptr0 + x4, xmask)
    tmp1 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4096 + x0 + 65536 * x2), xmask,
        eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = 0.5
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + x4, tmp6, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x0, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_2, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_3, (128,), (1,))
    assert_size_stride(primals_4, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_5, (1,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2, 
            2), padding=(1, 1), dilation=(1, 1), transposed=True,
            output_padding=(1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (128, 128, 66, 66), (524288, 4096, 64, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_add_mul_0[grid(1048576)](buf1, primals_3, primals_4,
            1048576, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_3
        del primals_4
        buf2 = empty_strided_cuda((128, 128, 66, 66), (524288, 1, 8, 128),
            torch.float32)
        extern_kernels.softmax(buf1, 1, out=buf2)
        del buf1
        triton_poi_fused_add_mul_1[grid(128000)](buf2, primals_5, 128000,
            XBLOCK=128, num_warps=4, num_stages=1)
        del primals_5
    return reinterpret_tensor(buf2, (128, 128, 64, 64), (524288, 1, 8, 128),
        0), primals_1, primals_2, primals_5


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_5 = self.scaling_factor
        primals_1 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4, primals_5])
        return output[0]

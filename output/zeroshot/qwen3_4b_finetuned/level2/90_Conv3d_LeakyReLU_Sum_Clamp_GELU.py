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
def triton_poi_fused_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 115351680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4096
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096 * x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x2 % 64), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (64 * x2 // 64 % 64), xmask, eviction_policy=
        'evict_last')
    tmp3 = tmp2 + tmp1
    tmp4 = 0.2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr1 + x2, tmp8, xmask)


@triton.jit
def triton_poi_fused_clamp_gelu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 115351680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + x2, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 1.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp4 * tmp6
    tmp8 = 0.7071067811865476
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 * tmp8
    tmp11 = tmp9 + tmp10
    tl.store(in_out_ptr0 + x2, tmp11, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (64, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (128, 8, 16, 64, 64), (8192, 1024, 64, 1,
        1))
    assert_size_stride(primals_4, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1,
            1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 64, 14, 62, 62), (518464, 8192, 39488,
            64, 1))
        buf1 = empty_strided_cuda((64,), (1,), torch.float32)
        buf16 = empty_strided_cuda((128, 64, 14, 62, 62), (5481728, 8192,
            39488, 64, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_leaky_relu_0[grid(115351680)](buf0, primals_2,
            buf1, buf16, 115351680, XBLOCK=1024, num_warps=4, num_stages=1)
        del buf0
        del primals_2
        buf2 = buf1
        del buf1
        triton_poi_fused_clamp_gelu_1[grid(115351680)](buf2, primals_4, 
            115351680, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_4
    return buf2, primals_1, primals_3, buf2, buf16


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.sum_tensor
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]

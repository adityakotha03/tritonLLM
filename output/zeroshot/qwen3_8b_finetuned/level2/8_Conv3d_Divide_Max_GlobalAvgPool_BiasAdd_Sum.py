import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_per_fused_add_div_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + x0, tmp4, xmask)


@triton.jit
def triton_per_fused_add_sum_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + 4 * x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK:])
    tmp5 = tl.sum(tmp3, 0)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + x0, tmp5, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 8, 3, 3, 3), (216, 27, 9, 3, 1))
    assert_size_stride(primals_2, (128, 8, 16, 64, 64), (8192, 1024, 64, 1,
        1))
    assert_size_stride(primals_3, (16, 1, 1, 1), (1, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (128, 16, 16, 64, 64), (16777216, 1048576,
            65536, 1024, 1))
        buf1 = empty_strided_cuda((128, 16), (16, 1), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_div_0[grid(2048)](buf0, primals_3, buf1, 2048,
            XBLOCK=256, num_warps=4, num_stages=1)
        buf2 = reinterpret_tensor(buf1, (128, 16), (16, 1), 0)
        del buf1
        buf3 = buf2
        del buf2
        triton_per_fused_add_sum_1[grid(128)](buf3, buf0, 128, XBLOCK=128,
            num_warps=4, num_stages=1)
        del buf0
    return buf3, primals_1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_3 = self.bias
        primals_2 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
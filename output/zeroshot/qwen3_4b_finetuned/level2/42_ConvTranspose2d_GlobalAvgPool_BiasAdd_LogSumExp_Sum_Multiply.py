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
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = xindex // 32768
    tmp0 = tl.load(in_out_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, xmask)


@triton.jit
def triton_poi_fused_add_logsumexp_mean_mul_sum_1(in_ptr0, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0 * 10.0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp6 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp10 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp3 = tl.sum(tmp4, 0)
    tmp5 = tl.sum(tmp6, 0)
    tmp7 = tl.sum(tmp8, 0)
    tmp9 = tl.sum(tmp10, 0)
    tmp11 = tmp3 + tmp5
    tmp12 = tmp7 + tmp9
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + x0, tmp13, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 512, 512), (2097152, 32768, 
        64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.conv_transpose_with_indices_backward(
            primals_3, primals_1, stride=(2, 2), padding=(0, 0), dilation=(
            1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None
        )
        assert_size_stride(buf0, (16, 128, 1023, 1023), (13312608, 1048576,
            1024, 1))
        buf1 = buf0
        del buf0
        buf3 = empty_strided_cuda((16, 128, 1023, 1023), (13312608, 1048576,
            1024, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_0[grid(20480)](buf1, primals_2, 20480, XBLOCK=
            256, num_warps=4, num_stages=1)
        del primals_2
        buf5 = empty_strided_cuda((16, 128, 1, 1), (128, 1, 128, 128), torch
            .float32)
        extern_kernels.sum(buf1, [2, 3, 4], out=buf5)
        buf6 = reinterpret_tensor(buf5, (16, 128, 1, 1), (128, 1, 1, 1),
            0)
        del buf5
        triton_poi_fused_add_logsumexp_mean_mul_sum_1[grid(1)](buf6, buf3, 
            1, XBLOCK=1, num_warps=2, num_stages=1)
    return reinterpret_tensor(buf3, (16, 1), (1, 1), 0), buf1, primals_1,
    primals_3, buf6


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, global average pooling, adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, 
            kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

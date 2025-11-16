import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_add_clamp_log_sum_exp_mul_0(in_ptr0, in_ptr1, out_ptr0,
    out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 577248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = xindex // 1024 % 16
    tmp0 = tl.load(in_ptr0 + x3, xmask)
    tmp1 = tl.load(in_ptr1 + x1, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = libdevice.logsumexp(tmp2)
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sigmoid(tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8 * tmp3
    tmp10 = 0.0
    tmp11 = tmp9 > tmp10
    tmp12 = tmp9 < tmp10
    tmp13 = tmp11.to(tl.float32)
    tmp14 = -1.0
    tmp15 = tl.where(tmp12, tmp14, tmp13)
    tmp16 = tl.where(tmp12, tmp14, tmp9)
    tmp17 = tmp14 > tmp16
    tmp18 = tl.where(tmp17, tmp14, tmp16)
    tl.store(out_ptr0 + x3, tmp3, xmask)
    tl.store(out_ptr1 + x3, tmp18, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3, 3), (81, 27, 9, 3, 1))
    assert_size_stride(primals_2, (16,), (1,))
    assert_size_stride(primals_3, (128, 3, 16, 32, 32), (16384, 5462, 32, 1,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (128,
            3, 16, 32, 32), (8192, 2736, 273, 9, 1), 0), reinterpret_tensor(
            primals_1, (3, 16, 3, 3, 3), (81, 3, 9, 3, 1), 0), stride=(2, 1,
            1, 1, 1), padding=(1, 0, 0, 0, 0), dilation=(1, 1, 1, 1, 1),
            transposed=True, output_padding=(0, 0, 0, 0, 0), groups=1,
            bias=None)
        assert_size_stride(buf0, (128, 16, 32, 32, 32), (16384, 1024, 32, 1,
            1))
        buf1 = empty_strided_cuda((128, 1, 32, 32, 32), (32768, 32768, 1, 1,
            1), torch.float32)
        buf2 = empty_strided_cuda((128, 1, 32, 32, 32), (32768, 32768, 1, 1,
            1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_clamp_log_sum_exp_mul_0[grid(577248)](buf0,
            primals_2, buf1, buf2, 577248, XBLOCK=256, num_warps=4,
            num_stages=1)
        del primals_2
    return reinterpret_tensor(buf1, (128, 1, 32, 32, 32), (32768, 1, 1, 1, 1),
        0), primals_1, primals_3, buf0, reinterpret_tensor(buf2, (128, 1,
        32, 32, 32), (32768, 1, 1, 1, 1), 0)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1)) 

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]

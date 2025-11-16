import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4239360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4800
    x1 = xindex // 4800
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4800 * x1), xmask)
    tl.store(out_ptr0 + x2, tmp0, xmask)


@triton.jit
def triton_poi_fused_add_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK:
    tl.constexpr):
    xnumel = 4239360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4800
    x1 = xindex // 4800
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1 + 4800 * x0), xmask, eviction_policy=
        'evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 + tmp0
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (16, 32, 16, 32, 32), (49152, 1508, 480, 15,
        1))
    assert_size_stride(primals_2, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_3, (64,), (1,))
    assert_size_stride(primals_4, (64, 1, 1, 1), (64, 64, 64, 64))
    assert_size_stride(primals_5, (64, 1, 1, 1), (64, 64, 64, 64))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(2,
            2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=True,
            output_padding=(1, 1, 1), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 17, 33, 33))
        buf1 = empty_strided_cuda((16, 32, 16, 32, 32), (49152, 1508, 480,
            15, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(4239360)](primals_1, buf1, 4239360,
            XBLOCK=512, num_warps=4, num_stages=1)
        del primals_1
        buf2 = empty_strided_cuda((16, 64, 17, 33, 33), (4239360, 64, 2400,
            72, 1), torch.float32)
        extern_kernels.addmm(primals_3, reinterpret_tensor(buf0, (4239360, 
            64), (64, 1), 0), reinterpret_tensor(primals_4, (64, 64), (64, 1
            ), 0), alpha=1, beta=1, out=buf2)
        del primals_3
        buf3 = buf0
        del buf0
        triton_poi_fused_add_mul_1[grid(4239360)](primals_5, buf2, buf3,
            4239360, XBLOCK=2048, num_warps=10, num_stages=1)
        del primals_5
    return buf3, primals_2, primals_4, reinterpret_tensor(buf0, (4239360, 64
        ), (64, 1), 0), buf2, buf1


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, 
            kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, input_0):
        primals_2 = self.conv_transpose.weight
        primals_3 = self.conv_transpose.bias
        primals_4 = self.bias
        primals_5 = torch.randn(64, 1, 1, 1,  out = torch.cuda._DeviceGuard(0)
            .current_raw_stream, dtype = torch.float32, device = 'cuda:0'
            )
        primals_5.requires_grad = False
        primals_5_1 = primals_5
        del primals_5
        output = call([input_0, primals_2, primals_3, primals_4, primals_5_1])
        return output[0]

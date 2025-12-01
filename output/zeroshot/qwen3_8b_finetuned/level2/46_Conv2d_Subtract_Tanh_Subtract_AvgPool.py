import torch
from torch._inductor.select_algorithm import extern_kernels
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
def triton_poi_fused_add_sub_tanh_sub_0(in_ptr0, in_ptr1, out_ptr0,
    xnumel, XBLOCK: tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + 1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = libdevice.tanh(tmp6)
    tmp10 = tmp9 - tmp8
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_add_sub_tanh_sub_1(in_ptr0, in_ptr1, out_ptr0, xnumel,
    XBLOCK: tl.constexpr):
    xnumel = 204800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + 0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + 1)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp6 = tmp3 - tmp5
    tmp9 = libdevice.tanh(tmp6)
    tmp10 = tmp9 - tmp8
    tl.store(out_ptr0 + x0, tmp10, xmask)


@triton.jit
def triton_poi_fused_add_sub_tanh_sub_2(in_ptr0, out_ptr0, xnumel, rnumel,
    XBLOCK: tl.constexpr):
    xnumel = 128
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128 * x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr0 + (r1 + 128 * x0), tmp6, xmask)


def call(args):
    (primals_1, primals_2, primals_3, primals_4) = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (128, 64, 128, 128), (1048576, 16384, 128,
        1))
    assert_size_stride(primals_4, (128, 128, 126, 126), (20736384, 159744, 
        126, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1,
            64, 128, 128), (1048576, 1, 8192, 64), 0), primals_1, stride=(1,
            1), padding=(1, 1), dilation=(1, 1), transposed=False,
            output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 128, 127, 127), (20736384, 159744, 126,
            1))
        del primals_1
        buf1 = empty_strided_cuda((128, 128, 127, 127), (20736384, 159744,
            126, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_sub_tanh_sub_0[grid(204800)](buf0, primals_2,
            buf1, 204800, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_2
        buf2 = empty_strided_cuda((128, 128, 126, 126), (20736384, 159744,
            126, 1), torch.float32)
        triton_poi_fused_add_sub_tanh_sub_1[grid(204800)](buf1, primals_4,
            buf2, 204800, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_4
        buf3 = empty_strided_cuda((128, 128, 126, 126), (20736384, 159744,
            126, 1), torch.float32)
        triton_poi_fused_add_sub_tanh_sub_2[grid(128)](buf2, buf3, 128,
            64, XBLOCK=128, num_warps=4, num_stages=1)
        del buf2
    return buf3, reinterpret_tensor(buf0, (1, 128, 127, 127), (20736384, 1,
        126, 1), 0), primals_3, buf1


class ModelNew(nn.Module):
    """
    Model that performs a convolution, subtraction, tanh activation, subtraction and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, input_0):
        primals_1 = self.conv.weight
        primals_2 = self.conv.bias
        primals_4 = self.avgpool.weight
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3, primals_4])
        return output[0]
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
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 1024 % 64
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x1 + 1024 * x2), xmask)
    tl.store(out_ptr0 + x3, tmp0, xmask)


@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 6912
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x2), xmask, other=0.0)
    tmp1 = triton_helpers.maximum(tmp0, tmp0, 1)
    tmp2 = tmp0 - tmp1
    tmp3 = tl_math.exp(tmp2)
    tl.store(out_ptr0 + (r1 + 64 * x2), tmp3, xmask)


@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK:
    tl.constexpr):
    xnumel = 6912
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64 * x2), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp0 / tmp4
    tl.store(out_ptr0 + (r1 + 64 * x2), tmp5, xmask)


@triton.jit
def triton_poi_fused_sigmoid_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr
    ):
    xnumel = 1119744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + x0, tmp1, xmask)


def call(args):
    (primals_1, primals_2, primals_3) = args
    args.clear()
    assert_size_stride(primals_1, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 32, 16, 32, 32), (163840, 5120, 320, 
        10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_2[grid(4194304)](primals_1, buf0, 4194304,
            XBLOCK=128, num_warps=8, num_stages=1)
        del primals_1
        buf1 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_3, (2048,
            1024), (1024, 1), 0), reinterpret_tensor(buf0, (1024, 2048), (
            1, 1024), 0), alpha=1, beta=1, out=buf1)
        del primals_2
        buf2 = empty_strided_cuda((16, 64, 36, 36, 36), (41472, 648, 18, 1,
            1), torch.float32)
        buf3 = reinterpret_tensor(buf2, (6912, 64), (4096, 1), 0)
        del buf2
        triton_poi_fused__softmax_0[grid(6912)](buf1, buf3, 6912, 64,
            XBLOCK=64, num_warps=4, num_stages=1)
        buf4 = reinterpret_tensor(buf1, (6912, 64), (4096, 1), 0)
        del buf1
        triton_poi_fused__softmax_1[grid(6912)](buf3, buf4, 6912, 64,
            XBLOCK=64, num_warps=4, num_stages=1)
        buf5 = empty_strided_cuda((16, 64, 36, 36, 36), (41472, 648, 18, 1,
            1), torch.float32)
        triton_poi_fused_sigmoid_3[grid(1119744)](buf4, buf5, 1119744,
            XBLOCK=256, num_warps=4, num_stages=1)
        del buf4
    return buf5, reinterpret_tensor(primals_3, (2048, 1024), (1024, 1), 0
        ), buf0, buf3


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_0):
        primals_1 = self.conv_transpose.weight
        primals_2 = self.conv_transpose.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
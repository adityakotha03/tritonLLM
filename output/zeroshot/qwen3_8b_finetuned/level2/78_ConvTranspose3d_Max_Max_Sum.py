import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_max_pool2d_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 512
    x1 = xindex // 512
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16 * x1 + 16 * 16 * x2)), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 1 + (16 * x1 + 16 * 16 * x2)), xmask, other=
        -inf)
    tmp3 = tl.load(in_ptr0 + (x0 + 2 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp5 = tl.load(in_ptr0 + (x0 + 3 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp7 = tl.load(in_ptr0 + (x0 + 4 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp9 = tl.load(in_ptr0 + (x0 + 5 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp11 = tl.load(in_ptr0 + (x0 + 6 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp13 = tl.load(in_ptr0 + (x0 + 7 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.where(xmask, tmp1, tmp2)
    tmp6 = tl.where(xmask, tmp3, tmp4)
    tmp8 = tl.where(xmask, tmp5, tmp6)
    tmp10 = tl.where(xmask, tmp7, tmp8)
    tmp12 = tl.where(xmask, tmp9, tmp10)
    tmp14 = tl.where(xmask, tmp11, tmp12)
    tmp15 = tl.where(xmask, tmp13, tmp14)
    tl.store(out_ptr0 + x0, tmp15, xmask)


@triton.jit
def triton_poi_fused_max_pool3d_1(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: 
    constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x1 = xindex // 128
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16 * x1 + 16 * 16 * x2)), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 1 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp3 = tl.load(in_ptr0 + (x0 + 2 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp5 = tl.load(in_ptr0 + (x0 + 3 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp7 = tl.load(in_ptr0 + (x0 + 4 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp9 = tl.load(in_ptr0 + (x0 + 5 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp11 = tl.load(in_ptr0 + (x0 + 6 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp13 = tl.load(in_ptr0 + (x0 + 7 + (16 * x1 + 16 * 16 * x2)), xmask, other
        =-inf)
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.where(xmask, tmp1, tmp2)
    tmp6 = tl.where(xmask, tmp3, tmp4)
    tmp8 = tl.where(xmask, tmp5, tmp6)
    tmp10 = tl.where(xmask, tmp7, tmp8)
    tmp12 = tl.where(xmask, tmp9, tmp10)
    tmp14 = tl.where(xmask, tmp11, tmp12)
    tmp15 = tl.where(xmask, tmp13, tmp14)
    tl.store(out_ptr0 + x0, tmp15, xmask)


@triton.jit
def triton_poi_fused_sum_2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl.
    constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x1 = xindex // 64
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1 * x1 + 1 * 5 * x2)), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 1 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp3 = tl.load(in_ptr0 + (x0 + 2 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp5 = tl.load(in_ptr0 + (x0 + 3 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (x0 + 4 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp9 = tl.load(in_ptr0 + (x0 + 5 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp11 = tl.load(in_ptr0 + (x0 + 6 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr0 + (x0 + 7 + (1 * x1 + 1 * 5 * x2)), xmask, other=0.0)
    tmp2 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tmp1 + tmp2
    tmp6 = tmp3 + tmp4
    tmp8 = tmp5 + tmp6
    tmp10 = tmp7 + tmp8
    tmp12 = tmp9 + tmp10
    tmp14 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + x0, tmp15, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 32, 32, 32, 32), (32768, 1024, 32, 1, 1))
    assert_size_stride(arg1_1, (64, 32, 32, 32, 32), (32768, 512, 16, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 32, 32, 32), (131072, 2048, 64, 2, 1),
            torch.float32)
        buf1 = empty_strided_cuda((16, 64, 16, 16, 16), (4096, 64, 16, 2, 1),
            torch.float32)
        buf2 = empty_strided_cuda((16, 64, 5, 5, 5), (125, 20, 5, 1, 1),
            torch.float32)
        buf3 = empty_strided_cuda((16, 1, 5, 5, 5), (125, 25, 5, 1, 1),
            torch.float32)
        get_raw_stream(0)
        triton_poi_fused_max_pool2d_0[grid(2048)](arg1_1, buf1, 2048, 512,
            XBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        triton_poi_fused_max_pool3d_1[grid(2048)](buf1, buf2, 2048, 128,
            XBLOCK=128, num_warps=4, num_stages=1)
        triton_poi_fused_sum_2[grid(4096)](buf2, buf3, 4096, 64, XBLOCK=128,
            num_warps=4, num_stages=1)
    return buf3, arg0_1, buf0, buf1, buf2, buf3


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(32, 64, kernel_size=(5, 5, 5),
            stride=(2, 2, 2), padding=(2, 2, 2))

    def forward(self, input_0):
        arg0_1 = input_0
        arg1_1 = self.conv_transpose(arg0_1)
        output = call([arg0_1, arg1_1])
        return output[0]
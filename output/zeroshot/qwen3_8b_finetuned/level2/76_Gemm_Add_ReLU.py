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
def triton_poi_fused_gemm_add_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    ynumel, xoffset, yoffset, rnumel, XBLOCK: tl.constexpr, YBLOCK:
    tl.constexpr):
    xnumel = xnumel
    ynumel = ynumel
    yoffset = yoffset
    yindex = tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xindex = tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, rnumel)
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    r0 = rindex
    y0 = yindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (y0 * 8192) + (r0 * 1024)), xmask &
        ymask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0 + (y0 * 8192) + (x0 * 8192)), xmask &
        ymask, other=0.0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, YBLOCK, rnumel])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp5, [rnumel, XBLOCK, YBLOCK])
    tmp8 = tl.where(ymask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 2)[:, :, None]
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, YBLOCK, rnumel])
    tmp11 = tl.where(xmask & ymask, tmp10, 0)
    tmp12 = tl.where(xmask & ymask, tmp11, 0)
    tl.store(out_ptr0 + (y0 + (x0 * 8192) + (r0 * 1024)), tmp12, xmask &
        ymask)


@triton.jit
def triton_poi_fused_add_1(in_ptr0, in_ptr1, out_ptr0, xnumel, xoffset,
    XBLOCK: tl.constexpr):
    xnumel = xnumel
    xoffset = xoffset
    xindex = tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = tl.full([XBLOCK, 1], 0, tl.int32)
    tmp0 = tl.load(in_ptr0 + (x0 + (y0 * 8192) + (x0 * 1024)), xmask, other=
        0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (y0 * 8192) + (x0 * 1024)), xmask, other=
        0.0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + (y0 * 8192) + (x0 * 1024)), tmp2, xmask)


@triton.jit
def triton_poi_fused_relu_2(in_ptr0, out_ptr0, xnumel, xoffset, XBLOCK:
    tl.constexpr):
    xnumel = xnumel
    xoffset = xoffset
    xindex = tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    y0 = tl.full([XBLOCK, 1], 0, tl.int32)
    tmp0 = tl.load(in_ptr0 + (x0 + (y0 * 8192) + (x0 * 1024)), xmask, other=
        0.0)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 > tmp1
    tmp3 = tl.full(tmp2.shape, 0, tl.int32)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.store(out_ptr0 + (x0 + (y0 * 8192) + (x0 * 1024)), tmp4, xmask)


def call(args):
    (primals_1, primals_2, primals_3) = args
    args.clear()
    assert_size_stride(primals_1, (8192, 8192), (8192, 1))
    assert_size_stride(primals_2, (1024, 8192), (8192, 1))
    assert_size_stride(primals_3, (8192,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gemm_add_relu_0[grid(1024, 8192, 1024)](primals_2,
            primals_1, buf0, 1024, 8192, 0, 0, 8192, XBLOCK=256, YBLOCK=256)
        del primals_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_add_1[grid(1024, 1, 1024)](buf0, primals_3,
            buf1, 1024, 0, XBLOCK=256)
        del primals_3
        del buf0
        buf2 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        triton_poi_fused_relu_2[grid(1024, 1, 1024)](buf1, buf2, 1024, 0,
            XBLOCK=256)
        del buf1
    return buf2, primals_2, primals_3


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0):
        primals_1 = self.gemm.weight
        primals_2 = self.gemm.weight
        primals_3 = self.bias
        output = call([primals_1, primals_2, primals_3, input_0])
        return output[0]
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
empty_cuda = torch._C._dynamo.guards._empty_cuda


@triton.jit
def triton_poi_fused_matmul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, rnumel, XBLOCK: tl.constexpr):
    xnumel = 2048
    ynumel = 2048
    yoffset = tl.program_id(1) * XBLOCK
    y = yoffset + tl.arange(0, XBLOCK)[None, :]
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * YBLOCK
    xindex = xoffset + tl.arange(0, YBLOCK)[:, None]
    tl.full([YBLOCK, XBLOCK], True, tl.int1)
    rindex = tl.arange(0, rnumel)
    tl.full([YBLOCK, XBLOCK], True, tl.int1)
    r0 = rindex
    y0 = y
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0 + 4096 * y0), r0 <= 4095, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (r0 + 4096 * x0), r0 <= 4095, eviction_policy=
        'evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [YBLOCK, XBLOCK])
    tmp5 = tl.where(r0 <= 4095, tmp3, 0)
    tmp6 = tl.sum(tmp5, 0)
    tl.store(out_ptr0 + (y0 + 2048 * x0), tmp6, None)


def triton_poi_fused_matmul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1024
    ynumel = 1024
    yoffset = tl.program_id(1) * XBLOCK
    y = yoffset + tl.arange(0, XBLOCK)[None, :]
    tl.full([XBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * YBLOCK
    xindex = xoffset + tl.arange(0, YBLOCK)[:, None]
    tl.full([YBLOCK, XBLOCK], True, tl.int1)
    rindex = tl.arange(0, rnumel)
    tl.full([YBLOCK, XBLOCK], True, tl.int1)
    r0 = rindex
    y0 = y
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0 + 4096 * y0), r0 <= 4095, eviction_policy=
        'evict_last')
    tmp1 = tl.load(in_ptr1 + (r0 + 4096 * x0), r0 <= 4095, eviction_policy=
        'evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [YBLOCK, XBLOCK])
    tmp5 = tl.where(r0 <= 4095, tmp3, 0)
    tmp6 = tl.sum(tmp5, 0)
    tl.store(out_ptr0 + (y0 + 2048 * x0), tmp6, None)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2048, 1024), (1024, 1))
    assert_size_stride(arg1_1, (2048, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
        buf1 = empty_strided_cuda((2048, 1024), (1024, 1), torch.float32)
        del arg1_1
        get_raw_stream(0)
        triton_poi_fused_matmul_0[grid(2048, 2048, 4096)](arg0_1, buf0, buf1,
            2048, 2048, 0, 0, 4096, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
        buf2 = empty_strided_cuda((1024, 2048), (2048, 1), torch.float32)
        triton_poi_fused_matmul_1[grid(1024, 1024, 4096)](buf0, arg1_1, buf2,
            1024, 1024, 0, 0, 4096, XBLOCK=128, num_warps=4, num_stages=1)
    return buf2, buf1


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_0, input_1):
        arg0_1 = input_0
        arg1_1 = input_1
        output = call([arg0_1, arg1_1])
        return output[0]
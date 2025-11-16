import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 2097152
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32768
    y1 = yindex // 32768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32768 * x2 + 16777216 * y1), xmask & ymask
        , eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 2097152
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32768
    y1 = yindex // 32768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32768 * x2 + 16777216 * y1), xmask & ymask
        , eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp0, xmask & ymask)


@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl
    .constexpr, XBLOCK: tl.constexpr):
    ynumel = 2097152
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32768
    y1 = yindex // 32768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 32768 * x2 + 16777216 * y1), xmask & ymask
        , eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp0, xmask & ymask)


def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 32, 512, 1024), (16777216, 524288, 
        1024, 1))
    assert_size_stride(arg1_1, (32, 32, 512, 1024), (16777216, 524288, 
        1024, 1))
    assert_size_stride(arg2_1, (32, 32, 512, 1024), (16777216, 524288, 
        1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288, 
            1024, 1), torch.float16)
        get_raw_stream(0)
        triton_poi_fused_clone_0[grid(2097152, 512)](arg0_1, buf0, 2097152,
            512, XBLOCK=32, YBLOCK=256, num_warps=4, num_stages=1)
        buf1 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288, 
            1024, 1), torch.float16)
        triton_poi_fused_clone_1[grid(2097152, 512)](arg1_1, buf1, 2097152,
            512, XBLOCK=64, YBLOCK=128, num_warps=4, num_stages=1)
        del arg1_1
        buf2 = empty_strided_cuda((32, 32, 512, 1024), (16777216, 524288, 
            1024, 1), torch.float16)
        triton_poi_fused_clone_2[grid(2097152, 512)](arg2_1, buf2, 2097152,
            512, XBLOCK=32, YBLOCK=256, num_warps=4, num_stages=1)
        del arg2_1
    return buf0, buf1, buf2, arg0_1


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_0, input_1, input_2):
        arg0_1 = input_0
        arg1_1 = input_1
        arg2_1 = input_2
        output = call([arg0_1, arg1_1, arg2_1])
        return output[0]

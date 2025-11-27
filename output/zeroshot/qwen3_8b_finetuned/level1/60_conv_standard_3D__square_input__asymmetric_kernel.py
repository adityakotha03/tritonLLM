import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK: tl
    .constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1 + 4096 * x2), xmask, eviction_policy
        = 'evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0 + 64 * x1 + 4096 * x2), tmp2, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 64, 64, 64, 64), (262144, 4096, 64, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 64, 64, 64), (262144, 4096, 64, 1, 1
            ), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(16)](arg0_1, buf0, 16, 1, XBLOCK
            =128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(3, 64, (3, 5, 7), stride=(1, 1, 1), bias=False
            )

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
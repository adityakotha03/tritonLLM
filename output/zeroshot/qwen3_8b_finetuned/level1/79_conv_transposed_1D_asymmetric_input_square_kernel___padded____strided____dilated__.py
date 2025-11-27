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
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = xindex // 64
    x4 = x3 % 131072
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 * 96 + x4), xmask)
    tmp1 = tl.load(in_ptr1 + x2, xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + x1, tmp4, xmask)


def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (64, 96), (96, 1))
    assert_size_stride(primals_2, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 64, 262148), (16384, 256, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(268435456)](primals_1, primals_2, buf0,
            268435456, XBLOCK=128, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
    return buf0,


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn((64, 96), dtype=torch.float32, device='cuda'))
        self.bias = nn.Parameter(torch.randn((64,), dtype=torch.float32, device='cuda'))

    def forward(self, input_0):
        arg0_1 = input_0
        arg0_2 = self.weight
        arg0_3 = self.bias
        output = call([arg0_1, arg0_2, arg0_3])
        return output[0]
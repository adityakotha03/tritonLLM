import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel, ynumel,
    xoffset, yoffset, ynumel2, rnumel):
    xnumel = 8
    YBLOCK: tl.constexpr = 128
    rnumel = 45
    RBLOCK: tl.constexpr = 45
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * xnumel
    xindex = xoffset + tl.arange(0, xnumel)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, YBLOCK], True, tl.int1)
    r0 = rindex
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (r0 + y0 * 5), rmask=r0 < rnumel, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1 + y0 * 508), xmask=xmask, ymask=ymask,
        other=0.0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x1 + y0 * 508), tmp2, xmask, ymask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (45,), (1,))
    assert_size_stride(primals_2, (8, 32, 512, 512), (512*512*32, 512*512, 512, 1))
    assert_size_stride(primals_3, (64,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 64, 508, 504), (508*504*64, 504*64, 64, 1),
            torch.float16)
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(8, 45, 128, 128)](primals_1,
            primals_2, buf0, 8, 45, 0, 0, 45, 45, num_warps=4, num_stages=1)
        del primals_1
        del primals_2
    return buf0, primals_3


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._convolution = torch.nn.Conv2d(32, 64, (5, 9), stride=(1, 1),
            padding=(0, 0), dilation=(1, 1), groups=1, bias=True,
            padding_mode='zeros')
        self._convolution.weight = torch.nn.Parameter(
            torch.randn((64, 32, 5, 9), dtype=torch.float16, device='cuda'))
        self._convolution.bias = torch.nn.Parameter(
            torch.randn((64,), dtype=torch.float16, device='cuda'))
    
    def forward(self, input_0):
        primals_1 = self._convolution.weight
        primals_2 = input_0
        primals_3 = self._convolution.bias
        output = call([primals_1, primals_2, primals_3])
        return output[0]
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn.functional as F
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, in_ptr1, out_ptr0, xnumel,
    rnumel, XBLOCK: tl.constexpr):
    xnumel = 2560000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex % 512
    x0 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = libdevice.maximum(tmp2, tmp3)
    tl.store(out_ptr0 + x2, tmp4, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 1, 1), (4096, 64, 1, 1))
    assert_size_stride(primals_2, (1, 128, 1024, 1024), (131072, 1024, 1, 1))
    assert_size_stride(primals_3, (128,), (1,))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 128, 1024, 1024), (131072, 1024, 1, 1),
            torch.float32)
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(2560000)](primals_2, primals_1,
            buf1, 2560000, 128, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
    return buf1, primals_2, primals_3


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1,
            1, dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, input_0):
        primals_1 = self.weight
        primals_2 = input_0
        primals_3 = self.bias
        output = call([primals_1, primals_2, primals_3])
        return output[0]
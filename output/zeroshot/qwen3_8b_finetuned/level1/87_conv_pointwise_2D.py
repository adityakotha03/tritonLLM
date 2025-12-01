import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_convolution_relu_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024 * x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r1, None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + r1, None, eviction_policy='evict_last')
    tmp3 = tmp0 * tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.full([1, 1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (r1 + 1024 * x0), tmp6, xmask)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_2, (128,), (1,))
    assert_size_stride(primals_3, (16, 64, 1024, 1024), (68710912, 1024, 1,
        1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1048576, 128), (128, 1), torch.float32)
        extern_kernels.mm(reinterpret_tensor(primals_3, (1048576, 64), (64,
            1), 0), reinterpret_tensor(primals_1, (64, 128), (1, 64), 0),
            out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((16, 128, 1024, 1024), (134217728, 1024,
            1, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_convolution_relu_0[grid(1048576)](buf0, primals_2,
            primals_2, buf1, 1048576, 128, XBLOCK=256, num_warps=4, num_stages
            =1)
        del buf0
        del primals_2
    return buf1, reinterpret_tensor(primals_3, (1048576, 64), (64, 1), 0)


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
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        
    def forward(self, input_0):
        primals_1 = self.conv1d.weight
        primals_2 = self.conv1d.bias
        primals_3 = input_0
        output = call([primals_1, primals_2, primals_3])
        return output[0]
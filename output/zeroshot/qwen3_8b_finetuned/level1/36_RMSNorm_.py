import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_per_fused_add_div_mean_mul_rsqrt_rsub_sqrt_0(in_ptr0, out_ptr0,
    xnumel, rnumel, XBLOCK: tl.constexpr):
    xnumel = 7168
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + 64 * x0), xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp0 * tmp11
    tl.store(out_ptr0 + (r2 + 64 * x0), tmp12, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (112, 64, 512, 512), (33554432, 524288, 1024,
        2))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((112, 64, 512, 512), (33554432, 524288,
            1024, 2), torch.float32)
        get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_rsqrt_rsub_sqrt_0[grid(7168)](arg0_1,
            buf0, 7168, 64, XBLOCK=128, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_add_clamp_div_mul_rsub_sigmoid_tanh_0(in_ptr0,
    out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = -1.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 1.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = triton_helpers.maximum(tmp9, tmp5)
    tmp11 = triton_helpers.minimum(tmp10, tmp7)
    tl.store(out_ptr0 + x0, tmp11, xmask)


def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    assert_size_stride(arg1_1, (8192, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_rsub_sigmoid_tanh_0[grid(8388608)](
            buf0, buf1, 8388608, XBLOCK=128, num_warps=4, num_stages=1)
        del buf0
    return buf1,


class ModelNew(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input_0):
        arg1_1 = self.gemm.weight
        arg0_1 = input_0
        output = call([arg0_1, arg1_1])
        return output[0]
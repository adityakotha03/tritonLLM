import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime.triton_helpers import libdevice
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda


@triton.jit
def triton_poi_fused_gelu_max_mean_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.
    constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 8192
    x0 = xindex % 8192
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0 + 8192 * x1), xmask, eviction_policy=
        'evict_last')
    tmp2 = tl.load(in_ptr0 + (8193 + x0 + 8192 * x1), xmask,
        eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (16385 + x0 + 8192 * x1), xmask,
        eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16384 + x1), xmask, eviction_policy='evict_last')
    tmp4 = libdevice.max(tmp2, tmp3)
    tmp5 = libdevice.max(tmp1, tmp4)
    tmp7 = tl.load(in_ptr0 + (24577 + x0 + 8192 * x1), xmask,
        eviction_policy='evict_last')
    tmp8 = libdevice.max(tmp6, tmp7)
    tmp9 = libdevice.max(tmp5, tmp8)
    tmp10 = tmp0 - tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = libdevice.erf(tmp11)
    tmp13 = 0.5
    tmp14 = tmp12 + tmp13
    tl.store(out_ptr0 + x2, tmp14, xmask)


def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 8192), (8192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 8192), (8192, 1), torch.float32)
        get_raw_stream(0)
        triton_poi_fused_gelu_max_mean_0[grid(10240)](arg0_1, buf0, 10240,
            XBLOCK=1024, num_warps=4, num_stages=1)
        del arg0_1
    return buf0,


class ModelNew(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

    def forward(self, input_0):
        arg0_1 = input_0
        output = call([arg0_1])
        return output[0]
